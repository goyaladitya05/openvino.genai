// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/genai/lora_adapter.hpp"
#include "openvino/genai/visibility.hpp"

namespace ov::genai {

// Joint video+audio diffusion transformer used by LTX2. Unlike LTXVideoTransformer3DModel, a
// single inference produces noise predictions for BOTH modalities from BOTH latent streams.
class OPENVINO_GENAI_EXPORTS LTX2VideoTransformer3DModel {
public:
    struct OPENVINO_GENAI_EXPORTS Config {
        size_t in_channels = 8;
        size_t patch_size = 1;
        size_t patch_size_t = 1;

        size_t audio_in_channels = 128;
        size_t audio_patch_size = 1;
        size_t audio_patch_size_t = 1;

        size_t num_attention_heads = 32;
        size_t attention_head_dim = 128;
        size_t audio_num_attention_heads = 32;
        size_t audio_attention_head_dim = 64;

        size_t pos_embed_max_pos = 20;
        size_t audio_pos_embed_max_pos = 20;
        size_t base_height = 2048;
        size_t base_width = 2048;

        float rope_theta = 10000.0f;
        size_t causal_offset = 1;

        // Video VAE per-axis (frame, height, width) compression ratios, used to convert latent
        // patch coordinates into pixel-space coordinates for rope.
        std::vector<size_t> vae_scale_factors = {8, 32, 32};
        size_t audio_scale_factor = 4;

        size_t audio_sampling_rate = 16000;
        size_t audio_hop_length = 160;

        explicit Config(const std::filesystem::path& config_path);
    };

    struct Output {
        ov::Tensor video;
        ov::Tensor audio;
    };

    explicit LTX2VideoTransformer3DModel(const std::filesystem::path& root_dir);

    LTX2VideoTransformer3DModel(const std::filesystem::path& root_dir,
                                const std::string& device,
                                const ov::AnyMap& properties = {});

    LTX2VideoTransformer3DModel(const LTX2VideoTransformer3DModel&);

    LTX2VideoTransformer3DModel clone();

    const Config& get_config() const;

    LTX2VideoTransformer3DModel& compile(const std::string& device, const ov::AnyMap& properties = {});

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<LTX2VideoTransformer3DModel&, Properties...> compile(const std::string& device,
                                                                                        Properties&&... properties) {
        return compile(device, ov::AnyMap{std::forward<Properties>(properties)...});
    }

    void set_adapters(const std::optional<AdapterConfig>& adapters);

    /// @brief Runs inference. 'timestep' is a per-batch rank-1 [B] tensor; it is adapted internally
    /// to the compiled model's schema (legacy rank-1 'timestep', or rank-2 [B, S] per-token
    /// 'timestep' plus a rank-1 'audio_timestep' for current exports).
    Output infer(const ov::Tensor& hidden_states,
                const ov::Tensor& audio_hidden_states,
                const ov::Tensor& encoder_hidden_states,
                const ov::Tensor& audio_encoder_hidden_states,
                const ov::Tensor& timestep,
                const ov::Tensor& encoder_attention_mask,
                const ov::Tensor& audio_encoder_attention_mask,
                size_t num_frames,
                size_t height,
                size_t width,
                float fps,
                size_t audio_num_frames,
                const ov::Tensor& video_coords,
                const ov::Tensor& audio_coords);

    LTX2VideoTransformer3DModel& reshape(int64_t batch_size);

    size_t get_expected_batch_size() const;

private:
    Config m_config;
    AdapterController m_adapter_controller;
    std::string m_lora_prefix;
    ov::InferRequest m_request;
    std::shared_ptr<ov::Model> m_model;
    size_t m_expected_batch_size = 0;
    size_t m_timestep_rank = 1;
    bool m_has_audio_timestep = false;
};

}  // namespace ov::genai
