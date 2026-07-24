// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/properties.hpp"

#include "openvino/genai/visibility.hpp"

namespace ov::genai {

// LTX2's audio VAE decoder: decodes a mel-spectrogram latent into a mel spectrogram (still not a
// waveform - LTX2Vocoder does that final step). Decoder-only, matching the text-to-video scope.
class OPENVINO_GENAI_EXPORTS AutoencoderKLLTX2Audio {
public:
    struct OPENVINO_GENAI_EXPORTS Config {
        size_t latent_channels = 8;
        size_t mel_bins = 64;
        size_t mel_compression_ratio = 4;
        size_t temporal_compression_ratio = 4;
        size_t sample_rate = 16000;
        size_t mel_hop_length = 160;
        std::vector<float> latents_mean_data, latents_std_data;

        explicit Config(const std::filesystem::path& config_path);
    };

    explicit AutoencoderKLLTX2Audio(const std::filesystem::path& vae_decoder_path);

    AutoencoderKLLTX2Audio(const std::filesystem::path& vae_decoder_path,
                           const std::string& device,
                           const ov::AnyMap& properties = {});

    AutoencoderKLLTX2Audio& compile(const std::string& device, const ov::AnyMap& properties = {});

    AutoencoderKLLTX2Audio clone();

    ov::Tensor decode(const ov::Tensor& latent);

    const Config& get_config() const;

    AutoencoderKLLTX2Audio& reshape(int64_t batch_size);

private:
    Config m_config;
    ov::InferRequest m_decoder_request;
    std::shared_ptr<ov::Model> m_decoder_model;
};

}  // namespace ov::genai
