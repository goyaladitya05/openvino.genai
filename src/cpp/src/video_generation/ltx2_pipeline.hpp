// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

#include "image_generation/numpy_utils.hpp"
#include "image_generation/threaded_callback.hpp"
#include "json_utils.hpp"
#include "lora/helper.hpp"
#include "openvino/genai/image_generation/gemma3_text_encoder_model.hpp"
#include "openvino/genai/video_generation/autoencoder_kl_ltx_video.hpp"
#include "openvino/genai/video_generation/autoencoder_kl_ltx2_audio.hpp"
#include "openvino/genai/video_generation/ltx2_text_connectors.hpp"
#include "openvino/genai/video_generation/ltx2_video_transformer_3d_model.hpp"
#include "openvino/genai/video_generation/ltx2_vocoder.hpp"
#include "generation_config_utils.hpp"
#include "logger.hpp"
#include "video_generation/video_pipeline.hpp"
#include "video_generation/video_math_utils.hpp"

#include "utils.hpp"

using namespace ov::genai;

namespace {

const VideoGenerationConfig LTX2_DEFAULT_CONFIG = VideoGenerationConfig{
    std::nullopt,            // negative_prompt
    1,                       // num_videos_per_prompt
    nullptr,                 // generator
    4.0f,                    // guidance_scale
    512,                     // height
    768,                     // width
    40,                      // num_inference_steps
    1024,                    // max_sequence_length
    0.0f,                    // guidance_rescale
    121,                     // num_frames
    24.0f,                   // frame_rate
    std::nullopt,            // taylorseer_config
    std::nullopt,            // adapters
    std::nullopt,            // audio_guidance_scale (falls back to guidance_scale)
    std::nullopt             // audio_guidance_rescale (falls back to guidance_rescale)
};

void replace_ltx2_defaults(VideoGenerationConfig& config) {
    if (-1 == config.height) {
        config.height = LTX2_DEFAULT_CONFIG.height;
    }
    if (-1 == config.width) {
        config.width = LTX2_DEFAULT_CONFIG.width;
    }
    if (-1 == config.num_inference_steps) {
        config.num_inference_steps = LTX2_DEFAULT_CONFIG.num_inference_steps;
    }
    if (-1 == config.max_sequence_length) {
        config.max_sequence_length = LTX2_DEFAULT_CONFIG.max_sequence_length;
    }
    if (!config.guidance_rescale.has_value()) {
        config.guidance_rescale = LTX2_DEFAULT_CONFIG.guidance_rescale;
    }
    if (0 == config.num_frames) {
        config.num_frames = LTX2_DEFAULT_CONFIG.num_frames;
    }
    if (!config.frame_rate.has_value()) {
        config.frame_rate = LTX2_DEFAULT_CONFIG.frame_rate;
    }
}

// Simple (non-patched) audio latent packing: [B, C, L, M] -> [B, L, C * M]. LTX2's audio latents
// are packed with an implicit patch_size = M (all mel bins form one patch) and patch_size_t = 1.
ov::Tensor pack_audio_latents(const ov::Tensor& latents) {
    const ov::Shape shape = latents.get_shape();
    OPENVINO_ASSERT(shape.size() == 4, "pack_audio_latents expects [B, C, L, M]");
    const size_t B = shape[0], C = shape[1], L = shape[2], M = shape[3];

    ov::Tensor result(latents.get_element_type(), {B, L, C * M});
    const float* src = latents.data<const float>();
    float* dst = result.data<float>();
    for (size_t b = 0; b < B; ++b) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t l = 0; l < L; ++l) {
                const float* src_row = src + ((b * C + c) * L + l) * M;
                float* dst_row = dst + (b * L + l) * (C * M) + c * M;
                std::memcpy(dst_row, src_row, M * sizeof(float));
            }
        }
    }
    return result;
}

// Inverse of pack_audio_latents: [B, S, D] = [B, L, C * M] -> [B, C, L, M]
ov::Tensor unpack_audio_latents(const ov::Tensor& latents, size_t num_mel_bins) {
    const ov::Shape shape = latents.get_shape();
    OPENVINO_ASSERT(shape.size() == 3, "unpack_audio_latents expects [B, S, D]");
    const size_t B = shape[0], L = shape[1], D = shape[2];
    OPENVINO_ASSERT(D % num_mel_bins == 0, "D must be divisible by num_mel_bins");
    const size_t C = D / num_mel_bins;

    ov::Tensor result(latents.get_element_type(), {B, C, L, num_mel_bins});
    const float* src = latents.data<const float>();
    float* dst = result.data<float>();
    for (size_t b = 0; b < B; ++b) {
        for (size_t l = 0; l < L; ++l) {
            for (size_t c = 0; c < C; ++c) {
                const float* src_row = src + (b * L + l) * D + c * num_mel_bins;
                float* dst_row = dst + ((b * C + c) * L + l) * num_mel_bins;
                std::memcpy(dst_row, src_row, num_mel_bins * sizeof(float));
            }
        }
    }
    return result;
}

// Denormalizes packed audio latents [B, S, D] in-place: latents = latents * std + mean (no
// scaling_factor division, unlike video's denormalize_latents - see LTX2's
// _denormalize_audio_latents).
void denormalize_audio_latents_inplace(ov::Tensor& latents, const std::vector<float>& mean, const std::vector<float>& std_data) {
    const ov::Shape shape = latents.get_shape();
    OPENVINO_ASSERT(shape.size() == 3, "denormalize_audio_latents_inplace expects [B, S, D]");
    const size_t B = shape[0], S = shape[1], D = shape[2];
    OPENVINO_ASSERT(mean.size() == D && std_data.size() == D,
                    "audio latents_mean/std size (", mean.size(), ") does not match feature dim (", D, ")");

    float* data = latents.data<float>();
    for (size_t b = 0; b < B; ++b) {
        for (size_t s = 0; s < S; ++s) {
            float* row = data + (b * S + s) * D;
            for (size_t d = 0; d < D; ++d) {
                row[d] = row[d] * std_data[d] + mean[d];
            }
        }
    }
}

// Per-patch [start, end) pixel-space coordinates for the video rope, matching
// LTX2AudioVideoRotaryPosEmbed.prepare_video_coords. Since the transformer's video patch_size /
// patch_size_t are both 1, this reduces to a plain per-token (f, h, w) grid.
ov::Tensor prepare_video_coords(size_t batch_size,
                                size_t latent_num_frames,
                                size_t latent_height,
                                size_t latent_width,
                                const std::vector<size_t>& vae_scale_factors,
                                size_t causal_offset,
                                float fps) {
    const size_t num_patches = latent_num_frames * latent_height * latent_width;
    ov::Tensor coords(ov::element::f32, {batch_size, 3, num_patches, 2});
    float* data = coords.data<float>();

    size_t patch_idx = 0;
    for (size_t f = 0; f < latent_num_frames; ++f) {
        for (size_t h = 0; h < latent_height; ++h) {
            for (size_t w = 0; w < latent_width; ++w, ++patch_idx) {
                const float axis_index[3] = {static_cast<float>(f), static_cast<float>(h), static_cast<float>(w)};
                for (size_t axis = 0; axis < 3; ++axis) {
                    float start = axis_index[axis] * static_cast<float>(vae_scale_factors[axis]);
                    float end = (axis_index[axis] + 1.0f) * static_cast<float>(vae_scale_factors[axis]);
                    if (axis == 0) {
                        // Temporal axis: shift by the causal offset and rescale by fps.
                        start = std::max(0.0f, start + static_cast<float>(causal_offset) - static_cast<float>(vae_scale_factors[0]));
                        end = std::max(0.0f, end + static_cast<float>(causal_offset) - static_cast<float>(vae_scale_factors[0]));
                        start /= fps;
                        end /= fps;
                    }
                    for (size_t b = 0; b < batch_size; ++b) {
                        float* dst = data + ((b * 3 + axis) * num_patches + patch_idx) * 2;
                        dst[0] = start;
                        dst[1] = end;
                    }
                }
            }
        }
    }
    return coords;
}

// Per-patch [start, end) second-space timestamps for the audio rope, matching
// LTX2AudioVideoRotaryPosEmbed.prepare_audio_coords (patch_size_t = 1).
ov::Tensor prepare_audio_coords(size_t batch_size,
                                size_t audio_num_frames,
                                size_t audio_scale_factor,
                                size_t causal_offset,
                                size_t hop_length,
                                size_t sampling_rate) {
    ov::Tensor coords(ov::element::f32, {batch_size, 1, audio_num_frames, 2});
    float* data = coords.data<float>();

    for (size_t l = 0; l < audio_num_frames; ++l) {
        float start_mel = static_cast<float>(l) * static_cast<float>(audio_scale_factor);
        start_mel = std::max(0.0f, start_mel + static_cast<float>(causal_offset) - static_cast<float>(audio_scale_factor));
        float end_mel = static_cast<float>(l + 1) * static_cast<float>(audio_scale_factor);
        end_mel = std::max(0.0f, end_mel + static_cast<float>(causal_offset) - static_cast<float>(audio_scale_factor));

        const float start_s = start_mel * static_cast<float>(hop_length) / static_cast<float>(sampling_rate);
        const float end_s = end_mel * static_cast<float>(hop_length) / static_cast<float>(sampling_rate);

        for (size_t b = 0; b < batch_size; ++b) {
            float* dst = data + (b * audio_num_frames + l) * 2;
            dst[0] = start_s;
            dst[1] = end_s;
        }
    }
    return coords;
}

ov::Tensor i64_to_f32(const ov::Tensor& t) {
    ov::Tensor result(ov::element::f32, t.get_shape());
    const int64_t* src = t.data<const int64_t>();
    float* dst = result.data<float>();
    for (size_t i = 0; i < t.get_size(); ++i) {
        dst[i] = static_cast<float>(src[i]);
    }
    return result;
}

// Repeats a [B, ...] coordinates tensor along the batch dimension: result = cat([t, t], dim=0).
ov::Tensor repeat_batch(const ov::Tensor& t) {
    const ov::Shape shape = t.get_shape();
    ov::Shape out_shape = shape;
    out_shape[0] *= 2;
    ov::Tensor result(t.get_element_type(), out_shape);

    const float* src = t.data<const float>();
    float* dst = result.data<float>();
    std::memcpy(dst, src, t.get_size() * sizeof(float));
    std::memcpy(dst + t.get_size(), src, t.get_size() * sizeof(float));
    return result;
}

// x0 = sample - velocity * sigma (per-element), matching LTX2's convert_velocity_to_x0.
void velocity_to_x0(const float* sample, const float* velocity, float sigma, size_t n, float* out_x0) {
    for (size_t i = 0; i < n; ++i) {
        out_x0[i] = sample[i] - velocity[i] * sigma;
    }
}

// velocity = (sample - x0) / sigma (per-element), matching LTX2's convert_x0_to_velocity.
void x0_to_velocity(const float* sample, const float* x0, float sigma, size_t n, float* out_velocity) {
    OPENVINO_ASSERT(sigma != 0.0f, "x0_to_velocity: sigma must not be zero");
    for (size_t i = 0; i < n; ++i) {
        out_velocity[i] = (sample[i] - x0[i]) / sigma;
    }
}

}  // anonymous namespace

namespace ov::genai {

// LTX2 text-to-audio-video pipeline: a joint video+audio diffusion transformer conditioned on a
// Gemma3-based text encoder, with independent video and audio VAEs and a vocoder producing the
// final waveform.
class LTX2Pipeline : public VideoPipeline {
    std::shared_ptr<IScheduler> m_scheduler, m_audio_scheduler;
    std::shared_ptr<Gemma3TextEncoderModel> m_text_encoder;
    std::shared_ptr<LTX2TextConnectors> m_connectors;
    std::shared_ptr<LTX2VideoTransformer3DModel> m_transformer;
    std::shared_ptr<AutoencoderKLLTXVideo> m_vae;
    std::shared_ptr<AutoencoderKLLTX2Audio> m_audio_vae;
    std::shared_ptr<LTX2Vocoder> m_vocoder;

    std::filesystem::path m_models_dir;
    std::string m_text_encode_device, m_denoise_device, m_vae_device;
    ov::AnyMap m_compile_properties;
    bool m_is_compiled = false;
    size_t m_reshape_batch_size_multiplier = 0;

    int32_t m_num_train_timesteps = 1000;
    size_t m_max_image_seq_len = 4096;
    int64_t m_vae_spatial_compression_ratio = 32, m_vae_temporal_compression_ratio = 8;

    bool do_classifier_free_guidance(float guidance_scale, float audio_guidance_scale) const {
        return guidance_scale > 1.0f || audio_guidance_scale > 1.0f;
    }

    void read_vae_compression_ratios() {
        std::ifstream file(m_models_dir / "vae_decoder" / "config.json");
        OPENVINO_ASSERT(file.is_open(), "Failed to open ", m_models_dir / "vae_decoder" / "config.json");
        nlohmann::json data = nlohmann::json::parse(file);
        utils::read_json_param(data, "spatial_compression_ratio", m_vae_spatial_compression_ratio);
        utils::read_json_param(data, "temporal_compression_ratio", m_vae_temporal_compression_ratio);
    }

    void read_scheduler_params() {
        std::ifstream file(m_models_dir / "scheduler" / "scheduler_config.json");
        OPENVINO_ASSERT(file.is_open(), "Failed to open ", m_models_dir / "scheduler" / "scheduler_config.json");
        nlohmann::json data = nlohmann::json::parse(file);
        utils::read_json_param(data, "num_train_timesteps", m_num_train_timesteps);
        utils::read_json_param(data, "max_image_seq_len", m_max_image_seq_len);
    }

    void check_inputs(const VideoGenerationConfig& config) const {
        utils::validate_generation_config(config);
        OPENVINO_ASSERT(config.height > 0 && config.height % m_vae_spatial_compression_ratio == 0,
                        "Height must be positive and divisible by ", m_vae_spatial_compression_ratio);
        OPENVINO_ASSERT(config.width > 0 && config.width % m_vae_spatial_compression_ratio == 0,
                        "Width must be positive and divisible by ", m_vae_spatial_compression_ratio);
        OPENVINO_ASSERT(config.num_frames > 0, "num_frames must be positive");
        if (config.frame_rate.has_value()) {
            OPENVINO_ASSERT(std::isfinite(*config.frame_rate) && *config.frame_rate > 0.0f,
                            "frame_rate must be a positive finite value, got ", *config.frame_rate);
        }
    }

public:
    explicit LTX2Pipeline(const std::filesystem::path& root_dir) {
        m_models_dir = root_dir;
        read_vae_compression_ratios();
        read_scheduler_params();

        m_scheduler = cast_scheduler(Scheduler::from_config(root_dir / "scheduler/scheduler_config.json"));
        m_audio_scheduler = cast_scheduler(Scheduler::from_config(root_dir / "scheduler/scheduler_config.json"));

        m_text_encoder = std::make_shared<Gemma3TextEncoderModel>(root_dir / "text_encoder");
        m_connectors = std::make_shared<LTX2TextConnectors>(root_dir / "connectors");
        m_transformer = std::make_shared<LTX2VideoTransformer3DModel>(root_dir / "transformer");
        m_vae = std::make_shared<AutoencoderKLLTXVideo>(root_dir / "vae_decoder");
        m_audio_vae = std::make_shared<AutoencoderKLLTX2Audio>(root_dir / "audio_vae_decoder");
        m_vocoder = std::make_shared<LTX2Vocoder>(root_dir / "vocoder");

        m_generation_config = LTX2_DEFAULT_CONFIG;
    }

    LTX2Pipeline(const std::filesystem::path& models_dir, const std::string& device, const ov::AnyMap& properties)
        : LTX2Pipeline(models_dir) {
        compile(device, device, device, properties);
    }

    std::shared_ptr<VideoPipeline> clone() override {
        OPENVINO_ASSERT(m_is_compiled, "Cannot clone an uncompiled LTX2Pipeline");
        auto cloned = std::make_shared<LTX2Pipeline>(*this);
        cloned->m_generation_config.generator.reset();
        cloned->m_scheduler = cast_scheduler(Scheduler::from_config(m_models_dir / "scheduler/scheduler_config.json"));
        cloned->m_audio_scheduler = cast_scheduler(Scheduler::from_config(m_models_dir / "scheduler/scheduler_config.json"));
        cloned->m_text_encoder = m_text_encoder->clone();
        cloned->m_connectors = std::make_shared<LTX2TextConnectors>(m_connectors->clone());
        cloned->m_transformer = std::make_shared<LTX2VideoTransformer3DModel>(m_transformer->clone());
        cloned->m_vae = std::make_shared<AutoencoderKLLTXVideo>(m_vae->clone());
        cloned->m_audio_vae = std::make_shared<AutoencoderKLLTX2Audio>(m_audio_vae->clone());
        cloned->m_vocoder = std::make_shared<LTX2Vocoder>(m_vocoder->clone());
        return cloned;
    }

    void compile(const std::string& text_encode_device,
                const std::string& denoise_device,
                const std::string& vae_device,
                const ov::AnyMap& properties) override {
        update_adapters_from_properties(properties, m_generation_config.adapters);
        m_text_encoder->compile(text_encode_device, properties);
        m_connectors->compile(text_encode_device, properties);
        m_transformer->compile(denoise_device, properties);
        m_vae->compile(vae_device, properties);
        m_audio_vae->compile(vae_device, properties);
        m_vocoder->compile(vae_device, properties);

        m_text_encode_device = text_encode_device;
        m_denoise_device = denoise_device;
        m_vae_device = vae_device;
        m_compile_properties = properties;
        m_is_compiled = true;
    }

    void reshape(int64_t num_videos_per_prompt,
                int64_t num_frames,
                int64_t height,
                int64_t width,
                float guidance_scale) override {
        OPENVINO_ASSERT(height > 0 && height % m_vae_spatial_compression_ratio == 0,
                        "Height must be positive and divisible by ", m_vae_spatial_compression_ratio);
        OPENVINO_ASSERT(width > 0 && width % m_vae_spatial_compression_ratio == 0,
                        "Width must be positive and divisible by ", m_vae_spatial_compression_ratio);

        // The reshape API has no audio parameter; take audio_guidance_scale from the stored config.
        const float audio_guidance_scale = m_generation_config.audio_guidance_scale.value_or(guidance_scale);
        const size_t batch_size_multiplier = do_classifier_free_guidance(guidance_scale, audio_guidance_scale) ? 2 : 1;
        m_reshape_batch_size_multiplier = batch_size_multiplier;
        // VAEs, connectors and vocoder are left dynamic-shaped.
        m_text_encoder->reshape(static_cast<int>(batch_size_multiplier), static_cast<int>(m_generation_config.max_sequence_length));
        m_transformer->reshape(batch_size_multiplier * num_videos_per_prompt);
    }

    VideoGenerationResult generate(const std::string& positive_prompt,
                                   const ov::Tensor& initial_image,
                                   const ov::AnyMap& properties) override {
        OPENVINO_ASSERT(!initial_image,
                        "LTX2Pipeline does not support image-to-video generation yet; only "
                        "text-to-video is currently supported.");

        const auto gen_start = std::chrono::steady_clock::now();
        m_perf_metrics.clean_up();

        VideoGenerationConfig config = m_generation_config;
        utils::update_generation_config(config, properties);
        replace_ltx2_defaults(config);
        check_inputs(config);

        const float guidance_scale = config.guidance_scale;
        const float guidance_rescale = config.guidance_rescale.value_or(0.0f);
        const float audio_guidance_scale = config.audio_guidance_scale.value_or(guidance_scale);
        const float audio_guidance_rescale = config.audio_guidance_rescale.value_or(guidance_rescale);

        const bool use_cfg = do_classifier_free_guidance(guidance_scale, audio_guidance_scale);
        const size_t batch_size_multiplier = use_cfg ? 2 : 1;
        if (m_is_compiled) {
            const size_t expected = m_transformer->get_expected_batch_size();
            if (expected > 0) {
                OPENVINO_ASSERT(expected % config.num_videos_per_prompt == 0,
                                "Compiled batch size must be divisible by num_videos_per_prompt");
                OPENVINO_ASSERT(expected / config.num_videos_per_prompt == batch_size_multiplier,
                                "The compiled model was reshaped for a different guidance_scale/audio_guidance_scale "
                                "combination (CFG on vs. off). Reshape and recompile for the requested settings.");
            }
        } else if (m_reshape_batch_size_multiplier == 0) {
            // Record the multiplier so a later compile()/reshape() can detect a mismatched request.
            m_reshape_batch_size_multiplier = batch_size_multiplier;
        } else {
            OPENVINO_ASSERT(m_reshape_batch_size_multiplier == batch_size_multiplier,
                            "The pipeline was reshaped for a different guidance_scale/audio_guidance_scale "
                            "combination (CFG on vs. off). Reshape for the requested settings.");
        }

        m_transformer->set_adapters(config.adapters);

        std::shared_ptr<ThreadedCallbackWrapper> callback_ptr = nullptr;
        auto callback_iter = properties.find(ov::genai::callback.name());
        if (callback_iter != properties.end()) {
            callback_ptr = std::make_shared<ThreadedCallbackWrapper>(
                callback_iter->second.as<std::function<bool(size_t, size_t, ov::Tensor&)>>());
            callback_ptr->start();
        }

        const size_t B = config.num_videos_per_prompt;
        const int64_t latent_num_frames = (config.num_frames - 1) / m_vae_temporal_compression_ratio + 1;
        const int64_t latent_height = config.height / m_vae_spatial_compression_ratio;
        const int64_t latent_width = config.width / m_vae_spatial_compression_ratio;
        const float frame_rate = *config.frame_rate;

        const auto& t_config = m_transformer->get_config();
        const size_t num_mel_bins = m_audio_vae->get_config().mel_bins;
        const size_t mel_compression_ratio = m_audio_vae->get_config().mel_compression_ratio;
        const size_t latent_mel_bins = num_mel_bins / mel_compression_ratio;
        const size_t audio_latent_channels = m_audio_vae->get_config().latent_channels;

        // Use the effective decoded frame count so audio duration stays in sync with video.
        const int64_t effective_num_frames = (latent_num_frames - 1) * m_vae_temporal_compression_ratio + 1;
        if (static_cast<size_t>(effective_num_frames) != config.num_frames) {
            GENAI_WARN("num_frames should satisfy (num_frames - 1) % " +
                       std::to_string(m_vae_temporal_compression_ratio) + " == 0; generating " +
                       std::to_string(effective_num_frames) + " frames instead of " +
                       std::to_string(config.num_frames));
        }
        const float duration_s = static_cast<float>(effective_num_frames) / frame_rate;
        const float audio_latents_per_second = static_cast<float>(t_config.audio_sampling_rate) /
                                               static_cast<float>(t_config.audio_hop_length) /
                                               static_cast<float>(m_audio_vae->get_config().temporal_compression_ratio);
        const size_t audio_num_frames = static_cast<size_t>(std::lround(duration_s * audio_latents_per_second));

        // 1. Text encoding + connectors
        auto infer_start = std::chrono::steady_clock::now();
        ov::Tensor prompt_embeds = m_text_encoder->infer(positive_prompt,
                                                         config.negative_prompt.value_or(""),
                                                         use_cfg,
                                                         config.max_sequence_length,
                                                         {ov::genai::pad_to_max_length(true),
                                                          ov::genai::max_length(config.max_sequence_length),
                                                          ov::genai::add_special_tokens(true),
                                                          ov::genai::padding_side("left")});
        m_perf_metrics.encoder_inference_duration["text_encoder"] =
            std::chrono::duration<float, std::milli>{std::chrono::steady_clock::now() - infer_start}.count();
        ov::Tensor prompt_attention_mask_i64 = m_text_encoder->get_prompt_attention_mask();

        // Connectors expect a float attention mask.
        ov::Tensor prompt_attention_mask_f32 = i64_to_f32(prompt_attention_mask_i64);

        infer_start = std::chrono::steady_clock::now();
        LTX2TextConnectors::Output connector_output = m_connectors->infer(prompt_embeds, prompt_attention_mask_f32);
        m_perf_metrics.encoder_inference_duration["connectors"] =
            std::chrono::duration<float, std::milli>{std::chrono::steady_clock::now() - infer_start}.count();

        ov::Tensor video_text_embedding = numpy_utils::repeat(connector_output.video_text_embedding, B);
        ov::Tensor audio_text_embedding = numpy_utils::repeat(connector_output.audio_text_embedding, B);
        ov::Tensor connector_attention_mask = numpy_utils::repeat(connector_output.connector_attention_mask, B);

        // The transformer's video-side mask input is i64, but the audio-side mask input is f32.
        ov::Tensor audio_encoder_attention_mask = i64_to_f32(connector_attention_mask);

        // 2. Prepare video and audio latents
        OPENVINO_ASSERT(config.generator, "Generator must not be null");
        ov::Shape video_noise_shape{B, static_cast<size_t>(t_config.in_channels),
                                    static_cast<size_t>(latent_num_frames), static_cast<size_t>(latent_height),
                                    static_cast<size_t>(latent_width)};
        ov::Tensor video_noise = config.generator->randn_tensor(video_noise_shape);
        ov::Tensor latents = pack_latents(video_noise, t_config.patch_size, t_config.patch_size_t);

        ov::Shape audio_noise_shape{B, audio_latent_channels, audio_num_frames, latent_mel_bins};
        ov::Tensor audio_noise = config.generator->randn_tensor(audio_noise_shape);
        ov::Tensor audio_latents = pack_audio_latents(audio_noise);

        const size_t video_sequence_length = latents.get_shape()[1];
        const size_t audio_sequence_length = audio_latents.get_shape()[1];

        // 3. Timesteps: both schedulers use the same (fixed max_image_seq_len-derived) mu, matching
        // LTX2's own quirk of not scaling mu by the actual sequence length.
        const double mu = m_scheduler->calculate_shift(m_max_image_seq_len);
        m_scheduler->set_timesteps_with_mu(mu, config.num_inference_steps, 1.0f);
        m_audio_scheduler->set_timesteps_with_mu(mu, config.num_inference_steps, 1.0f);
        m_scheduler->set_begin_index(0);
        m_audio_scheduler->set_begin_index(0);
        std::vector<float> timesteps = m_scheduler->get_float_timesteps();

        // 4. Rope coordinates (fixed for every denoising step)
        ov::Tensor video_coords = ::prepare_video_coords(B, latent_num_frames, latent_height, latent_width,
                                                         t_config.vae_scale_factors, t_config.causal_offset, frame_rate);
        ov::Tensor audio_coords = ::prepare_audio_coords(B, audio_num_frames, t_config.audio_scale_factor,
                                                         t_config.causal_offset, t_config.audio_hop_length,
                                                         t_config.audio_sampling_rate);
        if (use_cfg) {
            video_coords = repeat_batch(video_coords);
            audio_coords = repeat_batch(audio_coords);
        }

        // 5. Denoising loop
        // x0 scratch is only needed for rescale; plain CFG on velocities is algebraically identical.
        const size_t x0_scratch_size = (use_cfg && (guidance_rescale > 0.0f || audio_guidance_rescale > 0.0f))
                                           ? std::max(latents.get_size(), audio_latents.get_size())
                                           : 0;
        std::vector<float> x0_cond(x0_scratch_size), x0_uncond(x0_scratch_size), guided_x0(x0_scratch_size);

        ov::Tensor velocity_video_tensor(ov::element::f32, latents.get_shape());
        ov::Tensor velocity_audio_tensor(ov::element::f32, audio_latents.get_shape());
        ov::Tensor latent_model_input, audio_latent_model_input;
        if (use_cfg) {
            ov::Shape s = latents.get_shape();
            s[0] *= 2;
            latent_model_input = ov::Tensor(ov::element::f32, s);
            ov::Shape as = audio_latents.get_shape();
            as[0] *= 2;
            audio_latent_model_input = ov::Tensor(ov::element::f32, as);
        }
        ov::Tensor timestep(ov::element::f32, {B * batch_size_multiplier});

        // Applies CFG (+ optional x0-space rescale) to one modality's [uncond; cond] prediction.
        auto apply_guidance = [&](const ov::Tensor& prediction, const ov::Tensor& sample_t,
                                  float scale, float rescale, float sigma, ov::Tensor& out_velocity) {
            const size_t n = sample_t.get_size();
            const float* v_uncond = prediction.data<const float>();
            const float* v_cond = v_uncond + n;
            float* out = out_velocity.data<float>();
            if (rescale > 0.0f) {
                const float* sample = sample_t.data<const float>();
                velocity_to_x0(sample, v_uncond, sigma, n, x0_uncond.data());
                velocity_to_x0(sample, v_cond, sigma, n, x0_cond.data());
                for (size_t k = 0; k < n; ++k) {
                    guided_x0[k] = x0_cond[k] + (scale - 1.0f) * (x0_cond[k] - x0_uncond[k]);
                }
                rescale_noise_cfg(guided_x0.data(), x0_cond.data(), sample_t.get_shape()[0],
                                  n / sample_t.get_shape()[0], rescale);
                x0_to_velocity(sample, guided_x0.data(), sigma, n, out);
            } else {
                for (size_t k = 0; k < n; ++k) {
                    out[k] = v_cond[k] + (scale - 1.0f) * (v_cond[k] - v_uncond[k]);
                }
            }
        };

        for (size_t i = 0; i < timesteps.size(); ++i) {
            auto step_start = std::chrono::steady_clock::now();
            const float sigma = timesteps[i] / static_cast<float>(m_num_train_timesteps);

            if (use_cfg) {
                numpy_utils::batch_copy(latents, latent_model_input, 0, 0, B);
                numpy_utils::batch_copy(latents, latent_model_input, 0, B, B);
                numpy_utils::batch_copy(audio_latents, audio_latent_model_input, 0, 0, B);
                numpy_utils::batch_copy(audio_latents, audio_latent_model_input, 0, B, B);
            } else {
                latent_model_input = latents;
                audio_latent_model_input = audio_latents;
            }

            std::fill_n(timestep.data<float>(), timestep.get_size(), timesteps[i]);

            infer_start = std::chrono::steady_clock::now();
            LTX2VideoTransformer3DModel::Output transformer_output =
                m_transformer->infer(latent_model_input,
                                     audio_latent_model_input,
                                     video_text_embedding,
                                     audio_text_embedding,
                                     timestep,
                                     connector_attention_mask,
                                     audio_encoder_attention_mask,
                                     static_cast<size_t>(latent_num_frames),
                                     static_cast<size_t>(latent_height),
                                     static_cast<size_t>(latent_width),
                                     frame_rate,
                                     audio_num_frames,
                                     video_coords,
                                     audio_coords);
            const auto infer_duration = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - infer_start);
            m_perf_metrics.raw_metrics.transformer_inference_durations.emplace_back(MicroSeconds(infer_duration));

            ov::Tensor video_velocity, audio_velocity;
            if (use_cfg) {
                apply_guidance(transformer_output.video, latents, guidance_scale, guidance_rescale,
                               sigma, velocity_video_tensor);
                apply_guidance(transformer_output.audio, audio_latents, audio_guidance_scale,
                               audio_guidance_rescale, sigma, velocity_audio_tensor);
                video_velocity = velocity_video_tensor;
                audio_velocity = velocity_audio_tensor;
            } else {
                video_velocity = transformer_output.video;
                audio_velocity = transformer_output.audio;
            }

            auto video_step_result = m_scheduler->step(video_velocity, latents, i, config.generator);
            latents = video_step_result["latent"];
            auto audio_step_result = m_audio_scheduler->step(audio_velocity, audio_latents, i, config.generator);
            audio_latents = audio_step_result["latent"];

            if (callback_ptr && callback_ptr->has_callback() &&
                callback_ptr->write(i, timesteps.size(), latents) == CallbackStatus::STOP) {
                callback_ptr->end();
                auto step_ms = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - step_start);
                m_perf_metrics.raw_metrics.iteration_durations.emplace_back(MicroSeconds(step_ms));
                m_perf_metrics.generate_duration =
                    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - gen_start).count();
                return {ov::Tensor(ov::element::u8, {}), m_perf_metrics, ov::Tensor()};
            }

            auto step_ms = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - step_start);
            m_perf_metrics.raw_metrics.iteration_durations.emplace_back(MicroSeconds(step_ms));
        }
        if (callback_ptr != nullptr) {
            callback_ptr->end();
        }

        // 6. Decode video
        ov::Tensor video_latent_unpacked = unpack_latents(latents, static_cast<size_t>(latent_num_frames),
                                                          static_cast<size_t>(latent_height), static_cast<size_t>(latent_width),
                                                          t_config.patch_size, t_config.patch_size_t);
        video_latent_unpacked = denormalize_latents(video_latent_unpacked,
                                                     tensor_from_vector(m_vae->get_config().latents_mean_data),
                                                     tensor_from_vector(m_vae->get_config().latents_std_data),
                                                     m_vae->get_config().scaling_factor);
        const auto decode_start = std::chrono::steady_clock::now();
        ov::Tensor video = m_vae->decode(video_latent_unpacked);
        m_perf_metrics.vae_decoder_inference_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - decode_start).count();

        // 7. Decode audio: denormalize (packed) -> unpack -> audio VAE decode -> vocoder
        denormalize_audio_latents_inplace(audio_latents, m_audio_vae->get_config().latents_mean_data,
                                          m_audio_vae->get_config().latents_std_data);
        ov::Tensor audio_latent_unpacked = unpack_audio_latents(audio_latents, latent_mel_bins);
        ov::Tensor mel_spectrogram = m_audio_vae->decode(audio_latent_unpacked);
        ov::Tensor audio = m_vocoder->infer(mel_spectrogram);

        m_perf_metrics.generate_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - gen_start).count();

        return {video, m_perf_metrics, audio};
    }

    VideoGenerationResult decode(const ov::Tensor& latent) override {
        ov::Tensor video = m_vae->decode(latent);
        return VideoGenerationResult{video, m_perf_metrics, ov::Tensor()};
    }

protected:
    void replace_config_defaults(VideoGenerationConfig& config) const override {
        replace_ltx2_defaults(config);
    }
};

}  // namespace ov::genai
