// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>

#include <nlohmann/json.hpp>

#include "openvino/genai/video_generation/text2video_pipeline.hpp"
#include "generation_config_utils.hpp"

namespace {

inline std::string get_class_name(const std::filesystem::path& root_dir) {
    const std::filesystem::path model_index_path = root_dir / "model_index.json";
    std::ifstream file(model_index_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", model_index_path);
    return nlohmann::json::parse(file)["_class_name"].get<std::string>();
}

}  // namespace

namespace ov::genai {

enum class VideoPipelineType {
    TEXT_2_VIDEO = 0,
    IMAGE_2_VIDEO = 1,
};

// Internal abstract base for video generation model families, analogous to DiffusionPipeline
// in image_generation. The concrete implementation is selected based on model_index.json's
// '_class_name'.
class VideoPipeline {
public:
    const VideoGenerationConfig& get_generation_config() const {
        return m_generation_config;
    }

    void set_generation_config(const VideoGenerationConfig& generation_config) {
        utils::validate_generation_config(generation_config);
        m_generation_config = generation_config;
        replace_config_defaults(m_generation_config);
    }

    virtual VideoGenerationResult generate(const std::string& positive_prompt,
                                           const ov::Tensor& initial_image,
                                           const ov::AnyMap& properties) = 0;

    virtual VideoGenerationResult decode(const ov::Tensor& latent) = 0;

    virtual void reshape(int64_t num_videos_per_prompt,
                         int64_t num_frames,
                         int64_t height,
                         int64_t width,
                         float guidance_scale) = 0;

    virtual void compile(const std::string& device, const ov::AnyMap& properties) {
        compile(device, device, device, properties);
    }

    virtual void compile(const std::string& text_encode_device,
                         const std::string& denoise_device,
                         const std::string& vae_device,
                         const ov::AnyMap& properties) = 0;

    virtual std::shared_ptr<VideoPipeline> clone() = 0;

    VideoGenerationPerfMetrics get_performance_metrics() const {
        return m_perf_metrics;
    }

    void save_load_time(std::chrono::steady_clock::time_point start_time) {
        auto stop_time = std::chrono::steady_clock::now();
        m_load_time_ms += std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count();
    }

    virtual void export_model(const std::filesystem::path& export_dir) {
        OPENVINO_THROW("Export model is not implemented for this pipeline");
    }

    virtual ~VideoPipeline() = default;

protected:
    // Fills in any sentinel/unset fields of 'config' with this model family's defaults.
    virtual void replace_config_defaults(VideoGenerationConfig& config) const = 0;

    VideoGenerationConfig m_generation_config;
    float m_load_time_ms = 0.0f;
    VideoGenerationPerfMetrics m_perf_metrics;
};

}  // namespace ov::genai
