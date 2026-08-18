// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <string>

#include "openvino/core/any.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/properties.hpp"

#include "openvino/genai/visibility.hpp"

namespace ov::genai {

// Converts a mel spectrogram (as produced by AutoencoderKLLTX2Audio::decode) into a raw audio
// waveform.
class OPENVINO_GENAI_EXPORTS LTX2Vocoder {
public:
    struct OPENVINO_GENAI_EXPORTS Config {
        size_t output_sampling_rate = 24000;

        explicit Config(const std::filesystem::path& config_path);
    };

    explicit LTX2Vocoder(const std::filesystem::path& root_dir);

    LTX2Vocoder(const std::filesystem::path& root_dir, const std::string& device, const ov::AnyMap& properties = {});

    LTX2Vocoder& compile(const std::string& device, const ov::AnyMap& properties = {});

    LTX2Vocoder clone();

    const Config& get_config() const;

    ov::Tensor infer(const ov::Tensor& mel_spectrogram);

private:
    Config m_config;
    ov::InferRequest m_request;
    std::shared_ptr<ov::Model> m_model;
};

}  // namespace ov::genai
