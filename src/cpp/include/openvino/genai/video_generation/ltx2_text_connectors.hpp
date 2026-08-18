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

// Projects the text encoder's concatenated hidden states into separate video- and
// audio-conditioning embeddings, as required by LTX2's joint transformer.
class OPENVINO_GENAI_EXPORTS LTX2TextConnectors {
public:
    struct Output {
        ov::Tensor video_text_embedding;
        ov::Tensor audio_text_embedding;
        ov::Tensor connector_attention_mask;
    };

    explicit LTX2TextConnectors(const std::filesystem::path& root_dir);

    LTX2TextConnectors(const std::filesystem::path& root_dir, const std::string& device, const ov::AnyMap& properties = {});

    LTX2TextConnectors& compile(const std::string& device, const ov::AnyMap& properties = {});

    LTX2TextConnectors clone();

    Output infer(const ov::Tensor& text_encoder_hidden_states, const ov::Tensor& attention_mask);

private:
    ov::InferRequest m_request;
    std::shared_ptr<ov::Model> m_model;
};

}  // namespace ov::genai
