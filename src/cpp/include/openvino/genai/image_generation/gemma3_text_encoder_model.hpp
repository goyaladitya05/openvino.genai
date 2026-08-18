// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include "openvino/genai/visibility.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/lora_adapter.hpp"

#include "openvino/core/any.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov {
namespace genai {

// Gemma3-based text encoder used by LTX2 for text conditioning. Unlike T5EncoderModel, its
// output is the concatenation of several intermediate hidden-state layers.
class OPENVINO_GENAI_EXPORTS Gemma3TextEncoderModel {
public:
    explicit Gemma3TextEncoderModel(const std::filesystem::path& root_dir);

    Gemma3TextEncoderModel(const std::filesystem::path& root_dir,
                           const std::string& device,
                           const ov::AnyMap& properties = {});

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    Gemma3TextEncoderModel(const std::filesystem::path& root_dir,
                           const std::string& device,
                           Properties&&... properties)
        : Gemma3TextEncoderModel(root_dir, device, ov::AnyMap{std::forward<Properties>(properties)...}) { }

    Gemma3TextEncoderModel(const Gemma3TextEncoderModel&);

    std::shared_ptr<Gemma3TextEncoderModel> clone();

    Gemma3TextEncoderModel& reshape(int batch_size, int max_sequence_length);

    Gemma3TextEncoderModel& compile(const std::string& device, const ov::AnyMap& properties = {});

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<Gemma3TextEncoderModel&, Properties...> compile(
            const std::string& device,
            Properties&&... properties) {
        return compile(device, ov::AnyMap{std::forward<Properties>(properties)...});
    }

    /**
     * Tokenizes and encodes the prompt(s), returning the concatenation of intermediate hidden-state
     * layers exposed by the exported model (shape [B, S, H * num_hidden_state_layers]), matching
     * LTX2's `torch.stack(hidden_states, dim=-1).flatten(2, 3)` prompt-encoding convention.
     */
    ov::Tensor infer(const std::string& pos_prompt,
                     const std::string& neg_prompt,
                     bool do_classifier_free_guidance,
                     int max_sequence_length,
                     const ov::AnyMap& tokenization_params = {});

    ov::Tensor get_prompt_attention_mask() const;

private:
    AdapterController m_adapter_controller;
    ov::InferRequest m_request;
    std::shared_ptr<ov::Model> m_model;
    std::vector<std::string> m_hidden_state_output_names;
    ov::Tensor m_prompt_attention_mask;

    Tokenizer m_tokenizer;
};

} // namespace genai
} // namespace ov
