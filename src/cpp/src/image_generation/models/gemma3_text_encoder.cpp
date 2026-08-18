// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/image_generation/gemma3_text_encoder_model.hpp"

#include <algorithm>
#include <cstring>
#include <regex>
#include <type_traits>

#include "lora/helper.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

std::filesystem::path get_tokenizer_path_by_text_encoder(const std::filesystem::path& text_encoder_path);

namespace {

// Discovers the model's 'hidden_states.N' outputs (N = 0, 1, ..., num_hidden_layers), in order.
// The encoder is exported with output_hidden_states=True, so every layer is a separate output.
std::vector<std::string> discover_hidden_state_outputs(const std::vector<ov::Output<ov::Node>>& outputs) {
    std::vector<std::pair<size_t, std::string>> indexed_names;
    const std::regex hidden_state_pattern(R"(hidden_states\.(\d+))");

    for (const auto& output : outputs) {
        for (const auto& name : output.get_names()) {
            std::smatch match;
            if (std::regex_match(name, match, hidden_state_pattern)) {
                indexed_names.emplace_back(std::stoul(match[1].str()), name);
                break;
            }
        }
    }

    OPENVINO_ASSERT(!indexed_names.empty(),
                    "Gemma3TextEncoderModel: no 'hidden_states.N' outputs found. "
                    "The model may be exported without 'output_hidden_states=True'.");

    std::sort(indexed_names.begin(), indexed_names.end());

    std::vector<std::string> names;
    names.reserve(indexed_names.size());
    for (auto& [index, name] : indexed_names) {
        names.push_back(std::move(name));
    }
    return names;
}

}  // namespace

Gemma3TextEncoderModel::Gemma3TextEncoderModel(const std::filesystem::path& root_dir)
    : m_tokenizer(get_tokenizer_path_by_text_encoder(root_dir)) {
    m_model = utils::singleton_core().read_model(root_dir / "openvino_model.xml");
    m_hidden_state_output_names = discover_hidden_state_outputs(m_model->outputs());
}

Gemma3TextEncoderModel::Gemma3TextEncoderModel(const std::filesystem::path& root_dir,
                                               const std::string& device,
                                               const ov::AnyMap& properties)
    : Gemma3TextEncoderModel(root_dir) {
    compile(device, properties);
}

Gemma3TextEncoderModel::Gemma3TextEncoderModel(const Gemma3TextEncoderModel&) = default;

std::shared_ptr<Gemma3TextEncoderModel> Gemma3TextEncoderModel::clone() {
    OPENVINO_ASSERT((m_model != nullptr) ^ static_cast<bool>(m_request),
                    "Gemma3TextEncoderModel must have exactly one of m_model or m_request initialized");

    std::shared_ptr<Gemma3TextEncoderModel> cloned = std::make_shared<Gemma3TextEncoderModel>(*this);

    if (m_model) {
        cloned->m_model = m_model->clone();
    } else {
        cloned->m_request = m_request.get_compiled_model().create_infer_request();
    }

    return cloned;
}

Gemma3TextEncoderModel& Gemma3TextEncoderModel::reshape(int batch_size, int max_sequence_length) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot reshape already compiled model");

    std::map<std::string, ov::PartialShape> name_to_shape;
    for (auto&& input : m_model->inputs()) {
        std::string input_name = input.get_any_name();
        if (input_name == "input_ids" || input_name == "attention_mask") {
            name_to_shape[input_name] = {batch_size, max_sequence_length};
        } else {
            name_to_shape[input_name] = input.get_partial_shape();
        }
    }
    m_model->reshape(name_to_shape);

    return *this;
}

Gemma3TextEncoderModel& Gemma3TextEncoderModel::compile(const std::string& device, const ov::AnyMap& properties) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot re-compile already compiled model");
    ov::CompiledModel compiled_model = utils::singleton_core().compile_model(m_model, device, *extract_adapters_from_properties(properties));
    ov::genai::utils::print_compiled_model_properties(compiled_model, "Gemma3 text encoder model");
    m_request = compiled_model.create_infer_request();
    // release the original model
    m_model.reset();

    return *this;
}

ov::Tensor Gemma3TextEncoderModel::infer(const std::string& pos_prompt,
                                         const std::string& neg_prompt,
                                         bool do_classifier_free_guidance,
                                         int max_sequence_length,
                                         const ov::AnyMap& tokenization_params) {
    OPENVINO_ASSERT(m_request, "Gemma3 text encoder model must be compiled first. Cannot infer non-compiled model");

    const int32_t pad_token_id = m_tokenizer.get_pad_token_id();
    const size_t batch_size = do_classifier_free_guidance ? 2 : 1;
    const size_t seq_len = static_cast<size_t>(max_sequence_length);

    ov::Tensor input_ids = m_request.get_tensor("input_ids");
    input_ids.set_shape({batch_size, seq_len});
    ov::Tensor attention_mask = m_request.get_tensor("attention_mask");
    attention_mask.set_shape({batch_size, seq_len});
    m_prompt_attention_mask = ov::Tensor(attention_mask.get_element_type(), {batch_size, seq_len});

    auto perform_tokenization = [&](const std::string& prompt, size_t batch_idx) {
        auto tokenizer_output = m_tokenizer.encode(prompt, tokenization_params);
        ov::Tensor input_ids_token = tokenizer_output.input_ids;
        ov::Tensor attention_mask_token = tokenizer_output.attention_mask;
        const size_t token_len = std::min(seq_len, input_ids_token.get_shape()[1]);
        const int64_t* src_ids = input_ids_token.data<int64_t>();
        const int64_t* src_mask = attention_mask_token.data<int64_t>();

        auto fill_rows = [&](auto* ids_data, auto* mask_data, auto* out_mask_data) {
            using T = std::remove_pointer_t<decltype(ids_data)>;
            auto* ids_row = ids_data + batch_idx * seq_len;
            auto* mask_row = mask_data + batch_idx * seq_len;
            auto* out_mask_row = out_mask_data + batch_idx * seq_len;
            std::fill_n(ids_row, seq_len, static_cast<T>(pad_token_id));
            std::fill_n(mask_row, seq_len, T{0});
            std::fill_n(out_mask_row, seq_len, T{0});
            for (size_t i = 0; i < token_len; ++i) {
                ids_row[i] = static_cast<T>(src_ids[i]);
                mask_row[i] = static_cast<T>(src_mask[i]);
                out_mask_row[i] = static_cast<T>(src_mask[i]);
            }
        };

        if (input_ids.get_element_type() == ov::element::i32) {
            fill_rows(input_ids.data<int32_t>(),
                      attention_mask.data<int32_t>(),
                      m_prompt_attention_mask.data<int32_t>());
        } else {
            fill_rows(input_ids.data<int64_t>(),
                      attention_mask.data<int64_t>(),
                      m_prompt_attention_mask.data<int64_t>());
        }
    };

    size_t current_batch_idx = 0;
    if (do_classifier_free_guidance) {
        perform_tokenization(neg_prompt, current_batch_idx);
        ++current_batch_idx;
    }
    perform_tokenization(pos_prompt, current_batch_idx);

    m_request.infer();

    // Gather every 'hidden_states.N' layer and interleave them along the feature dimension, matching
    // LTX2's `torch.stack(hidden_states, dim=-1).flatten(2, 3)` prompt-encoding convention: the output
    // at feature position h * L + l equals hidden_states[l][..., h], NOT a sequential per-layer concat.
    const size_t num_layers = m_hidden_state_output_names.size();
    ov::Tensor first_layer = m_request.get_tensor(m_hidden_state_output_names[0]);
    const ov::Shape layer_shape = first_layer.get_shape();
    OPENVINO_ASSERT(layer_shape.size() == 3, "Gemma3TextEncoderModel: expected rank-3 hidden_states outputs");
    const size_t hidden_size = layer_shape[2];

    ov::Tensor result(ov::element::f32, {batch_size, seq_len, hidden_size * num_layers});
    float* result_data = result.data<float>();

    // Iterate tokens outermost and layers innermost so the destination is written sequentially
    // exactly once (the layer-outermost order re-traverses the whole result once per layer).
    std::vector<const float*> layer_data(num_layers);
    for (size_t l = 0; l < num_layers; ++l) {
        layer_data[l] = m_request.get_tensor(m_hidden_state_output_names[l]).data<float>();
    }
    for (size_t token = 0; token < batch_size * seq_len; ++token) {
        float* dst = result_data + token * hidden_size * num_layers;
        const size_t token_offset = token * hidden_size;
        for (size_t h = 0; h < hidden_size; ++h) {
            for (size_t l = 0; l < num_layers; ++l) {
                dst[h * num_layers + l] = layer_data[l][token_offset + h];
            }
        }
    }

    return result;
}

ov::Tensor Gemma3TextEncoderModel::get_prompt_attention_mask() const {
    OPENVINO_ASSERT(m_prompt_attention_mask,
                    "Prompt attention mask must be initialized before use. You must call infer.");
    return m_prompt_attention_mask;
}

} // namespace genai
} // namespace ov
