// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/video_generation/ltx2_text_connectors.hpp"

#include "utils.hpp"
#include "lora/helper.hpp"

using namespace ov::genai;

LTX2TextConnectors::LTX2TextConnectors(const std::filesystem::path& root_dir) {
    m_model = utils::singleton_core().read_model(root_dir / "openvino_model.xml");
}

LTX2TextConnectors::LTX2TextConnectors(const std::filesystem::path& root_dir,
                                       const std::string& device,
                                       const ov::AnyMap& properties)
    : LTX2TextConnectors(root_dir) {
    compile(device, properties);
}

LTX2TextConnectors& LTX2TextConnectors::compile(const std::string& device, const ov::AnyMap& properties) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot re-compile already compiled model");
    std::optional<AdapterConfig> unused;
    auto filtered_properties = extract_adapters_from_properties(properties, &unused);
    ov::CompiledModel compiled_model = utils::singleton_core().compile_model(m_model, device, *filtered_properties);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "LTX2 text connectors model");
    m_request = compiled_model.create_infer_request();
    m_model.reset();

    return *this;
}

LTX2TextConnectors LTX2TextConnectors::clone() {
    OPENVINO_ASSERT((m_model != nullptr) ^ static_cast<bool>(m_request),
                    "LTX2TextConnectors must have exactly one of m_model or m_request initialized");

    LTX2TextConnectors cloned = *this;

    if (m_model) {
        cloned.m_model = m_model->clone();
    } else {
        cloned.m_request = m_request.get_compiled_model().create_infer_request();
    }

    return cloned;
}

LTX2TextConnectors::Output LTX2TextConnectors::infer(const ov::Tensor& text_encoder_hidden_states,
                                                     const ov::Tensor& attention_mask) {
    OPENVINO_ASSERT(m_request, "Connectors model must be compiled first. Cannot infer non-compiled model");

    m_request.set_tensor("text_encoder_hidden_states", text_encoder_hidden_states);
    m_request.set_tensor("attention_mask", attention_mask);
    m_request.infer();

    // Copy to owned tensors - get_tensor() aliases the infer request's internal
    // buffers, which would be overwritten on the next infer() call.
    auto copy_owned = [](const ov::Tensor& t) {
        ov::Tensor owned(t.get_element_type(), t.get_shape());
        t.copy_to(owned);
        return owned;
    };
    Output output;
    output.video_text_embedding = copy_owned(m_request.get_tensor("video_text_embedding"));
    output.audio_text_embedding = copy_owned(m_request.get_tensor("audio_text_embedding"));
    output.connector_attention_mask = copy_owned(m_request.get_tensor("connector_attention_mask"));
    return output;
}
