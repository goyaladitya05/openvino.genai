// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/video_generation/ltx2_vocoder.hpp"

#include <fstream>

#include "utils.hpp"
#include "json_utils.hpp"
#include "lora/helper.hpp"

using namespace ov::genai;

LTX2Vocoder::Config::Config(const std::filesystem::path& config_path) {
    std::ifstream file(config_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", config_path);

    nlohmann::json data = nlohmann::json::parse(file);
    using utils::read_json_param;

    read_json_param(data, "output_sampling_rate", output_sampling_rate);
}

LTX2Vocoder::LTX2Vocoder(const std::filesystem::path& root_dir)
    : m_config(root_dir / "config.json") {
    m_model = utils::singleton_core().read_model(root_dir / "openvino_model.xml");
}

LTX2Vocoder::LTX2Vocoder(const std::filesystem::path& root_dir, const std::string& device, const ov::AnyMap& properties)
    : LTX2Vocoder(root_dir) {
    compile(device, properties);
}

LTX2Vocoder& LTX2Vocoder::compile(const std::string& device, const ov::AnyMap& properties) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot re-compile already compiled model");
    std::optional<AdapterConfig> unused;
    auto filtered_properties = extract_adapters_from_properties(properties, &unused);
    ov::CompiledModel compiled_model = utils::singleton_core().compile_model(m_model, device, *filtered_properties);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "LTX2 vocoder model");
    m_request = compiled_model.create_infer_request();
    m_model.reset();

    return *this;
}

LTX2Vocoder LTX2Vocoder::clone() {
    OPENVINO_ASSERT((m_model != nullptr) ^ static_cast<bool>(m_request),
                    "LTX2Vocoder must have exactly one of m_model or m_request initialized");

    LTX2Vocoder cloned = *this;

    if (m_model) {
        cloned.m_model = m_model->clone();
    } else {
        cloned.m_request = m_request.get_compiled_model().create_infer_request();
    }

    return cloned;
}

const LTX2Vocoder::Config& LTX2Vocoder::get_config() const {
    return m_config;
}

ov::Tensor LTX2Vocoder::infer(const ov::Tensor& mel_spectrogram) {
    OPENVINO_ASSERT(m_request, "Vocoder model must be compiled first. Cannot infer non-compiled model");

    m_request.set_tensor("hidden_states", mel_spectrogram);
    m_request.infer();
    return m_request.get_output_tensor();
}
