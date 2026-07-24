// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/video_generation/ltx2_video_transformer_3d_model.hpp"

#include <fstream>
#include <set>

#include "json_utils.hpp"
#include "utils.hpp"
#include "lora/helper.hpp"
#include "lora/names_mapping.hpp"

using namespace ov::genai;

LTX2VideoTransformer3DModel::Config::Config(const std::filesystem::path& config_path) {
    std::ifstream file(config_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", config_path);

    nlohmann::json data = nlohmann::json::parse(file);
    using utils::read_json_param;

    read_json_param(data, "in_channels", in_channels);
    read_json_param(data, "patch_size", patch_size);
    read_json_param(data, "patch_size_t", patch_size_t);

    read_json_param(data, "audio_in_channels", audio_in_channels);
    read_json_param(data, "audio_patch_size", audio_patch_size);
    read_json_param(data, "audio_patch_size_t", audio_patch_size_t);

    read_json_param(data, "num_attention_heads", num_attention_heads);
    read_json_param(data, "attention_head_dim", attention_head_dim);
    read_json_param(data, "audio_num_attention_heads", audio_num_attention_heads);
    read_json_param(data, "audio_attention_head_dim", audio_attention_head_dim);

    read_json_param(data, "pos_embed_max_pos", pos_embed_max_pos);
    read_json_param(data, "audio_pos_embed_max_pos", audio_pos_embed_max_pos);
    read_json_param(data, "base_height", base_height);
    read_json_param(data, "base_width", base_width);

    read_json_param(data, "rope_theta", rope_theta);
    read_json_param(data, "causal_offset", causal_offset);

    read_json_param(data, "vae_scale_factors", vae_scale_factors);
    read_json_param(data, "audio_scale_factor", audio_scale_factor);

    read_json_param(data, "audio_sampling_rate", audio_sampling_rate);
    read_json_param(data, "audio_hop_length", audio_hop_length);
}

LTX2VideoTransformer3DModel::LTX2VideoTransformer3DModel(const std::filesystem::path& root_dir)
    : m_config(root_dir / "config.json") {
    m_model = utils::singleton_core().read_model(root_dir / "openvino_model.xml");
}

LTX2VideoTransformer3DModel::LTX2VideoTransformer3DModel(const std::filesystem::path& root_dir,
                                                          const std::string& device,
                                                          const ov::AnyMap& properties)
    : LTX2VideoTransformer3DModel(root_dir) {
    compile(device, properties);
}

LTX2VideoTransformer3DModel::LTX2VideoTransformer3DModel(const LTX2VideoTransformer3DModel&) = default;

LTX2VideoTransformer3DModel LTX2VideoTransformer3DModel::clone() {
    OPENVINO_ASSERT((m_model != nullptr) ^ static_cast<bool>(m_request),
                    "LTX2VideoTransformer3DModel must have exactly one of m_model or m_request initialized");

    LTX2VideoTransformer3DModel cloned = *this;

    if (m_model) {
        cloned.m_model = m_model->clone();
    } else {
        cloned.m_request = m_request.get_compiled_model().create_infer_request();
    }

    return cloned;
}

const LTX2VideoTransformer3DModel::Config& LTX2VideoTransformer3DModel::get_config() const {
    return m_config;
}

LTX2VideoTransformer3DModel& LTX2VideoTransformer3DModel::compile(const std::string& device, const ov::AnyMap& properties) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot re-compile already compiled model");
    std::optional<AdapterConfig> adapters;
    auto filtered_properties = extract_adapters_from_properties(properties, &adapters);
    if (adapters) {
        m_lora_prefix = adapters->get_tensor_name_prefix().value_or(detect_lora_prefix(*adapters));
        adapters->set_tensor_name_prefix(m_lora_prefix);
        m_adapter_controller = AdapterController(m_model, *adapters, device);
    }
    ov::CompiledModel compiled_model = utils::singleton_core().compile_model(m_model, device, *filtered_properties);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "LTX2 Video Transformer 3D model");
    m_request = compiled_model.create_infer_request();
    const auto& input_shape = compiled_model.input("hidden_states").get_partial_shape();
    m_expected_batch_size = input_shape.is_static() ? input_shape[0].get_length() : 0;
    // release the original model
    m_model.reset();

    return *this;
}

void LTX2VideoTransformer3DModel::set_adapters(const std::optional<AdapterConfig>& adapters) {
    OPENVINO_ASSERT(m_request, "Transformer model must be compiled first");
    if (adapters) {
        if (*adapters && !adapters->get_tensor_name_prefix().has_value()) {
            AdapterConfig adapters_with_prefix = *adapters;
            adapters_with_prefix.set_tensor_name_prefix(m_lora_prefix);
            m_adapter_controller.apply(m_request, adapters_with_prefix);
        } else {
            m_adapter_controller.apply(m_request, *adapters);
        }
    }
}

LTX2VideoTransformer3DModel::Output LTX2VideoTransformer3DModel::infer(const ov::Tensor& hidden_states,
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
                                                                       const ov::Tensor& audio_coords) {
    OPENVINO_ASSERT(m_request, "Transformer model must be compiled first. Cannot infer non-compiled model");

    auto make_scalar_i64 = [](size_t value) {
        ov::Tensor scalar(ov::element::i64, {});
        scalar.data<int64_t>()[0] = static_cast<int64_t>(value);
        return scalar;
    };
    auto make_scalar_f32 = [](float value) {
        ov::Tensor scalar(ov::element::f32, {});
        scalar.data<float>()[0] = value;
        return scalar;
    };

    m_request.set_tensor("hidden_states", hidden_states);
    m_request.set_tensor("audio_hidden_states", audio_hidden_states);
    m_request.set_tensor("encoder_hidden_states", encoder_hidden_states);
    m_request.set_tensor("audio_encoder_hidden_states", audio_encoder_hidden_states);
    m_request.set_tensor("timestep", timestep);
    m_request.set_tensor("encoder_attention_mask", encoder_attention_mask);
    m_request.set_tensor("audio_encoder_attention_mask", audio_encoder_attention_mask);
    m_request.set_tensor("num_frames", make_scalar_i64(num_frames));
    m_request.set_tensor("height", make_scalar_i64(height));
    m_request.set_tensor("width", make_scalar_i64(width));
    m_request.set_tensor("fps", make_scalar_f32(fps));
    m_request.set_tensor("audio_num_frames", make_scalar_i64(audio_num_frames));
    m_request.set_tensor("video_coords", video_coords);
    m_request.set_tensor("audio_coords", audio_coords);

    m_request.infer();

    Output output;
    output.video = m_request.get_tensor("out_sample");
    output.audio = m_request.get_tensor("audio_out_sample");
    return output;
}

size_t LTX2VideoTransformer3DModel::get_expected_batch_size() const {
    return m_expected_batch_size;
}

LTX2VideoTransformer3DModel& LTX2VideoTransformer3DModel::reshape(int64_t batch_size) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot reshape already compiled model");

    // Scalar (rank-0) micro-condition inputs are left untouched; everything else gets its batch
    // dimension fixed and the rest left dynamic. Mirrors optimum-intel's own
    // OVLTX2Pipeline._reshape_transformer, which reshapes the exported model the same way.
    static const std::set<std::string> scalar_inputs = {"num_frames", "height", "width", "fps", "audio_num_frames"};

    std::map<std::string, ov::PartialShape> name_to_shape;
    for (auto&& input : m_model->inputs()) {
        const std::string input_name = input.get_any_name();
        ov::PartialShape shape = input.get_partial_shape();
        if (scalar_inputs.count(input_name) == 0 && shape.rank().is_static() && shape.rank().get_length() >= 1) {
            shape[0] = batch_size;
            for (int64_t i = 1; i < shape.rank().get_length(); ++i) {
                shape[i] = -1;
            }
        }
        name_to_shape[input_name] = shape;
    }
    m_model->reshape(name_to_shape);

    return *this;
}
