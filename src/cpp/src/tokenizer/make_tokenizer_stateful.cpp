// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tokenizer/make_tokenizer_stateful.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/constant.hpp"
#include <openvino/pass/manager.hpp>
#include <openvino/core/graph_util.hpp>

using namespace ov;
using namespace ov::op;

bool ov::genai::MakeAddSpecialTokensSatateful::run_on_model(const std::shared_ptr<ov::Model>& model) {
    // Inserts Selects to add special tokens inputs, so that they can be requlated in runtime.
    std::shared_ptr<ov::Node> combine_seg_node;
    for (auto node: model->get_ordered_ops()) {
        if (strcmp(node->get_type_info().name, "CombineSegments") == 0) {
            combine_seg_node = node;
        }
    }
    if (!combine_seg_node) { 
        return false; 
    }
    
    size_t num_segments = (combine_seg_node->get_input_size() - 1) / 3;
    // Inputs responsible for adding special tokens are at 3*i + 1.
    std::vector<Input<Node>> add_spec_inputs;
    add_spec_inputs.reserve(num_segments);

    for (size_t i = 0; i < num_segments; i++) {
        auto begin_input = combine_seg_node->input_value(3*i + 0).get_node_shared_ptr();
        auto end_input = combine_seg_node->input_value(3*i + 1).get_node_shared_ptr();

        // If it's not a main sequence inputs, then it's a special tokens.
        if (!ov::as_type_ptr<v1::Add>(end_input) && !ov::as_type_ptr<v1::Subtract>(begin_input) && strcmp(end_input->get_type_name(), "Truncate") != 0) {
            add_spec_inputs.emplace_back(combine_seg_node->input(3*i + 1));
        }
    }
    if (add_spec_inputs.empty()) { 
        return false; 
    }

    // Default mode is add_special_tokens.
    auto default_mode_const = std::make_shared<v0::Constant>(ov::element::boolean, ov::Shape{}, std::vector{true});
    auto variable = std::make_shared<op::util::Variable>(op::util::VariableInfo{Shape{}, element::boolean, ADD_SPECIAL_TOKENS_VAR_ID});
    auto read_value = std::make_shared<v6::ReadValue>(default_mode_const, variable);
    auto zero_constant = std::make_shared<v0::Constant>(ov::element::i32, ov::Shape{}, std::vector{0});

    for (size_t i = 0; i < add_spec_inputs.size(); i++) {
        auto select_node = std::make_shared<v1::Select>(read_value, add_spec_inputs[i].get_source_output(), zero_constant);
        add_spec_inputs[i].replace_source_output(select_node);
    }

    auto assign = std::make_shared<v6::Assign>(read_value, variable);
    model->add_sinks({assign});
    model->add_variables({variable});
    return true;
}


bool ov::genai::MakeVocabDecoderSatateful::run_on_model(const std::shared_ptr<ov::Model>& model) {
    std::shared_ptr<ov::Node> vocab_decoder_node;
    for (auto node: model->get_ordered_ops()) {
        if (strcmp(node->get_type_info().name, "VocabDecoder") == 0)
            vocab_decoder_node = node;
    }

    if (!vocab_decoder_node || vocab_decoder_node->get_input_size() < 5)
        return false;
    if (!vocab_decoder_node->input_value(4).get_element_type().is_integral_number())
        return false;
    
    std::shared_ptr<v0::Constant> skip_tokens_const = std::dynamic_pointer_cast<v0::Constant>(vocab_decoder_node->get_input_node_shared_ptr(4));
    std::shared_ptr<v8::Slice> skip_tokens_slice = std::dynamic_pointer_cast<v8::Slice>(vocab_decoder_node->get_input_node_shared_ptr(4));
    if (!skip_tokens_const && !skip_tokens_slice)
        return false;

    auto start_const = std::make_shared<v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector{0});
    auto int_max_const = std::make_shared<v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector{std::numeric_limits<int>::max()});
    auto one_const = std::make_shared<v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector{1});
    
    // By default, INT_MAX will multiply with 1 and all skip_tokens will be selected.
    op::util::VariableInfo var_info{ov::Shape{1}, ov::element::i32, SKIP_SPECIAL_TOKENS_VAR_ID};
    auto variable = std::make_shared<op::util::Variable>(var_info);
    auto read_value = std::make_shared<v6::ReadValue>(one_const, variable);
    // if flag is set, then slice up to the int_max which means skip all tokens.
    auto stop = std::make_shared<v1::Multiply>(int_max_const, read_value);

    // If already has slice just replace the stop input.
    if (skip_tokens_slice) {
        skip_tokens_slice->input(2).replace_source_output(stop);
    } else {
        std::shared_ptr<v8::Slice> slice_node = std::make_shared<v8::Slice>(skip_tokens_const, start_const, stop, one_const);
        vocab_decoder_node->input(4).replace_source_output(slice_node->output(0));
    }
    
    auto assign = std::make_shared<v6::Assign>(read_value, variable);
    model->add_sinks({assign});
    model->add_variables({variable});
    return true;
}

class ReadPadRightAttributes : public ov::AttributeVisitor {
private:
    bool m_pad_right = true;
public:
    void on_adapter(const std::string& name, ov::ValueAccessor<void>& adapter) override {
        if (name != "pad_right") {
            return;
        }
        if (auto a = ov::as_type<ov::AttributeAdapter<bool>>(&adapter)) {
            m_pad_right = a->get();
        }
    }

    bool get_pad_right() const {
        return m_pad_right;
    }
};

namespace {

// Walk back from a named model output (input_ids / attention_mask) to the Slice that
// truncates it. The SentencePiece backend emits: ... -> Slice -> Convert -> Result.
std::shared_ptr<v8::Slice> find_truncation_slice(const std::shared_ptr<ov::Model>& model, const std::string& output_name) {
    for (const auto& output : model->outputs()) {
        if (output.get_names().count(output_name) == 0) {
            continue;
        }
        auto node = output.get_node_shared_ptr();
        for (int hop = 0; hop < 6 && node; ++hop) {
            if (auto slice = ov::as_type_ptr<v8::Slice>(node)) {
                return slice;
            }
            if (node->get_input_size() == 0) {
                break;
            }
            node = node->get_input_node_shared_ptr(0);
        }
    }
    return nullptr;
}

// SentencePiece-form tokenizers have no CombineSegments; truncation is a constant Slice on
// the already-assembled dense tensor (hf_parser.py convert_sentencepiece_model_tokenizer):
//   right padding: start=[0],           stop=[max_length]
//   left  padding: start=[-max_length], stop=[INT_MAX]
// MakePaddingSatateful can't hook that, so max_length is silently ignored. Make only the
// truncation bound settable via MAX_LENGTH_VAR_ID; default stays the baked model_max_length,
// so behavior is unchanged unless the caller sets max_length. No TRUNCATION_VAR_ID here:
// set_state_if_necessary forces truncation=false by default, which would drop the baked cap.
bool make_sp_truncation_stateful(const std::shared_ptr<ov::Model>& model) {
    auto ids_slice = find_truncation_slice(model, "input_ids");
    auto mask_slice = find_truncation_slice(model, "attention_mask");
    std::vector<std::shared_ptr<v8::Slice>> slices;
    if (ids_slice) {
        slices.push_back(ids_slice);
    }
    if (mask_slice && mask_slice != ids_slice) {
        slices.push_back(mask_slice);
    }
    if (slices.empty()) {
        return false;
    }

    auto start_const = ov::as_type_ptr<v0::Constant>(slices[0]->get_input_node_shared_ptr(1));
    auto stop_const = ov::as_type_ptr<v0::Constant>(slices[0]->get_input_node_shared_ptr(2));
    if (!start_const || !stop_const) {
        return false;
    }
    const int64_t start_v = start_const->cast_vector<int64_t>().at(0);
    const int64_t stop_v = stop_const->cast_vector<int64_t>().at(0);

    // Right padding truncates via stop (input 2); left padding via a negative start (input 1).
    bool right_padding;
    size_t bound_input;
    int64_t baked_max_length;
    std::shared_ptr<v0::Constant> bound_const;
    if (start_v == 0 && stop_v > 0 && stop_v < std::numeric_limits<int32_t>::max()) {
        right_padding = true;
        bound_input = 2;
        baked_max_length = stop_v;
        bound_const = stop_const;
    } else if (start_v < 0) {
        right_padding = false;
        bound_input = 1;
        baked_max_length = -start_v;
        bound_const = start_const;
    } else {
        return false;  // unrecognized truncation form, leave untouched
    }

    // The state must be i32: set_state_value writes an i32 tensor for max_length, and the CPU
    // plugin does not support an i64 ReadValue. Convert to the Slice bound's element type afterwards.
    const auto bound_type = bound_const->get_output_element_type(0);
    const auto eshape = bound_const->get_output_shape(0);
    op::util::VariableInfo var_info{eshape, ov::element::i32, ov::genai::MAX_LENGTH_VAR_ID};
    auto max_length_var = std::make_shared<op::util::Variable>(var_info);
    auto default_const =
        std::make_shared<v0::Constant>(ov::element::i32, eshape, std::vector<int32_t>{static_cast<int32_t>(baked_max_length)});
    auto max_length_rv = std::make_shared<v6::ReadValue>(default_const, max_length_var);
    model->add_sinks({std::make_shared<v6::Assign>(max_length_rv, max_length_var)});
    model->add_variables({max_length_var});

    std::shared_ptr<ov::Node> bound = max_length_rv;
    if (!right_padding) {
        auto minus_one = std::make_shared<v0::Constant>(ov::element::i32, eshape, std::vector<int32_t>{-1});
        bound = std::make_shared<v1::Multiply>(bound, minus_one);
    }
    if (bound_type != ov::element::i32) {
        bound = std::make_shared<v0::Convert>(bound, bound_type);
    }
    for (const auto& slice : slices) {
        slice->input(bound_input).replace_source_output(bound->output(0));
    }
    return true;
}

}  // namespace

bool ov::genai::MakePaddingSatateful::run_on_model(const std::shared_ptr<ov::Model>& model) {
    std::shared_ptr<ov::Node> combine_seg_node;
    for (auto node: model->get_ordered_ops()) {
        if (strcmp(node->get_type_name(), "CombineSegments") == 0) {
            combine_seg_node = node;
        }
    }
    if (!combine_seg_node) { return make_sp_truncation_stateful(model); }
    auto num_comb = combine_seg_node->get_input_size();
    
    size_t num_segments = (combine_seg_node->get_input_size() - 1) / 3;
    size_t number_of_main_tokens_inputs = 0;
    std::shared_ptr<Node> add_or_sub_node;
    std::shared_ptr<Node> trunc_node;
    for (size_t i = 0; i < num_segments; i++) {
        // Check all ends inputs of CombineSegments node.
        // For special tokens they are Constant/Select, 
        // for the ends input with main tokens sequence it's Add/Subtract.
        // If  Add then it's a right truncation, if Subtract then it's a left truncation.
        // For left truncation subtract is inserted on 0th input.
        auto tmp_sub_node = combine_seg_node->input_value(3*i + 0).get_node_shared_ptr();
        auto tmp_add_node = combine_seg_node->input_value(3*i + 1).get_node_shared_ptr();
        if (ov::as_type_ptr<v1::Add>(tmp_add_node)) {
            number_of_main_tokens_inputs += 1;
            add_or_sub_node = tmp_add_node;
        } else if (ov::as_type_ptr<v1::Subtract>(tmp_sub_node)) {
            number_of_main_tokens_inputs += 1;
            add_or_sub_node = tmp_sub_node;
        } else if (strcmp(tmp_add_node->get_type_name(), "Truncate") == 0) {
            number_of_main_tokens_inputs += 1;
            trunc_node = tmp_add_node;
            // If it't truncate as a single operation
        }
    }
    
    // Exit if couldn't find main input or there are several.
    if (number_of_main_tokens_inputs != 1 && number_of_main_tokens_inputs != 2) { return false; }
    
    std::shared_ptr<ov::op::v0::Constant> const_node;
    if (add_or_sub_node) {
        // Minimum between max_length and length of token sequence.
        auto min_node = ov::as_type_ptr<v1::Minimum>(add_or_sub_node->get_input_node_shared_ptr(1));
        if (!min_node) { 
            return false; 
        }
        // constant containing final max_length - num_added tokens at the end of pipeline.
        const_node = ov::as_type_ptr<v0::Constant>(min_node->get_input_node_shared_ptr(1));
        if (!const_node) { 
            return false; 
        }

    } else if (trunc_node) {
        // If truncation is done by Truncate node then we need to check if it has a constant input.
        const_node = ov::as_type_ptr<v0::Constant>(trunc_node->get_input_node_shared_ptr(number_of_main_tokens_inputs*3));
    } else {
        return false;
    }

    op::util::VariableInfo var_info{const_node->get_output_shape(0), const_node->get_output_element_type(0), ov::genai::MAX_LENGTH_VAR_ID};
    auto max_length_var = std::make_shared<op::util::Variable>(var_info);

    size_t num_added_tokens = num_segments - number_of_main_tokens_inputs;
    // Constant which stores number of added_tokens.
    auto num_added_tokens_const = std::make_shared<v0::Constant>(
        const_node->get_output_element_type(0), const_node->get_output_shape(0), std::vector{num_added_tokens});
    
    OPENVINO_ASSERT(const_node->get_element_type() == element::i32);
    auto values = const_node->get_vector<int32_t>();
    OPENVINO_ASSERT(values.size() == 1);
    // Since const_node contain value = max_length - num_added tokens, 
    size_t default_max_length = values[0] + num_added_tokens;

    auto default_max_length_const = std::make_shared<v0::Constant>(
        const_node->get_output_element_type(0), const_node->get_output_shape(0), std::vector{default_max_length});

    // Save targets before adding new target with ReadValue to avoid recursion.
    auto target_inputs = const_node->output(0).get_target_inputs();
    auto max_length_rv = std::make_shared<v6::ReadValue>(default_max_length_const, max_length_var);
    model->add_sinks({std::make_shared<v6::Assign>(max_length_rv, max_length_var)});
    model->add_variables({max_length_var});
    auto max_length_node = std::make_shared<v1::Subtract>(max_length_rv, num_added_tokens_const);

    var_info = {ov::Shape{}, ov::element::boolean, ov::genai::TRUNCATION_VAR_ID};
    auto truncation_var = std::make_shared<op::util::Variable>(var_info);
    auto defaul_false_const = std::make_shared<v0::Constant>(ov::element::boolean, ov::Shape{}, std::vector{false});
    auto truncation_rv = std::make_shared<v6::ReadValue>(defaul_false_const, truncation_var);
    auto truncation_assign = std::make_shared<v6::Assign>(truncation_rv, truncation_var);
    model->add_sinks({truncation_assign});
    model->add_variables({truncation_var});
    
    // TODO: int32_max 2147483647 becomes -2147483648 when accessed from Truncate be inputs[inputs.size() - 3].data<const int32_t>()[0]
    // int32_max - 1 (-2, -3, etc.) also strangely becomes still -2147483648.
    // Only starting from int32_max - 64 it is casted to adequate positive value.
    auto int32_max_constant = std::make_shared<v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int32_t>{std::numeric_limits<int32_t>::max() - 64});
    auto max_length_for_trunc = std::make_shared<v1::Select>(truncation_rv, max_length_node, int32_max_constant);
    
    for (auto target_input : target_inputs) {
        target_input.replace_source_output(max_length_for_trunc->output(0));
    }
    
    // We need to check if user requested to not add special tokens.
    std::shared_ptr<v6::ReadValue> read_value_spec_tokens;
    for (const auto& sink : model->get_sinks()) {
        // Check if sink accepts input from Assign, and if that't the case get the ReadValus node input.
        if (auto read_value = ov::as_type_ptr<v6::ReadValue>(sink->get_input_node_shared_ptr(0))) {
            if (read_value->get_variable()->get_info().variable_id == ov::genai::ADD_SPECIAL_TOKENS_VAR_ID) {
                read_value_spec_tokens = read_value;
                break;
            }
        }
    }

    // If user requested to not add special tokens in order to correctly calculate 
    // truncation we need to enforce num_added_tokens to 0 regardless the hardcoded value of Constant.
    auto zero_constant = std::make_shared<v0::Constant>(ov::element::i32, ov::Shape{}, std::vector{0});
    if (read_value_spec_tokens && num_added_tokens_const) {
        auto select_node = std::make_shared<v1::Select>(read_value_spec_tokens, num_added_tokens_const, zero_constant);
        max_length_node->input(1).replace_source_output(select_node->output(0));
    }

    std::vector<std::shared_ptr<ov::Node>> ragged_to_dense_nodes;
    for (auto node: model->get_ordered_ops()) {
        if (strcmp(node->get_type_info().name, "RaggedToDense") == 0) {
            ragged_to_dense_nodes.emplace_back(node);
        }
    }

    if (ragged_to_dense_nodes.size() < 1) {
        return true;  // true since at this point we already have modified the graph.s
    }
    
    // By default do not pad to max_length
    auto pad_to_max_length_var = std::make_shared<op::util::Variable>(op::util::VariableInfo{ov::Shape{1}, ov::element::boolean, ov::genai::PAD_TO_MAX_LENGTH_VAR_ID});
    auto default_false_const = std::make_shared<v0::Constant>(ov::element::boolean, ov::Shape{1}, std::vector{false});
    auto pad_to_max_length_rv = std::make_shared<v6::ReadValue>(default_false_const, pad_to_max_length_var);
    auto select_node = std::make_shared<v1::Select>(pad_to_max_length_rv, max_length_rv, zero_constant);

    // If user called encode without explicitly stating padding side, then we should pad it to the default side.
    // Here we get that side from the RaggedToDense nodes attribute. 
    auto pad_right_attr_visitor = ReadPadRightAttributes();
    bool first_iter = true;
    bool default_pad_right = true;
    for (auto ragged_to_dense_node : ragged_to_dense_nodes) {
        if (!ragged_to_dense_node) {
            return true;  // true since at this point we already have modified the graph.
        }
        ragged_to_dense_node->visit_attributes(pad_right_attr_visitor);
        if (first_iter) {
            default_pad_right = pad_right_attr_visitor.get_pad_right();
        } else if (pad_right_attr_visitor.get_pad_right() != default_pad_right) {
            return true;  // true since at this point we already have modified the graph.
        }
        first_iter = false;
    }

    // Add padding side variable.
    auto pad_right_var = std::make_shared<op::util::Variable>(op::util::VariableInfo{ov::Shape{}, ov::element::boolean, ov::genai::PAD_RIGHT_VAR_ID});
    auto pad_right_const = std::make_shared<v0::Constant>(ov::element::boolean, ov::Shape{}, std::vector{default_pad_right});
    auto pad_right_rv = std::make_shared<v6::ReadValue>(pad_right_const, pad_right_var);
    
    // This cycle cannot be united with the cycle above since first we need to ensure that all RaggedToDense nodes have the same padding side
    // and only after that start to modify. Therefore we need to iterate over RaggedToDense nodes twice. In 99% of cases there is only one RaggedToDense node
    // and in the rest of cases it would be two RaggedToDense nodes with the same padding side if they are created by the openvino_tokenizers.
    for (auto ragged_to_dense_node : ragged_to_dense_nodes) {
        if (!ragged_to_dense_node) {
            return true;  // true since at this point we already have modified the graph.
        }

        auto new_inputs = ragged_to_dense_node->input_values();
        new_inputs.emplace_back(pad_right_rv->output(0));
        auto new_ragged_to_dense = ragged_to_dense_node->clone_with_new_inputs(new_inputs);
        
        auto max_op = std::make_shared<v1::Maximum>(new_ragged_to_dense->input_value(3), select_node);
        new_ragged_to_dense->input(3).replace_source_output(max_op->output(0));
        
        ov::replace_node(ragged_to_dense_node, new_ragged_to_dense);
    }

    model->add_sinks({std::make_shared<v6::Assign>(pad_right_rv, pad_right_var)});
    model->add_variables({pad_right_var});
    model->add_sinks({std::make_shared<v6::Assign>(pad_to_max_length_rv, pad_to_max_length_var)});
    model->add_variables({pad_to_max_length_var});

    return true;
}
