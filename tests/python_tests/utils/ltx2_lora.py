# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import openvino as ov

# Modules covered by the tiny LTX-2 LoRA: diffusers name -> original (Lightricks / ComfyUI) name
LTX2_LORA_TRANSFORMER_MODULES = {
    "transformer_blocks.0.attn1.to_q": "transformer_blocks.0.attn1.to_q",
    "transformer_blocks.0.attn2.to_out.0": "transformer_blocks.0.attn2.to_out.0",
    "transformer_blocks.0.audio_attn1.to_k": "transformer_blocks.0.audio_attn1.to_k",
    "transformer_blocks.0.audio_to_video_attn.to_v": "transformer_blocks.0.audio_to_video_attn.to_v",
    "transformer_blocks.0.ff.net.0.proj": "transformer_blocks.0.ff.net.0.proj",
    "transformer_blocks.0.audio_ff.net.2": "transformer_blocks.0.audio_ff.net.2",
    "proj_in": "patchify_proj",
    "audio_proj_in": "audio_patchify_proj",
    "time_embed.linear": "adaln_single.linear",
    "audio_time_embed.linear": "audio_adaln_single.linear",
    "av_cross_attn_video_scale_shift.linear": "av_ca_video_scale_shift_adaln_single.linear",
    "caption_projection.linear_1": "caption_projection.linear_1",
}
LTX2_LORA_CONNECTORS_MODULES = {"text_proj_in": "aggregate_embed"}


def make_ltx2_lora(model_dir, path, original_format, rank=4, seed=0):
    """Writes a random LoRA for a tiny LTX-2 export, in diffusers or original key format"""
    from safetensors.numpy import save_file

    rng = np.random.default_rng(seed)
    tensors = {}
    parts = [
        ("transformer", "self.model.", LTX2_LORA_TRANSFORMER_MODULES, "diffusion_model"),
        ("connectors", "self.", LTX2_LORA_CONNECTORS_MODULES, "text_embedding_projection"),
    ]
    for subfolder, weight_prefix, modules, original_prefix in parts:
        model = ov.Core().read_model(Path(model_dir) / subfolder / "openvino_model.xml")
        shapes = {
            op.get_friendly_name().removesuffix("_compressed"): op.get_output_shape(0)
            for op in model.get_ops()
            if op.get_type_name() == "Constant"
        }
        for diffusers_name, original_name in modules.items():
            out_features, in_features = shapes[f"{weight_prefix}{diffusers_name}.weight"]
            key = f"{original_prefix}.{original_name}" if original_format else f"{subfolder}.{diffusers_name}"
            tensors[f"{key}.lora_A.weight"] = rng.standard_normal((rank, in_features), dtype=np.float32)
            tensors[f"{key}.lora_B.weight"] = rng.standard_normal((out_features, rank), dtype=np.float32)
    save_file(tensors, str(path))
    return str(path)
