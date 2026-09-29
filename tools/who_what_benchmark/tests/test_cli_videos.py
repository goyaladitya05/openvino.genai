import sys
from pathlib import Path
import pytest
from test_cli_image import get_similarity
from conftest import convert_model, run_wwb


@pytest.mark.xfail(sys.platform == "darwin", reason="Not enough memory on macOS CI runners. Ticket CVS-179749")
@pytest.mark.xfail(sys.platform == "win32", reason="Access violation in OVLTXPipeline on Windows. Ticket CVS-179750")
@pytest.mark.parametrize(
    ("model_id", "model_type", "genai_args"),
    [
        (
            "optimum-intel-internal-testing/tiny-random-ltx-video",
            "text-to-video",
            ["--taylorseer-config", '{"disable_cache_after_step": 0}'],
        ),
        ("optimum-intel-internal-testing/tiny-random-ltx2", "text-to-video", []),
    ],
)
def test_video_model_genai(model_id, model_type, genai_args, tmp_path):
    GT_FILE = tmp_path / "gt.csv"
    MODEL_PATH = convert_model(model_id)

    run_wwb(
        [
            "--base-model",
            model_id,
            "--num-samples",
            "1",
            "--gt-data",
            GT_FILE,
            "--device",
            "CPU",
            "--model-type",
            model_type,
            "--num-inference-steps",
            "2",
            "--video-frames-num",
            "9",
        ]
    )
    assert GT_FILE.exists()
    assert (tmp_path / "reference").exists()

    output = run_wwb(
        [
            "--target-model",
            MODEL_PATH,
            "--num-samples",
            "1",
            "--gt-data",
            GT_FILE,
            "--device",
            "CPU",
            "--model-type",
            model_type,
            "--genai",
            "--num-inference-steps",
            "2",
            "--video-frames-num",
            "9",
            "--output",
            tmp_path,
            *genai_args,
        ]
    )

    assert "Metrics for model" in output
    similarity = get_similarity(output)
    assert similarity >= 0.88
    assert (tmp_path / "target").exists()

    # test w/o models
    run_wwb(
        [
            "--target-data",
            tmp_path / "target.csv",
            "--num-samples",
            "1",
            "--gt-data",
            GT_FILE,
            "--device",
            "CPU",
            "--model-type",
            model_type,
            "--num-inference-steps",
            "2",
            "--video-frames-num",
            "9",
        ]
    )


@pytest.mark.xfail(sys.platform == "darwin", reason="Not enough memory on macOS CI runners. Ticket CVS-179749")
@pytest.mark.xfail(sys.platform == "win32", reason="Access violation in OVLTXPipeline on Windows. Ticket CVS-179750")
@pytest.mark.parametrize(
    ("model_id", "model_type"),
    [("optimum-intel-internal-testing/tiny-random-ltx-video", "text-to-video")],
)
def test_video_model_genai_with_taylorseer(model_id, model_type, tmp_path):
    GT_FILE = tmp_path / "gt.csv"
    MODEL_PATH = convert_model(model_id)

    run_wwb(
        [
            "--base-model",
            model_id,
            "--num-samples",
            "1",
            "--gt-data",
            GT_FILE,
            "--device",
            "CPU",
            "--model-type",
            model_type,
            "--num-inference-steps",
            "4",
            "--video-frames-num",
            "9",
        ]
    )
    assert GT_FILE.exists()
    assert (tmp_path / "reference").exists()

    # Test with full TaylorSeer config
    output = run_wwb(
        [
            "--target-model",
            MODEL_PATH,
            "--num-samples",
            "1",
            "--gt-data",
            GT_FILE,
            "--device",
            "CPU",
            "--model-type",
            model_type,
            "--genai",
            "--num-inference-steps",
            "4",
            "--video-frames-num",
            "9",
            "--output",
            tmp_path,
            "--taylorseer-config",
            '{"cache_interval": 3, "disable_cache_before_step": 4, "disable_cache_after_step": -1}',
        ]
    )

    assert "Metrics for model" in output
    assert "TaylorSeer config:" in output
    similarity = get_similarity(output)
    assert similarity >= 0.97


@pytest.mark.xfail(sys.platform == "darwin", reason="Not enough memory on macOS CI runners. Ticket CVS-179749")
@pytest.mark.xfail(sys.platform == "win32", reason="Access violation in OVLTXPipeline on Windows. Ticket CVS-179750")
@pytest.mark.parametrize(
    ("model_id", "model_type"),
    [
        ("optimum-intel-internal-testing/tiny-random-ltx-video", "image-to-video"),
        ("optimum-intel-internal-testing/tiny-random-ltx2", "image-to-video"),
    ],
)
def test_image2video_model_genai(model_id, model_type, tmp_path):
    GT_FILE = tmp_path / "gt.csv"
    MODEL_PATH = convert_model(model_id)

    run_wwb(
        [
            "--base-model",
            model_id,
            "--num-samples",
            "1",
            "--gt-data",
            GT_FILE,
            "--device",
            "CPU",
            "--model-type",
            model_type,
            "--hf",
            "--num-inference-steps",
            "2",
            "--video-frames-num",
            "9",
        ]
    )
    assert GT_FILE.exists()
    assert (tmp_path / "reference").exists()

    output = run_wwb(
        [
            "--target-model",
            MODEL_PATH,
            "--num-samples",
            "1",
            "--gt-data",
            GT_FILE,
            "--device",
            "CPU",
            "--model-type",
            model_type,
            "--genai",
            "--num-inference-steps",
            "2",
            "--video-frames-num",
            "9",
            "--output",
            tmp_path,
        ]
    )

    assert "Metrics for model" in output
    similarity = get_similarity(output)
    assert similarity >= 0.88
    assert (tmp_path / "target").exists()

    # test w/o models
    run_wwb(
        [
            "--target-data",
            tmp_path / "target.csv",
            "--num-samples",
            "1",
            "--gt-data",
            GT_FILE,
            "--device",
            "CPU",
            "--model-type",
            model_type,
            "--num-inference-steps",
            "2",
            "--video-frames-num",
            "9",
        ]
    )


def make_ltx2_lora(model_path, lora_file, rank=4):
    """Random LoRA in the original LTX-2 key format, so both HF and GenAI go through their key conversion"""
    import numpy as np
    import openvino as ov
    from safetensors.numpy import save_file

    modules = {
        "transformer": (
            "self.model.",
            "diffusion_model",
            {
                "transformer_blocks.0.attn1.to_q": "transformer_blocks.0.attn1.to_q",
                "transformer_blocks.0.audio_attn1.to_k": "transformer_blocks.0.audio_attn1.to_k",
                "transformer_blocks.0.ff.net.0.proj": "transformer_blocks.0.ff.net.0.proj",
                "proj_in": "patchify_proj",
                "time_embed.linear": "adaln_single.linear",
            },
        ),
        "connectors": ("self.", "text_embedding_projection", {"text_proj_in": "aggregate_embed"}),
    }
    rng = np.random.default_rng(0)
    tensors = {}
    for subfolder, (weight_prefix, lora_prefix, names) in modules.items():
        model = ov.Core().read_model(Path(model_path) / subfolder / "openvino_model.xml")
        shapes = {
            op.get_friendly_name().removesuffix("_compressed"): op.get_output_shape(0)
            for op in model.get_ops()
            if op.get_type_name() == "Constant"
        }
        for weight_name, lora_name in names.items():
            out_features, in_features = shapes[f"{weight_prefix}{weight_name}.weight"]
            key = f"{lora_prefix}.{lora_name}"
            tensors[f"{key}.lora_A.weight"] = rng.standard_normal((rank, in_features), dtype=np.float32)
            tensors[f"{key}.lora_B.weight"] = rng.standard_normal((out_features, rank), dtype=np.float32)
    save_file(tensors, str(lora_file))
    return lora_file


def run_test_with_lora(model_id, model_type, tmp_path, *, genai_threshold):
    if sys.platform == "darwin":
        pytest.xfail("Not enough memory on macOS CI runners. Ticket CVS-179749")
    if sys.platform == "win32":
        pytest.xfail("Access violation in OVLTXPipeline on Windows. Ticket CVS-179750")

    from ov_utils import get_ov_cache_dir, download_hf_files_to_cache

    gt_file = tmp_path / "gt.csv"
    model_path = convert_model(model_id)

    if "ltx2" in model_id:
        lora_file = make_ltx2_lora(model_path, tmp_path / "ltx2_lora.safetensors")
    else:
        lora_cache_dir = get_ov_cache_dir() / "test_data" / "ltx_tiny_dummy_lora"
        lora_dir = download_hf_files_to_cache(
            "goyaladitya05/openvino-genai-test-files",
            lora_cache_dir,
            ["ltx_tiny_dummy_lora.safetensors"],
        )
        lora_file = lora_dir / "ltx_tiny_dummy_lora.safetensors"
    assert lora_file.exists(), f"LoRA adapter wasn't found: {lora_file}"

    # 1) Generate GT using HF + LoRA
    run_wwb(
        [
            "--base-model",
            model_id,
            "--num-samples",
            "1",
            "--gt-data",
            gt_file,
            "--device",
            "CPU",
            "--model-type",
            model_type,
            "--num-inference-steps",
            "2",
            "--video-frames-num",
            "9",
            "--adapters",
            str(lora_file),
            "--alphas",
            "0.9",
            "--hf",
        ]
    )
    assert gt_file.exists(), f"GT wasn't generated: {gt_file}"
    assert (tmp_path / "reference").exists()

    # 2) Target: GenAI + LoRA
    outputs_genai = tmp_path / "genai_lora"
    out_genai = run_wwb(
        [
            "--target-model",
            model_path,
            "--num-samples",
            "1",
            "--gt-data",
            gt_file,
            "--device",
            "CPU",
            "--model-type",
            model_type,
            "--genai",
            "--num-inference-steps",
            "2",
            "--video-frames-num",
            "9",
            "--adapters",
            str(lora_file),
            "--alphas",
            "0.9",
            "--output",
            outputs_genai,
        ]
    )

    assert "Metrics for model" in out_genai
    assert (outputs_genai / "target").exists()
    similarity = get_similarity(out_genai)
    assert similarity >= genai_threshold

    # 3) GenAI + LoRA at load time, empty AdapterConfig at generate time
    outputs_empty = tmp_path / "genai_empty_adapters"
    out_empty = run_wwb(
        [
            "--target-model",
            model_path,
            "--num-samples",
            "1",
            "--gt-data",
            gt_file,
            "--device",
            "CPU",
            "--model-type",
            model_type,
            "--genai",
            "--num-inference-steps",
            "2",
            "--video-frames-num",
            "9",
            "--adapters",
            str(lora_file),
            "--alphas",
            "0.9",
            "--empty_adapters",
            "--output",
            outputs_empty,
        ]
    )
    assert "Metrics for model" in out_empty
    assert (outputs_empty / "target").exists()


@pytest.mark.parametrize(
    ("model_id", "model_type"),
    [
        ("optimum-intel-internal-testing/tiny-random-ltx-video", "text-to-video"),
        ("optimum-intel-internal-testing/tiny-random-ltx2", "text-to-video"),
        ("optimum-intel-internal-testing/tiny-random-ltx2", "image-to-video"),
    ],
)
def test_video_model_genai_with_lora(model_id, model_type, tmp_path):
    run_test_with_lora(model_id, model_type, tmp_path, genai_threshold=0.88)
