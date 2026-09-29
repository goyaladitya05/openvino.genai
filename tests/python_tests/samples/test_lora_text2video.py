# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import sys

import numpy as np

from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR
from test_utils import run_sample, compare_videos
from utils.ltx2_lora import make_ltx2_lora


class TestLoraText2Video:
    PROMPT = "A woman with long brown hair smiles at another woman with long blonde hair"

    def _run_and_compare(self, model, lora, tmp_path, has_audio):
        py_dir = tmp_path / "python_output"
        cpp_dir = tmp_path / "cpp_output"
        py_dir.mkdir()
        cpp_dir.mkdir()

        py_script = SAMPLES_PY_DIR / "video_generation/lora_text2video.py"
        py_command = [sys.executable, py_script, model, self.PROMPT, "5", lora, "0.7"]
        run_sample(py_command, cwd=str(py_dir))

        cpp_sample = SAMPLES_CPP_DIR / "lora_text2video"
        cpp_command = [cpp_sample, model, self.PROMPT, "5", lora, "0.7"]
        run_sample(cpp_command, cwd=str(cpp_dir))

        for name in ("lora", "baseline"):
            py_video = py_dir / f"{name}_video.avi"
            cpp_video = cpp_dir / f"{name}_video.avi"

            assert py_video.exists(), f"Python video not found: {py_video}"
            assert cpp_video.exists(), f"C++ video not found: {cpp_video}"
            assert compare_videos(py_video, cpp_video), f"{name} video from Python and C++ samples are not identical"

            py_audio = py_dir / f"{name}_audio.wav"
            cpp_audio = cpp_dir / f"{name}_audio.wav"
            assert py_audio.exists() == has_audio, f"Unexpected audio output presence: {py_audio}"
            assert cpp_audio.exists() == has_audio, f"Unexpected audio output presence: {cpp_audio}"
            if has_audio:
                import soundfile as sf

                py_data, py_rate = sf.read(py_audio, dtype="float32")
                cpp_data, cpp_rate = sf.read(cpp_audio, dtype="float32")
                assert py_rate == cpp_rate, f"{name} audio sample rates from Python and C++ samples differ"
                assert np.array_equal(py_data, cpp_data), f"{name} audio from Python and C++ samples is not identical"

    @pytest.mark.samples
    @pytest.mark.video_generation
    @pytest.mark.parametrize("convert_model", ["tiny-random-ltx-video"], indirect=True)
    @pytest.mark.parametrize("download_test_content", ["ltx_tiny_dummy_lora.safetensors"], indirect=True)
    def test_sample_lora_text2video(self, convert_model, download_test_content, tmp_path):
        self._run_and_compare(convert_model, download_test_content, tmp_path, has_audio=False)

    @pytest.mark.samples
    @pytest.mark.video_generation
    @pytest.mark.parametrize("convert_model", ["tiny-random-ltx2"], indirect=True)
    def test_sample_lora_text2video_ltx2(self, convert_model, tmp_path):
        lora = make_ltx2_lora(convert_model, tmp_path / "ltx2_lora.safetensors", original_format=True)
        self._run_and_compare(convert_model, lora, tmp_path, has_audio=True)
