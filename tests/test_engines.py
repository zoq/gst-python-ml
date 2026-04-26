# test_engines.py
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Library General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Library General Public License for more details.
#
# You should have received a copy of the GNU Library General Public
# License along with this library; if not, write to the
# Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
# Boston, MA 02110-1301, USA.

"""Tests for ML engine instantiation and basic operations.

Each test is skipped if the engine's dependency is not installed.
Run with: pytest tests/test_engines.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure plugins are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "plugins" / "python"))

from engine.engine_factory import EngineFactory


# ── Helpers ──


def _skip_unless_importable(module_name, pip_name=None):
    """Skip test if module is not importable."""
    try:
        __import__(module_name)
    except ImportError:
        pip_name = pip_name or module_name
        pytest.skip(f"{pip_name} not installed")


def _make_dummy_frame(width=224, height=224):
    """Create a dummy RGB frame as uint8 numpy array."""
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)


# ── PyTorch ──


class TestPyTorchEngine:
    def setup_method(self):
        _skip_unless_importable("torch")
        self.engine = EngineFactory.create("pytorch")

    def test_create(self):
        assert self.engine is not None

    def test_set_device_cpu(self):
        self.engine.do_set_device("cpu")

    def test_load_resnet18(self):
        result = self.engine.do_load_model("resnet18")
        assert result is True

    def test_forward(self):
        self.engine.do_set_device("cpu")
        self.engine.do_load_model("resnet18")
        frame = _make_dummy_frame()
        result = self.engine.do_forward(frame)
        assert result is not None


# ── ONNX ──


class TestONNXEngine:
    def setup_method(self):
        _skip_unless_importable("onnxruntime")
        self.engine = EngineFactory.create("onnx")

    def test_create(self):
        assert self.engine is not None

    def test_set_device_cpu(self):
        self.engine.do_set_device("cpu")


# ── OpenVINO ──


class TestOpenVinoEngine:
    def setup_method(self):
        _skip_unless_importable("openvino")
        self.engine = EngineFactory.create("openvino")

    def test_create(self):
        assert self.engine is not None

    def test_set_device_cpu(self):
        self.engine.do_set_device("cpu")


# ── LiteRT (TFLite) ──


class TestLiteRTEngine:
    def setup_method(self):
        _skip_unless_importable("tflite_runtime", "tflite-runtime")
        self.engine = EngineFactory.create("tflite")

    def test_create(self):
        assert self.engine is not None


# ── TensorFlow ──


class TestTensorFlowEngine:
    def setup_method(self):
        _skip_unless_importable("tensorflow")
        self.engine = EngineFactory.create("tensorflow")

    def test_create(self):
        assert self.engine is not None


# ── TVM ──


class TestTVMEngine:
    def setup_method(self):
        _skip_unless_importable("tvm", "apache-tvm")
        self.engine = EngineFactory.create("tvm")

    def test_create(self):
        assert self.engine is not None


# ── tinygrad ──


class TestTinyGradEngine:
    def setup_method(self):
        _skip_unless_importable("tinygrad")
        self.engine = EngineFactory.create("tinygrad")

    def test_create(self):
        assert self.engine is not None

    def test_set_device_cpu(self):
        self.engine.do_set_device("cpu")

    def test_load_resnet18(self):
        _skip_unless_importable("torch")
        result = self.engine.do_load_model("resnet18")
        assert result is True

    @pytest.mark.xfail(
        reason="tinygrad state-dict forward requires model graph reconstruction"
    )
    def test_forward(self):
        _skip_unless_importable("torch")
        self.engine.do_set_device("cpu")
        self.engine.do_load_model("resnet18")
        frame = _make_dummy_frame()
        result = self.engine.do_forward(frame)
        assert result is not None


# ── MLX ──


class TestMLXEngine:
    def setup_method(self):
        _skip_unless_importable("mlx")
        self.engine = EngineFactory.create("mlx")

    def test_create(self):
        assert self.engine is not None

    def test_set_device_cpu(self):
        self.engine.do_set_device("cpu")


# ── ExecuTorch ──


class TestExecuTorchEngine:
    def setup_method(self):
        _skip_unless_importable("executorch")
        self.engine = EngineFactory.create("executorch")

    def test_create(self):
        assert self.engine is not None

    def test_set_device_cpu(self):
        self.engine.do_set_device("cpu")


# ── llama.cpp ──


class TestLlamaCppEngine:
    def setup_method(self):
        _skip_unless_importable("llama_cpp", "llama-cpp-python")
        self.engine = EngineFactory.create("llamacpp")

    def test_create(self):
        assert self.engine is not None

    def test_set_device_cpu(self):
        self.engine.do_set_device("cpu")


# ── Candle ──


class TestCandleEngine:
    def setup_method(self):
        _skip_unless_importable("candle")
        self.engine = EngineFactory.create("candle")

    def test_create(self):
        assert self.engine is not None


# ── JAX ──


class TestJAXEngine:
    def setup_method(self):
        _skip_unless_importable("jax")
        self.engine = EngineFactory.create("jax")

    def test_create(self):
        assert self.engine is not None

    def test_set_device_cpu(self):
        self.engine.do_set_device("cpu")
