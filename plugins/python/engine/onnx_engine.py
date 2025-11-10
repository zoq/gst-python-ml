# ONNXEngine
# Copyright (C) 2024-2025 Collabora Ltd.
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

import os
import numpy as np
import onnxruntime as ort
import torch
import tempfile
from torchvision import models
from transformers import (
    AutoTokenizer,
    AutoImageProcessor,
)
from optimum.onnxruntime import ORTModelForCausalLM, ORTModelForVision2Seq

from .ml_engine import MLEngine


class ONNXEngine(MLEngine):
    def __init__(self):
        super().__init__()
        self.model = None
        self.session = None
        self.model_type = None
        self.model_name = None
        self.kwargs = None
        self.provider = None
        self.input_names = None
        self.output_names = None

    def do_load_model(self, model_name, **kwargs):
        """Load a pre-trained model by name from TorchVision, Transformers (via Optimum ONNX), or a local ONNX path."""
        processor_name = kwargs.get("processor_name")
        tokenizer_name = kwargs.get("tokenizer_name")
        self.model_name = model_name
        self.kwargs = kwargs

        try:
            if os.path.isfile(model_name) and model_name.endswith(".onnx"):
                self.session = ort.InferenceSession(
                    model_name, providers=[self.provider]
                )
                self.model = self.session
                self.model_type = "custom"
                self.input_names = [inp.name for inp in self.session.get_inputs()]
                self.output_names = [out.name for out in self.session.get_outputs()]
                self.logger.info(f"ONNX model loaded from local path: {model_name}")
                return True
            else:
                if hasattr(models, model_name):
                    pt_model = getattr(models, model_name)(pretrained=True)
                    self.model_type = "classification"
                elif hasattr(models.detection, model_name):
                    pt_model = getattr(models.detection, model_name)(pretrained=True)
                    self.model_type = "detection"
                elif processor_name and tokenizer_name:
                    self.image_processor = AutoImageProcessor.from_pretrained(
                        processor_name
                    )
                    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                    self.model = ORTModelForVision2Seq.from_pretrained(
                        model_name, export=True, provider=self.provider
                    )
                    self.frame_stride = (
                        self.model.config.encoder.num_frames
                        if hasattr(self.model.config.encoder, "num_frames")
                        else 1
                    )
                    self.model_type = "vision_text"
                    self.logger.info(
                        f"Vision-Text model '{model_name}' loaded with processor and tokenizer via Optimum ONNX."
                    )
                    return True
                else:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.model = ORTModelForCausalLM.from_pretrained(
                        model_name, export=True, provider=self.provider
                    )
                    self.model_type = "llm"
                    self.logger.info(
                        f"Pre-trained LLM model '{model_name}' loaded via Optimum ONNX."
                    )
                    return True

                # For TorchVision models, export to ONNX
                pt_model.eval()
                dummy_input = torch.randn(1, 3, 224, 224)
                with tempfile.NamedTemporaryFile(
                    suffix=".onnx", delete=False
                ) as tmp_file:
                    torch.onnx.export(
                        pt_model,
                        dummy_input,
                        tmp_file.name,
                        opset_version=11,
                        input_names=["input"],
                        output_names=(
                            ["output"]
                            if self.model_type == "classification"
                            else ["boxes", "labels", "scores"]
                        ),
                        dynamic_axes=(
                            {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
                            if self.model_type == "classification"
                            else None
                        ),
                    )
                    self.session = ort.InferenceSession(
                        tmp_file.name, providers=[self.provider]
                    )
                os.unlink(tmp_file.name)
                self.model = self.session
                self.input_names = [inp.name for inp in self.session.get_inputs()]
                self.output_names = [out.name for out in self.session.get_outputs()]
                self.logger.info(
                    f"Pre-trained model '{model_name}' exported to ONNX and loaded."
                )

            return True

        except Exception as e:
            self.logger.error(f"Error loading model '{model_name}': {e}")
            self.tokenizer = None
            self.image_processor = None
            self.model = None
            self.session = None
            return False

    def do_set_device(self, device):
        """Set ONNX device for the model."""
        self.device = device
        self.logger.info(f"Setting device to {device}")

        if "cuda" in device:
            if not ort.get_device() == "GPU":
                self.logger.warning("GPU is not available, falling back to CPU")
                self.device = "cpu"
                self.provider = "CPUExecutionProvider"
            else:
                try:
                    index = int(device.split(":")[-1]) if ":" in device else 0
                    self.provider = ("CUDAExecutionProvider", {"device_id": index})
                    self.logger.info(f"GPU device set to cuda:{index}")
                except Exception as e:
                    self.logger.error(f"Failed to set GPU device: {e}")
                    self.logger.warning("Falling back to CPU")
                    self.device = "cpu"
                    self.provider = "CPUExecutionProvider"
        elif device == "cpu":
            self.provider = "CPUExecutionProvider"
        else:
            self.logger.error(f"Invalid device specified: {device}")
            self.device = "cpu"
            self.provider = "CPUExecutionProvider"

        # Reload model if already loaded
        if self.model_name:
            self.do_load_model(self.model_name, **self.kwargs)

    def _forward_classification(self, frames):
        """Handle inference for classification models."""
        is_batch = frames.ndim == 4
        img_array = np.array(frames, dtype=np.float32) / 255.0
        if is_batch:
            img_array = np.transpose(
                img_array, (0, 3, 1, 2)
            )  # (B, H, W, C) -> (B, C, H, W)
        else:
            img_array = np.transpose(img_array, (2, 0, 1))  # (H, W, C) -> (C, H, W)
            img_array = np.expand_dims(img_array, 0)
        inputs = {self.input_names[0]: img_array}
        outputs = self.session.run(self.output_names, inputs)
        preds = outputs[0]
        probs = np.exp(preds) / np.sum(np.exp(preds), axis=1, keepdims=True)
        top_classes = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        results = [
            {"labels": [int(c)], "scores": [float(s)]}
            for c, s in zip(top_classes, confidences)
        ]
        return results[0] if not is_batch else results

    def do_forward(self, frames):
        """Handle inference for different types of models, supporting single frames or batches."""
        is_batch = isinstance(frames, np.ndarray) and frames.ndim == 4
        if not isinstance(frames, (np.ndarray, str)):
            self.logger.error(f"Invalid input type for forward: {type(frames)}")
            return None

        if self.model_type == "vision_text":
            if is_batch:
                self.logger.error(
                    "Batch processing not supported for vision-text models with frame buffering."
                )
                return None
            if not hasattr(self, "counter"):
                self.counter = 0
                self.frame_buffer = []
            self.counter += 1
            if self.counter % self.frame_stride == 0:
                self.frame_buffer.append(frames)
            if len(self.frame_buffer) >= self.batch_size:
                self.logger.info(f"Processing {self.batch_size} frames")
                try:
                    gen_kwargs = {"min_length": 10, "max_length": 20, "num_beams": 8}
                    pixel_values = self.image_processor(
                        self.frame_buffer, return_tensors="pt"
                    ).pixel_values
                    tokens = self.model.generate(pixel_values, **gen_kwargs)
                    captions = self.tokenizer.batch_decode(
                        tokens, skip_special_tokens=True
                    )
                    self.logger.info(f"Captions: {captions}")
                    self.frame_buffer = []
                    return captions[0]
                except Exception as e:
                    self.logger.error(f"Failed to process frames: {e}")
                    self.frame_buffer = []
                    return None
            return None

        elif self.model_type == "llm":
            if is_batch:
                self.logger.error("Batch processing not supported for LLM-only models.")
                return None
            inputs = self.tokenizer(frames, return_tensors="pt")
            generated_tokens = self.model.generate(**inputs)
            generated_text = self.tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )
            self.logger.info(f"Generated text: {generated_text}")
            return generated_text

        elif self.model_type == "classification":
            return self._forward_classification(frames)

        elif self.model_type == "detection":
            writable_frames = np.array(frames, copy=True, dtype=np.float32) / 255.0
            if is_batch:
                img_array = np.transpose(
                    writable_frames, (0, 3, 1, 2)
                )  # (B, H, W, C) -> (B, C, H, W)
            else:
                img_array = np.transpose(writable_frames, (2, 0, 1))
                img_array = np.expand_dims(img_array, 0)
            inputs = {self.input_names[0]: img_array}
            outputs = self.session.run(self.output_names, inputs)
            # Assuming outputs for detection: [boxes, labels, scores]
            # But based on common format, might be [num_detections, detection_boxes, detection_scores, detection_classes]
            if len(outputs) == 4:
                num_dets, boxes, scores, labels = outputs
                results = []
                for i in range(len(num_dets)):
                    n = int(num_dets[i])
                    res = {
                        "boxes": boxes[i, :n],
                        "labels": labels[i, :n].astype(int),
                        "scores": scores[i, :n],
                    }
                    results.append(res)
            elif len(outputs) == 3:
                boxes, labels, scores = outputs
                results = [
                    {"boxes": boxes, "labels": labels.astype(int), "scores": scores}
                ]
                if is_batch:
                    results = [
                        {
                            "boxes": boxes[j],
                            "labels": labels[j].astype(int),
                            "scores": scores[j],
                        }
                        for j in range(boxes.shape[0])
                    ]
            else:
                raise ValueError("Unexpected output format for detection model.")
            self.logger.debug(
                f"Batch inference results: {len(results)} frames processed"
            )
            return results[0] if not is_batch else results

        elif self.model_type == "custom":
            # Generic forward for custom ONNX models
            input_tensor = (
                np.expand_dims(frames.astype(np.float32), axis=0)
                if not is_batch
                else frames.astype(np.float32)
            )
            inputs = {self.input_names[0]: input_tensor}
            outputs = self.session.run(self.output_names, inputs)
            return outputs if len(outputs) > 1 else outputs[0]

        else:
            raise ValueError("Unsupported model type.")

    def do_generate(self, input_text, max_length=1000, system_prompt=None):
        if self.model_type != "llm":
            raise ValueError("Generate is only supported for LLM models.")
        inputs = self.tokenizer(input_text, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=max_length)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.logger.info(f"Generated text: {generated_text}")
        return generated_text
