# CaptionQwen
# Copyright (C) 2024-2025 Collabora Ltd.
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Library General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
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

from log.global_logger import GlobalLogger

CAN_REGISTER_ELEMENT = True
try:
    import gi

    gi.require_version("Gst", "1.0")

    from gi.repository import Gst, GObject  # noqa: E402
    import traceback

    import torch
    from transformers import (
        Qwen2_5_VLForConditionalGeneration,
        AutoProcessor,
    )
    from qwen_vl_utils import process_vision_info
    from engine.pytorch_vision_engine import PyTorchVisionEngine
    from engine.engine_factory import EngineFactory
    from base_caption import BaseCaption

except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(
        f"The 'pyml_caption_qwen' element will not be available. Error {e}"
    )


class CaptionQwenEngine(PyTorchVisionEngine):
    def do_load_model(self, model_name, **kwargs):
        """Load a Qwen2.5-VL model from Hugging Face."""
        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype="auto",
                dtype=torch.float16,
                device_map="auto",
            )
            self.processor = AutoProcessor.from_pretrained(model_name)

            self.logger.info(f"{model_name} model and processor loaded successfully.")
            self.model.eval()

            # Skip .to() for quantized models
            if not (hasattr(self.model, "is_quantized") and self.model.is_quantized):
                self.execute_with_stream(lambda: self.model.to(self.device))
                self.logger.info(f"Model moved to {self.device}")

            return True

        except Exception as e:
            self.logger.error(f"Error loading model '{model_name}': {e}")
            self.processor = None
            self.model = None
            return False

    def _prepare_messages(self, images):
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": self.prompt})
        return [{"role": "user", "content": content}]

    def _process_inputs(self, prompt_text, images):
        image_inputs, video_inputs = process_vision_info(
            self._prepare_messages(images)
        )  # Note: Uses messages directly
        return self.processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

    def _trim_generated_ids(self, inputs, generate_ids):
        return [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generate_ids)
        ]


class CaptionQwen(BaseCaption):
    """
    GStreamer element for captioning video frames using Qwen Vision.
    """

    __gstmetadata__ = (
        "CaptionQwen",
        "Transform",
        "Captions video clips using Qwen Vision model",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    def __init__(self):
        super().__init__()
        # set engine_name directly on mgr, as engine_name property is read only
        self.mgr.engine_name = "pyml_caption_qwen_engine"
        EngineFactory.register(self.engine_name, CaptionQwenEngine)


if CAN_REGISTER_ELEMENT:
    GObject.type_register(CaptionQwen, "pyml_caption_qwen")
    __gstelementfactory__ = ("pyml_caption_qwen", Gst.Rank.NONE, CaptionQwen)
else:
    GlobalLogger().warning(
        "The 'pyml_caption_qwen' element will not be registered because required modules are missing."
    )
