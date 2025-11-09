# CaptionPhi
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

    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoProcessor,
        BitsAndBytesConfig,
    )

    from engine.pytorch_vision_engine import PyTorchVisionEngine
    from engine.engine_factory import EngineFactory
    from base_caption import BaseCaption

except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(
        f"The 'pyml_caption_phi' element will not be available. Error {e}"
    )


class CaptionPhiEngine(PyTorchVisionEngine):
    def do_load_model(self, model_name, **kwargs):
        """Load a Phi-3-vision model from Hugging Face."""
        try:
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                _attn_implementation="flash_attention_2",
            )
            self.processor = AutoProcessor.from_pretrained(
                model_name, trust_remote_code=True
            )
            self.logger.info("Phi-3.5-vision model and processor loaded successfully.")
            self.model.eval()

            # Skip .to() for 4-bit models
            if not (
                hasattr(self.model, "is_loaded_in_4bit")
                and self.model.is_loaded_in_4bit
            ):
                self.execute_with_stream(lambda: self.model.to(self.device))
                self.logger.info(f"Model moved to {self.device}")

            return True

        except Exception as e:
            self.logger.error(f"Error loading model '{model_name}': {e}")
            self.tokenizer = None
            self.model = None
            return False

    def _prepare_messages(self, images):
        prompt_content = (
            "\n".join([f"<|image_{i+1}|>" for i in range(len(images))])
            + f"\n{self.prompt}"
        )
        return [{"role": "user", "content": prompt_content}]

    def _process_inputs(self, prompt_text, images):
        return self.processor(prompt_text, images, return_tensors="pt").to(self.device)

    def _trim_generated_ids(self, inputs, generate_ids):
        return generate_ids[:, inputs["input_ids"].shape[1] :]


class CaptionPhi(BaseCaption):
    """
    GStreamer element for captioning video frames using Phi Vision.
    """

    __gstmetadata__ = (
        "CaptionPhi",
        "Transform",
        "Captions video clips using Phi Vision model",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    def __init__(self):
        super().__init__()
        # set engine_name directly on mgr, as engine_name property is read only
        self.mgr.engine_name = "pyml_caption_phi_engine"
        EngineFactory.register(self.engine_name, CaptionPhiEngine)


if CAN_REGISTER_ELEMENT:
    GObject.type_register(CaptionPhi, "pyml_caption_phi")
    __gstelementfactory__ = ("pyml_caption_phi", Gst.Rank.NONE, CaptionPhi)
else:
    GlobalLogger().warning(
        "The 'pyml_caption_phi' element will not be registered because required modules are missing."
    )
