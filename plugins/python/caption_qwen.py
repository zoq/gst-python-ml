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
    gi.require_version("GstBase", "1.0")
    gi.require_version("GstVideo", "1.0")
    gi.require_version("GLib", "2.0")

    from gi.repository import Gst, GObject, GstAnalytics, GLib, GstBase  # noqa: E402
    import numpy as np
    import cv2
    import traceback

    import torch
    from transformers import (
        Qwen2_5_VLForConditionalGeneration,
        AutoProcessor,
    )
    from qwen_vl_utils import process_vision_info
    from engine.pytorch_engine import PyTorchEngine
    from engine.engine_factory import EngineFactory
    from base_caption import BaseCaption

    class CaptionQwenEngine(PyTorchEngine):
        def load_model(self, model_name, **kwargs):
            """Load a Qwen2.5-VL model from Hugging Face."""
            try:
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype="auto",
                    dtype=torch.float16,
                    device_map="auto",
                )
                self.processor = AutoProcessor.from_pretrained(model_name)

                self.logger.info(
                    f"{model_name} model and processor loaded successfully."
                )
                self.model.eval()

                # Skip .to() for quantized models
                if not (
                    hasattr(self.model, "is_quantized") and self.model.is_quantized
                ):
                    self.execute_with_stream(lambda: self.model.to(self.device))
                    self.logger.info(f"Model moved to {self.device}")

                return True

            except Exception as e:
                self.logger.error(f"Error loading model '{model_name}': {e}")
                self.processor = None
                self.model = None
                return False

        def forward(self, frames):
            """Handle inference for Qwen2.5-VL, supporting single frames or batches."""
            is_batch = (
                isinstance(frames, np.ndarray) and frames.ndim == 4
            )  # (B, H, W, C)
            if not isinstance(frames, (np.ndarray, str)):
                self.logger.error(f"Invalid input type for forward: {type(frames)}")
                return None

            if self.processor:
                try:
                    from PIL import Image
                    import torch
                    import gc

                    # Convert frames to PIL images
                    if is_batch:
                        pil_images = [
                            Image.fromarray(np.uint8(frame)) for frame in frames
                        ]
                    else:
                        pil_images = [Image.fromarray(np.uint8(frames))]

                    # Create content list with images and prompt
                    content = [{"type": "image", "image": img} for img in pil_images]
                    content.append({"type": "text", "text": self.prompt})
                    messages = [{"role": "user", "content": content}]

                    # Apply chat template
                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )

                    # Process vision inputs
                    image_inputs, video_inputs = process_vision_info(messages)

                    # Process inputs for batch inference
                    inputs = self.processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    ).to(self.device)

                    # Run inference
                    generation_args = {
                        "max_new_tokens": 500,
                        "temperature": 0.0,
                        "do_sample": False,
                    }
                    with torch.inference_mode():
                        generated_ids = self.model.generate(
                            **inputs,
                            eos_token_id=self.processor.tokenizer.eos_token_id,
                            **generation_args,
                        )

                    # Trim to only generated tokens
                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :]
                        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]

                    # Decode response
                    output_text = self.processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    if not output_text:
                        self.logger.error("No output text generated.")
                        return None
                    response = output_text[0]

                    # Split response into per-frame captions (adjust based on model output)
                    if is_batch:
                        # Assume response contains captions separated by newlines or repeated
                        captions = (
                            response.split("\n")[: len(pil_images)]
                            if "\n" in response
                            else [response] * len(pil_images)
                        )
                    else:
                        captions = [response]

                    self.logger.info(f"Generated captions: {captions}")

                    # Clean up
                    del inputs, generated_ids
                    torch.cuda.empty_cache()
                    gc.collect()

                    return captions if is_batch else captions[0]

                except Exception as e:
                    self.logger.error(f"Vision-language inference error: {e}")
                    return None

        # no-op
        def generate(self, input_text, max_length=100):
            """Generate LLM text."""
            pass

    class CaptionQwen(BaseCaption):
        """
        GStreamer element for captioning video frames.
        """

        __gstmetadata__ = (
            "CaptionQwen",
            "Transform",
            "Captions video clips using Qwen Vision model",
            "Aaron Boxer <aaron.boxer@collabora.com>",
        )

        def __init__(self):
            super().__init__()
            # self.model_name = "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"
            # set engine_name directly on engine_helper, as engine_name property is read only
            self.engine_helper.engine_name = "pyml_caption_qwen_engine"
            EngineFactory.register(self.engine_name, CaptionQwenEngine)

except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(
        f"The 'pyml_caption_qwen' element will not be available. Error {e}"
    )


if CAN_REGISTER_ELEMENT:
    GObject.type_register(CaptionQwen, "pyml_caption_qwen")
    __gstelementfactory__ = ("pyml_caption_qwen", Gst.Rank.NONE, CaptionQwen)
else:
    GlobalLogger().warning(
        "The 'pyml_caption_qwen' element will not be registered because required modules are missing."
    )
