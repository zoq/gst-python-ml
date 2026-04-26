# VLM (Vision-Language Model)
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

from log.global_logger import GlobalLogger

CAN_REGISTER_ELEMENT = True
try:
    import json

    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstBase", "1.0")
    gi.require_version("GstVideo", "1.0")
    from gi.repository import Gst, GObject

    from video_transform import VideoTransform
    from utils.format_converter import FormatConverter
    from engine.pytorch_engine import PyTorchEngine
    from engine.engine_factory import EngineFactory

except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(f"The 'vlm' element will not be available. Error {e}")

# Header prefix for VLM response buffer metadata
VLM_META_HEADER = b"GST-VLM:"


class VlmEngine(PyTorchEngine):
    """
    PyTorch engine for Vision-Language Models.

    Supports HuggingFace VLM model IDs via AutoProcessor + AutoModelForVision2Seq:
      llava-hf/llava-1.5-7b-hf
      Qwen/Qwen2-VL-7B-Instruct
      OpenGVLab/InternVL2-8B
    """

    def __init__(self):
        super().__init__()
        self.processor = None

    def do_load_model(self, model_name, **kwargs):
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForVision2Seq

            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=self.device,
            )
            self.model.eval()
            self.logger.info(f"VLM model '{model_name}' loaded on {self.device}")
        except Exception as e:
            raise ValueError(f"Failed to load VLM model '{model_name}': {e}")

    def do_forward(
        self,
        frame,
        prompt="Describe this image in detail.",
        system_prompt=None,
        max_tokens=256,
        temperature=0.7,
    ):
        """
        Run VLM inference on a single video frame.

        Args:
            frame: numpy RGB array (H, W, 3).
            prompt: user prompt text.
            system_prompt: optional system prompt.
            max_tokens: maximum tokens to generate.
            temperature: sampling temperature.

        Returns:
            Generated text string, or None on failure.
        """
        import numpy as np
        from PIL import Image

        try:
            pil_img = Image.fromarray(frame.astype(np.uint8))
            text = self.do_generate(
                pil_img, prompt, system_prompt, max_tokens, temperature
            )
            return text
        except Exception as e:
            self.logger.error(f"VLM inference error: {e}")
            return None

    def do_generate(self, image, prompt, system_prompt, max_tokens, temperature):
        """
        Apply chat template, process image + text, and generate a response.

        Args:
            image: PIL Image.
            prompt: user prompt text.
            system_prompt: optional system prompt.
            max_tokens: maximum new tokens.
            temperature: sampling temperature.

        Returns:
            Generated text string.
        """
        import torch

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        )

        # Apply chat template if the processor supports it
        if hasattr(self.processor, "apply_chat_template"):
            text_input = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
        else:
            text_input = prompt

        inputs = self.processor(text=text_input, images=image, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )

        # Decode only the newly generated tokens
        input_len = inputs.get("input_ids", torch.tensor([])).shape[-1]
        generated = output_ids[0][input_len:]
        text = self.processor.decode(generated, skip_special_tokens=True)
        return text.strip()


class VlmTransform(VideoTransform):
    """
    GStreamer element for Vision-Language Model inference on video frames.

    For each processed frame, runs a VLM and appends the generated text as a
    GST-VLM: JSON memory chunk on the buffer.

    Downstream elements can read the generated text:
      for i in range(buf.n_memory()):
          data = bytes(buf.peek_memory(i).map(Gst.MapFlags.READ).data)
          if data.startswith(b"GST-VLM:"):
              result = json.loads(data[8:])
              text = result["text"]

    Use frame-stride to control processing frequency:
      pyml_vlm model-name=llava-hf/llava-1.5-7b-hf frame-stride=30 \\
        prompt="What activity is happening in this scene?"
    """

    __gstmetadata__ = (
        "Vision-Language Model",
        "Transform",
        "Vision-Language Model inference for video frame understanding",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    prompt = GObject.Property(
        type=str,
        default="Describe this image in detail.",
        nick="Prompt",
        blurb="User prompt for the VLM",
        flags=GObject.ParamFlags.READWRITE,
    )

    max_tokens = GObject.Property(
        type=int,
        default=256,
        minimum=1,
        maximum=4096,
        nick="Max Tokens",
        blurb="Maximum number of tokens to generate",
        flags=GObject.ParamFlags.READWRITE,
    )

    system_prompt = GObject.Property(
        type=str,
        default=None,
        nick="System Prompt",
        blurb="Optional system prompt for the VLM",
        flags=GObject.ParamFlags.READWRITE,
    )

    temperature = GObject.Property(
        type=float,
        default=0.7,
        minimum=0.0,
        maximum=2.0,
        nick="Temperature",
        blurb="Sampling temperature for generation (0 = greedy)",
        flags=GObject.ParamFlags.READWRITE,
    )

    def __init__(self):
        super().__init__()
        self.mgr.engine_name = "pyml_vlm_engine"
        EngineFactory.register(self.mgr.engine_name, VlmEngine)
        self.format_converter = FormatConverter()
        self._frame_count = 0

    @GObject.Property(type=str)
    def engine_name(self):
        """Machine Learning Engine (read-only for this element)."""
        return self.mgr.engine_name

    @engine_name.setter
    def engine_name(self, value):
        raise ValueError("'engine_name' is read-only for pyml_vlm")

    def do_transform_ip(self, buf):
        try:
            self._frame_count += 1
            if self.frame_stride > 1 and (self._frame_count % self.frame_stride) != 1:
                return Gst.FlowReturn.OK

            if self.engine is None:
                return Gst.FlowReturn.OK

            success, map_info = buf.map(Gst.MapFlags.READ)
            if not success:
                self.logger.error("Failed to map video buffer for reading")
                return Gst.FlowReturn.ERROR

            try:
                frame = self.format_converter.to_rgb(
                    map_info.data, self.width, self.height, buf, self.sinkpad
                )
            finally:
                buf.unmap(map_info)

            if frame is None:
                return Gst.FlowReturn.ERROR

            text = self.engine.do_forward(
                frame,
                prompt=self.prompt,
                system_prompt=self.system_prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            if text is None:
                return Gst.FlowReturn.OK

            result = {"text": text}
            # Append VLM response as a JSON memory chunk.
            # Use new_allocate+fill: PyGI hides the maxsize arg in new_wrapped
            # (it derives it from data length), so passing it explicitly shifts
            # all subsequent args and causes a GI assertion crash.
            meta_bytes = VLM_META_HEADER + json.dumps(result).encode("utf-8")
            tmp = Gst.Buffer.new_allocate(None, len(meta_bytes), None)
            tmp.fill(0, meta_bytes)
            buf.append_memory(tmp.get_memory(0))

            self.logger.debug(
                f"VLM response ({len(text)} chars): {text[:80]}..."
                if len(text) > 80
                else f"VLM response: {text}"
            )

            return Gst.FlowReturn.OK

        except Exception as e:
            self.logger.error(f"VLM transform error: {e}")
            return Gst.FlowReturn.ERROR


if CAN_REGISTER_ELEMENT:
    GObject.type_register(VlmTransform)
    __gstelementfactory__ = ("pyml_vlm", Gst.Rank.NONE, VlmTransform)
else:
    GlobalLogger().warning(
        "The 'pyml_vlm' element will not be registered because required modules are missing."
    )
