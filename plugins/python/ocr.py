# OCR
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
    import ctypes
    import json

    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstBase", "1.0")
    gi.require_version("GstVideo", "1.0")
    from gi.repository import Gst, GObject

    from video_transform import VideoTransform
    from utils.format_converter import FormatConverter
    from utils.muxed_buffer_processor import MuxedBufferProcessor
    from engine.pytorch_engine import PyTorchEngine
    from engine.engine_factory import EngineFactory

except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(f"The 'ocr' element will not be available. Error {e}")

# Header prefix for OCR text buffer metadata
OCR_META_HEADER = b"GST-OCR:"


class OcrEngine(PyTorchEngine):
    """
    PyTorch engine for TrOCR text recognition.

    Supports HuggingFace model IDs:
      microsoft/trocr-base-printed
      microsoft/trocr-large-printed
      microsoft/trocr-base-handwritten
    """

    def do_load_model(self, model_name, **kwargs):
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

            self.processor = TrOCRProcessor.from_pretrained(model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
            self.execute_with_stream(lambda: self.model.to(self.device))
            self.model.eval()
            self.logger.info(f"TrOCR model '{model_name}' loaded on {self.device}")
        except Exception as e:
            raise ValueError(f"Failed to load TrOCR model '{model_name}': {e}")

    def do_forward(self, frames):
        import numpy as np
        import torch
        from PIL import Image

        is_batch = isinstance(frames, np.ndarray) and frames.ndim == 4
        if not is_batch:
            frames = frames[np.newaxis]

        results = []
        for frame in frames:
            try:
                pil_img = Image.fromarray(frame.astype(np.uint8))
                H, W = frame.shape[:2]

                # Split frame into horizontal strips for text region detection
                strip_height = max(H // 4, 32)
                texts = []
                regions = []
                for y_start in range(0, H, strip_height):
                    y_end = min(y_start + strip_height, H)
                    strip = pil_img.crop((0, y_start, W, y_end))
                    pixel_values = self.processor(
                        images=strip, return_tensors="pt"
                    ).pixel_values.to(self.device)

                    with torch.no_grad():
                        generated_ids = self.model.generate(pixel_values)

                    text = self.processor.batch_decode(
                        generated_ids, skip_special_tokens=True
                    )[0].strip()
                    if text:
                        texts.append(text)
                        regions.append(
                            {
                                "x": 0,
                                "y": y_start,
                                "w": W,
                                "h": y_end - y_start,
                                "text": text,
                            }
                        )

                results.append({"texts": texts, "regions": regions})
            except Exception as e:
                self.logger.error(f"OCR inference error on frame: {e}")
                results.append({"texts": [], "regions": []})

        return results[0] if not is_batch else results


class OCRTransform(VideoTransform):
    """
    GStreamer element for optical character recognition on video frames.

    Set model-name to a HuggingFace model ID, e.g.:
      microsoft/trocr-base-printed

    When draw-text=True (default), recognized text is drawn directly on the
    video frame. OCR results are always appended as a GST-OCR: memory chunk
    (JSON with recognized text and regions).
    """

    __gstmetadata__ = (
        "OCR",
        "Transform",
        "Optical character recognition using TrOCR on video frames",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    draw_text = GObject.Property(
        type=bool,
        default=True,
        nick="Draw Text",
        blurb="Draw recognized text on the video frame",
        flags=GObject.ParamFlags.READWRITE,
    )

    language = GObject.Property(
        type=str,
        default="en",
        nick="Language",
        blurb="Language hint for OCR (currently informational)",
        flags=GObject.ParamFlags.READWRITE,
    )

    def __init__(self):
        super().__init__()
        self.mgr.engine_name = "pyml_ocr_engine"
        EngineFactory.register(self.mgr.engine_name, OcrEngine)
        self.format_converter = FormatConverter()

    @GObject.Property(type=str)
    def engine_name(self):
        """Machine Learning Engine (read-only for this element)."""
        return self.mgr.engine_name

    @engine_name.setter
    def engine_name(self, value):
        raise ValueError("'engine_name' is read-only for pyml_ocr")

    def do_transform_ip(self, buf):
        try:
            processor = MuxedBufferProcessor(
                self.logger, self.width, self.height, 30, 1
            )
            frames, _, num_sources, fmt = processor.extract_frames(buf, self.sinkpad)
            if frames is None:
                return Gst.FlowReturn.ERROR

            result = self._do_forward(frames)
            if result is None:
                return Gst.FlowReturn.ERROR

            if num_sources == 1:
                self._apply_ocr(buf, result, fmt, frames)
            else:
                if isinstance(result, list) and len(result) > 0:
                    self._apply_ocr(
                        buf, result[0], fmt, frames[0] if frames.ndim == 4 else frames
                    )

            return Gst.FlowReturn.OK

        except Exception as e:
            self.logger.error(f"OCR transform error: {e}")
            return Gst.FlowReturn.ERROR

    def _do_forward(self, frames):
        if self.engine:
            return self.engine.do_forward(frames)
        return None

    def _apply_ocr(self, buf, result, fmt, frame):
        """Draw recognized text on frame and append OCR metadata."""
        import cv2
        import numpy as np

        regions = result.get("regions", [])

        # Draw text overlays before appending read-only metadata memory
        if self.draw_text and regions:
            overlay = frame.copy()
            for region in regions:
                x, y, w, h = region["x"], region["y"], region["w"], region["h"]
                text = region["text"]
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
                font_scale = max(0.4, min(w / 300.0, 1.0))
                cv2.putText(
                    overlay,
                    text,
                    (x + 4, y + h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

            output = self._convert_rgb_to_format(overlay, fmt)
            if output is not None:
                success, map_info = buf.map(Gst.MapFlags.WRITE)
                if success:
                    try:
                        frame_bytes = np.ascontiguousarray(output).tobytes()
                        dst = (ctypes.c_char * map_info.size).from_buffer(map_info.data)
                        ctypes.memmove(
                            dst, frame_bytes, min(len(frame_bytes), map_info.size)
                        )
                    finally:
                        buf.unmap(map_info)

        # Append OCR results as a custom buffer memory chunk
        if regions:
            meta_bytes = OCR_META_HEADER + json.dumps(regions).encode("utf-8")
            tmp = Gst.Buffer.new_allocate(None, len(meta_bytes), None)
            tmp.fill(0, meta_bytes)
            buf.append_memory(tmp.get_memory(0))

    @staticmethod
    def _convert_rgb_to_format(rgb, fmt):
        """Convert an RGB numpy array to the target GStreamer video format."""
        import cv2
        import numpy as np

        if fmt == "RGB":
            return rgb
        elif fmt == "BGR":
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        elif fmt == "RGBA":
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2RGBA)
        elif fmt == "BGRA":
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGRA)
        elif fmt == "ARGB":
            rgba = cv2.cvtColor(rgb, cv2.COLOR_RGB2RGBA)
            return np.roll(rgba, 1, axis=-1)
        elif fmt == "ABGR":
            bgra = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGRA)
            return np.roll(bgra, 1, axis=-1)
        else:
            return rgb


if CAN_REGISTER_ELEMENT:
    GObject.type_register(OCRTransform)
    __gstelementfactory__ = ("pyml_ocr", Gst.Rank.NONE, OCRTransform)
else:
    GlobalLogger().warning(
        "The 'pyml_ocr' element will not be registered because required modules are missing."
    )
