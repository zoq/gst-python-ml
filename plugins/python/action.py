# Action Recognition
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
    from collections import deque

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
    GlobalLogger().warning(f"The 'action' element will not be available. Error {e}")

# Header prefix for action classification buffer metadata
ACTION_META_HEADER = b"GST-ACTION:"


class ActionEngine(PyTorchEngine):
    """
    PyTorch engine for video action recognition using VideoMAE.

    Supports HuggingFace model IDs:
      MCG-NJU/videomae-base-finetuned-kinetics
      MCG-NJU/videomae-large-finetuned-kinetics
      facebook/timesformer-base-finetuned-k400
    """

    def do_load_model(self, model_name, **kwargs):
        try:
            from transformers import AutoImageProcessor, VideoMAEForVideoClassification

            self.image_processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = VideoMAEForVideoClassification.from_pretrained(model_name)
            self.execute_with_stream(lambda: self.model.to(self.device))
            self.model.eval()
            self.logger.info(f"VideoMAE model '{model_name}' loaded on {self.device}")
        except Exception as e:
            raise ValueError(f"Failed to load VideoMAE model '{model_name}': {e}")

    def do_forward(self, frame_buffer):
        """
        Classify a buffer of frames.

        Args:
            frame_buffer: list of numpy arrays (H, W, 3), length = num_frames

        Returns:
            dict with 'label', 'score', and 'top5' predictions
        """
        import numpy as np
        import torch
        from PIL import Image

        try:
            pil_frames = [Image.fromarray(f.astype(np.uint8)) for f in frame_buffer]

            inputs = self.image_processor(pil_frames, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=-1)
            top5_indices = probs.topk(5).indices.cpu().numpy()
            top5_scores = probs.topk(5).values.cpu().numpy()

            top1_idx = top5_indices[0]
            label = self.model.config.id2label.get(int(top1_idx), f"class_{top1_idx}")
            score = float(top5_scores[0])

            top5 = []
            for idx, s in zip(top5_indices, top5_scores):
                name = self.model.config.id2label.get(int(idx), f"class_{idx}")
                top5.append({"label": name, "score": float(s)})

            return {"label": label, "score": score, "top5": top5}

        except Exception as e:
            self.logger.error(f"Action recognition inference error: {e}")
            return None


class ActionTransform(VideoTransform):
    """
    GStreamer element for video action/activity recognition using VideoMAE.

    Buffers a window of frames internally and runs classification when the
    buffer is full. The predicted action label is attached as a GST-ACTION:
    memory chunk (JSON) and optionally drawn on the frame.

    Set model-name to a HuggingFace model ID, e.g.:
      MCG-NJU/videomae-base-finetuned-kinetics
    """

    __gstmetadata__ = (
        "Action Recognition",
        "Transform",
        "Video action/activity recognition using VideoMAE",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    num_frames = GObject.Property(
        type=int,
        default=16,
        minimum=2,
        maximum=128,
        nick="Num Frames",
        blurb="Number of frames to buffer before running classification",
        flags=GObject.ParamFlags.READWRITE,
    )

    draw_label = GObject.Property(
        type=bool,
        default=True,
        nick="Draw Label",
        blurb="Draw predicted action label on the video frame",
        flags=GObject.ParamFlags.READWRITE,
    )

    def __init__(self):
        super().__init__()
        self.mgr.engine_name = "pyml_action_engine"
        EngineFactory.register(self.mgr.engine_name, ActionEngine)
        self.format_converter = FormatConverter()
        self._frame_buffer = deque(maxlen=16)
        self._last_result = None

    @GObject.Property(type=str)
    def engine_name(self):
        """Machine Learning Engine (read-only for this element)."""
        return self.mgr.engine_name

    @engine_name.setter
    def engine_name(self, value):
        raise ValueError("'engine_name' is read-only for pyml_action")

    def do_transform_ip(self, buf):
        try:
            processor = MuxedBufferProcessor(
                self.logger, self.width, self.height, 30, 1
            )
            frames, _, num_sources, fmt = processor.extract_frames(buf, self.sinkpad)
            if frames is None:
                return Gst.FlowReturn.ERROR

            frame = frames[0] if frames.ndim == 4 else frames

            # Update deque maxlen if property changed
            if self._frame_buffer.maxlen != self.num_frames:
                self._frame_buffer = deque(self._frame_buffer, maxlen=self.num_frames)

            self._frame_buffer.append(frame.copy())

            # Run classification when buffer is full
            if len(self._frame_buffer) == self.num_frames:
                result = self._do_forward(list(self._frame_buffer))
                if result is not None:
                    self._last_result = result

            # Draw label and attach metadata using the latest result
            if self._last_result is not None:
                self._apply_action(buf, self._last_result, fmt, frame)

            return Gst.FlowReturn.OK

        except Exception as e:
            self.logger.error(f"Action recognition transform error: {e}")
            return Gst.FlowReturn.ERROR

    def _do_forward(self, frame_buffer):
        if self.engine:
            return self.engine.do_forward(frame_buffer)
        return None

    def _apply_action(self, buf, result, fmt, frame):
        """Draw action label on frame and append action metadata."""
        import cv2
        import numpy as np

        label = result.get("label", "")
        score = result.get("score", 0.0)

        # Draw label before appending read-only metadata memory
        if self.draw_label and label:
            overlay = frame.copy()
            text = f"{label} ({score:.2f})"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.rectangle(overlay, (8, 8), (16 + tw, 16 + th + 8), (0, 0, 0), -1)
            cv2.putText(
                overlay,
                text,
                (12, 12 + th),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
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

        # Append action metadata as a custom buffer memory chunk
        meta_bytes = ACTION_META_HEADER + json.dumps(result).encode("utf-8")
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
    GObject.type_register(ActionTransform)
    __gstelementfactory__ = ("pyml_action", Gst.Rank.NONE, ActionTransform)
else:
    GlobalLogger().warning(
        "The 'pyml_action' element will not be registered because required modules are missing."
    )
