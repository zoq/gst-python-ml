# SAM
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
    GlobalLogger().warning(f"The 'sam' element will not be available. Error {e}")

# Header prefix for segmentation mask buffer metadata
SAM_META_HEADER = b"GST-SAM:"

# Mask overlay colors (BGR) cycled across detected objects
MASK_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (128, 0, 255),
    (255, 128, 0),
    (0, 128, 255),
    (128, 255, 0),
]


class SamEngine(PyTorchEngine):
    """
    PyTorch engine for Segment Anything Model 2 (SAM2).

    Supports HuggingFace model IDs:
      facebook/sam2-hiera-large
      facebook/sam2-hiera-base-plus
      facebook/sam2-hiera-small
      facebook/sam2-hiera-tiny
    """

    def do_load_model(self, model_name, **kwargs):
        try:
            from transformers import Sam2Model, Sam2Processor

            self.processor = Sam2Processor.from_pretrained(model_name)
            self.model = Sam2Model.from_pretrained(model_name)
            self.execute_with_stream(lambda: self.model.to(self.device))
            self.model.eval()
            self.logger.info(f"SAM2 model '{model_name}' loaded on {self.device}")
        except Exception as e:
            raise ValueError(f"Failed to load SAM2 model '{model_name}': {e}")

    def do_forward(self, frames, max_masks=10):
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

                # Automatic mask generation: grid of input points
                grid_size = int(np.ceil(np.sqrt(max_masks)))
                xs = np.linspace(0, W - 1, grid_size).astype(int)
                ys = np.linspace(0, H - 1, grid_size).astype(int)
                points = [[int(x), int(y)] for y in ys for x in xs][:max_masks]
                input_points = [points]

                inputs = self.processor(
                    images=pil_img,
                    input_points=input_points,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs)

                masks = self.processor.post_process_masks(
                    outputs.pred_masks,
                    inputs["original_sizes"],
                    inputs["reshaped_input_sizes"],
                )
                scores = outputs.iou_scores

                mask_list = []
                if len(masks) > 0:
                    frame_masks = masks[0].cpu().numpy()
                    frame_scores = scores[0].cpu().numpy()
                    for j in range(min(frame_masks.shape[0], max_masks)):
                        best_idx = frame_scores[j].argmax()
                        mask = frame_masks[j, best_idx]
                        score = float(frame_scores[j, best_idx])
                        mask_list.append(
                            {"mask_idx": j, "score": score, "shape": list(mask.shape)}
                        )

                results.append(
                    {
                        "masks": mask_list,
                        "raw_masks": masks[0].cpu().numpy() if len(masks) > 0 else None,
                    }
                )
            except Exception as e:
                self.logger.error(f"SAM inference error on frame: {e}")
                results.append({"masks": [], "raw_masks": None})

        return results[0] if not is_batch else results


class SamTransform(VideoTransform):
    """
    GStreamer element for image segmentation using Segment Anything Model 2.

    Set model-name to a HuggingFace model ID, e.g.:
      facebook/sam2-hiera-large

    When visualize=True (default), colored mask overlays are drawn on the
    video frame. Mask metadata is always appended to the buffer as a
    GST-SAM: memory chunk (JSON with mask scores and shapes).
    """

    __gstmetadata__ = (
        "SAM Segmentation",
        "Transform",
        "Image segmentation using Segment Anything Model 2",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    visualize = GObject.Property(
        type=bool,
        default=True,
        nick="Visualize Masks",
        blurb="Overlay colored segmentation masks on the video frame",
        flags=GObject.ParamFlags.READWRITE,
    )

    max_masks = GObject.Property(
        type=int,
        default=10,
        minimum=1,
        maximum=100,
        nick="Max Masks",
        blurb="Maximum number of segmentation masks to generate",
        flags=GObject.ParamFlags.READWRITE,
    )

    mode = GObject.Property(
        type=str,
        default="auto",
        nick="Segmentation Mode",
        blurb="Segmentation mode: 'auto' for automatic, 'points' for point-prompted",
        flags=GObject.ParamFlags.READWRITE,
    )

    def __init__(self):
        super().__init__()
        self.mgr.engine_name = "pyml_sam_engine"
        EngineFactory.register(self.mgr.engine_name, SamEngine)
        self.format_converter = FormatConverter()

    @GObject.Property(type=str)
    def engine_name(self):
        """Machine Learning Engine (read-only for this element)."""
        return self.mgr.engine_name

    @engine_name.setter
    def engine_name(self, value):
        raise ValueError("'engine_name' is read-only for pyml_sam")

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
                self._apply_masks(buf, result, fmt, frames)
            else:
                if isinstance(result, list) and len(result) > 0:
                    self._apply_masks(
                        buf, result[0], fmt, frames[0] if frames.ndim == 4 else frames
                    )

            return Gst.FlowReturn.OK

        except Exception as e:
            self.logger.error(f"SAM transform error: {e}")
            return Gst.FlowReturn.ERROR

    def _do_forward(self, frames):
        if self.engine:
            return self.engine.do_forward(frames, max_masks=self.max_masks)
        return None

    def _apply_masks(self, buf, result, fmt, frame):
        """Overlay masks on frame and append mask metadata."""
        import cv2
        import numpy as np

        raw_masks = result.get("raw_masks")
        mask_info = result.get("masks", [])

        # Draw mask overlays before appending read-only metadata memory
        if self.visualize and raw_masks is not None:
            overlay = frame.copy()
            for j in range(min(raw_masks.shape[0], self.max_masks)):
                best_idx = 0
                if raw_masks.ndim == 4 and raw_masks.shape[1] > 1:
                    best_idx = raw_masks[j].sum(axis=(1, 2)).argmax()
                mask = raw_masks[j, best_idx]
                color = MASK_COLORS[j % len(MASK_COLORS)]
                colored = np.zeros_like(overlay)
                colored[:] = color
                mask_bool = mask > 0.5
                overlay[mask_bool] = cv2.addWeighted(
                    overlay[mask_bool], 0.5, colored[mask_bool], 0.5, 0
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

        # Append mask metadata as a custom buffer memory chunk
        if mask_info:
            meta_bytes = SAM_META_HEADER + json.dumps(mask_info).encode("utf-8")
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
    GObject.type_register(SamTransform)
    __gstelementfactory__ = ("pyml_sam", Gst.Rank.NONE, SamTransform)
else:
    GlobalLogger().warning(
        "The 'pyml_sam' element will not be registered because required modules are missing."
    )
