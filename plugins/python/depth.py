# Depth
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

    import gi
    import numpy as np

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
    GlobalLogger().warning(f"The 'depth' element will not be available. Error {e}")

# Header prefix for depth map buffer metadata
DEPTH_META_HEADER = b"GST-DEPTH:"

# cv2 colormap IDs for depth visualization
COLORMAP_IDS = {
    "inferno": 9,
    "jet": 2,
    "viridis": 16,
    "plasma": 18,
    "magma": 13,
}


class DepthAnythingEngine(PyTorchEngine):
    """
    PyTorch engine for DepthAnything V2 monocular depth estimation.

    Supports HuggingFace model IDs:
      depth-anything/Depth-Anything-V2-Small-hf  (fastest)
      depth-anything/Depth-Anything-V2-Base-hf
      depth-anything/Depth-Anything-V2-Large-hf  (most accurate)
    """

    def do_load_model(self, model_name, **kwargs):
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            self.image_processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForDepthEstimation.from_pretrained(model_name)
            self.execute_with_stream(lambda: self.model.to(self.device))
            self.model.eval()
            self.logger.info(
                f"DepthAnything model '{model_name}' loaded on {self.device}"
            )
        except Exception as e:
            raise ValueError(f"Failed to load depth model '{model_name}': {e}")

    def do_forward(self, frames):
        import torch
        import torch.nn.functional as F
        from PIL import Image

        is_batch = isinstance(frames, np.ndarray) and frames.ndim == 4
        if not is_batch:
            frames = frames[np.newaxis]

        results = []
        for frame in frames:
            try:
                pil_img = Image.fromarray(frame.astype(np.uint8))
                H, W = frame.shape[:2]
                inputs = self.image_processor(images=pil_img, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model(**inputs)
                # outputs.predicted_depth: [1, H', W']
                depth_up = F.interpolate(
                    outputs.predicted_depth.unsqueeze(0),
                    size=(H, W),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
                results.append(depth_up.cpu().numpy())
            except Exception as e:
                self.logger.error(f"Depth inference error on frame: {e}")
                results.append(None)

        return results[0] if not is_batch else results


class DepthTransform(VideoTransform):
    """
    GStreamer element for monocular depth estimation using DepthAnything V2.

    Set model-name to a HuggingFace model ID, e.g.:
      depth-anything/Depth-Anything-V2-Small-hf

    When visualize=True (default), the video frame is replaced with a
    colorized depth map. Use a tee element upstream to preserve the original
    video alongside the depth visualization.

    A uint8 normalized depth map is always appended to the buffer as a
    GST-DEPTH: memory chunk for downstream elements:
      for i in range(buf.n_memory()):
          data = bytes(buf.peek_memory(i).map(Gst.MapFlags.READ).data)
          if data.startswith(b"GST-DEPTH:"):
              depth = np.frombuffer(data[10:], dtype=np.uint8).reshape(H, W)

    Use frame-stride to skip frames and reduce compute:
      pyml_depth model-name=depth-anything/Depth-Anything-V2-Small-hf frame-stride=2
    """

    __gstmetadata__ = (
        "Depth",
        "Transform",
        "Monocular depth estimation using DepthAnything V2",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    visualize = GObject.Property(
        type=bool,
        default=True,
        nick="Visualize Depth",
        blurb="Replace video frame with a colorized depth map",
        flags=GObject.ParamFlags.READWRITE,
    )

    colormap = GObject.Property(
        type=str,
        default="inferno",
        nick="Colormap",
        blurb="Colormap for depth visualization: inferno, jet, viridis, plasma, magma",
        flags=GObject.ParamFlags.READWRITE,
    )

    def __init__(self):
        super().__init__()
        self.mgr.engine_name = "pyml_depth_engine"
        EngineFactory.register(self.mgr.engine_name, DepthAnythingEngine)
        self.format_converter = FormatConverter()

    @GObject.Property(type=str)
    def engine_name(self):
        """Machine Learning Engine (read-only for this element)."""
        return self.mgr.engine_name

    @engine_name.setter
    def engine_name(self, value):
        raise ValueError("'engine_name' is read-only for pyml_depth")

    def do_transform_ip(self, buf):
        try:
            processor = MuxedBufferProcessor(
                self.logger, self.width, self.height, 30, 1
            )
            frames, _, num_sources, fmt = processor.extract_frames(buf, self.sinkpad)
            if frames is None:
                return Gst.FlowReturn.ERROR

            if num_sources == 1:
                depth = self._do_forward(frames)
                if depth is None:
                    return Gst.FlowReturn.ERROR
                self._apply_depth(buf, depth, fmt)
            else:
                depths = self._do_forward(frames)
                if depths:
                    # For batch: apply only the first depth map (primary frame)
                    self._apply_depth(buf, depths[0], fmt)

            return Gst.FlowReturn.OK

        except Exception as e:
            self.logger.error(f"Depth transform error: {e}")
            return Gst.FlowReturn.ERROR

    def _do_forward(self, frames):
        if self.engine:
            return self.engine.do_forward(frames)
        return None

    def _apply_depth(self, buf, depth_map, fmt):
        """Normalize depth, optionally visualize, then append as metadata."""
        import cv2

        d_min, d_max = depth_map.min(), depth_map.max()
        if d_max > d_min:
            depth_norm = ((depth_map - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            depth_norm = np.zeros_like(depth_map, dtype=np.uint8)

        # Visualize first, before appending any read-only metadata memory.
        # (A READONLY chunk on the buffer would prevent buf.map(WRITE) from succeeding.)
        if self.visualize:
            cmap_id = COLORMAP_IDS.get(self.colormap, COLORMAP_IDS["inferno"])
            depth_bgr = cv2.applyColorMap(depth_norm, cmap_id)
            output = self._convert_bgr_to_format(depth_bgr, fmt)
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

        # Append uint8 depth map as a custom buffer memory chunk.
        # Use new_allocate+fill: PyGI hides the maxsize arg in new_wrapped
        # (it derives it from data length), so passing it explicitly shifts
        # all subsequent args and causes a GI assertion crash.
        depth_bytes = DEPTH_META_HEADER + depth_norm.tobytes()
        tmp = Gst.Buffer.new_allocate(None, len(depth_bytes), None)
        tmp.fill(0, depth_bytes)
        buf.append_memory(tmp.get_memory(0))

    @staticmethod
    def _convert_bgr_to_format(bgr, fmt):
        """Convert a BGR numpy array to the target GStreamer video format."""
        import cv2

        if fmt == "RGB":
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif fmt == "BGR":
            return bgr
        elif fmt == "RGBA":
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
        elif fmt == "BGRA":
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        elif fmt == "ARGB":
            rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
            return np.roll(rgba, 1, axis=-1)  # RGBA -> ARGB
        elif fmt == "ABGR":
            bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
            return np.roll(bgra, 1, axis=-1)  # BGRA -> ABGR
        else:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


if CAN_REGISTER_ELEMENT:
    GObject.type_register(DepthTransform)
    __gstelementfactory__ = ("pyml_depth", Gst.Rank.NONE, DepthTransform)
else:
    GlobalLogger().warning(
        "The 'pyml_depth' element will not be registered because required modules are missing."
    )
