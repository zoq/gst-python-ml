# Optical Flow
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
    GlobalLogger().warning(
        f"The 'optical_flow' element will not be available. Error {e}"
    )

# Colormap names for flow visualization
FLOW_COLORMAPS = {
    "hsv": None,  # custom HSV-based flow coloring
    "jet": 2,
    "viridis": 16,
    "inferno": 9,
}


class OpticalFlowEngine(PyTorchEngine):
    """
    PyTorch engine for dense optical flow estimation using RAFT.

    Supports torchvision RAFT model variants:
      raft_large   (most accurate)
      raft_small   (fastest)
    """

    def do_load_model(self, model_name, **kwargs):
        try:
            from torchvision.models.optical_flow import (
                raft_large,
                raft_small,
                Raft_Large_Weights,
                Raft_Small_Weights,
            )

            if model_name == "raft_small":
                weights = Raft_Small_Weights.DEFAULT
                self.model = raft_small(weights=weights)
            else:
                weights = Raft_Large_Weights.DEFAULT
                self.model = raft_large(weights=weights)

            self.transforms = weights.transforms()
            self.execute_with_stream(lambda: self.model.to(self.device))
            self.model.eval()
            self.logger.info(f"RAFT model '{model_name}' loaded on {self.device}")
        except Exception as e:
            raise ValueError(f"Failed to load RAFT model '{model_name}': {e}")

    def do_forward(self, prev_frame, curr_frame):
        import torch

        try:
            H, W = curr_frame.shape[:2]

            # Convert HWC uint8 -> CHW float tensor
            prev_t = torch.from_numpy(prev_frame).permute(2, 0, 1).float()
            curr_t = torch.from_numpy(curr_frame).permute(2, 0, 1).float()

            # RAFT requires dimensions divisible by 8
            pad_h = (8 - H % 8) % 8
            pad_w = (8 - W % 8) % 8
            if pad_h > 0 or pad_w > 0:
                prev_t = torch.nn.functional.pad(prev_t, (0, pad_w, 0, pad_h))
                curr_t = torch.nn.functional.pad(curr_t, (0, pad_w, 0, pad_h))

            prev_t, curr_t = self.transforms(prev_t, curr_t)
            prev_batch = prev_t.unsqueeze(0).to(self.device)
            curr_batch = curr_t.unsqueeze(0).to(self.device)

            with torch.no_grad():
                flow_predictions = self.model(prev_batch, curr_batch)

            # RAFT returns a list of flow predictions; take the last (finest)
            flow = flow_predictions[-1].squeeze(0).cpu().numpy()
            # flow shape: (2, H', W') -> transpose to (H, W, 2) and crop
            flow = flow.transpose(1, 2, 0)[:H, :W]
            return flow

        except Exception as e:
            self.logger.error(f"Optical flow inference error: {e}")
            return None


class OpticalFlowTransform(VideoTransform):
    """
    GStreamer element for dense optical flow estimation using RAFT.

    Set model-name to a RAFT variant: raft_large or raft_small.

    Computes dense optical flow between consecutive frames. When
    visualize=True (default), the flow is rendered as a color-coded overlay
    on the video frame using HSV color space (hue=direction, value=magnitude).
    """

    __gstmetadata__ = (
        "Optical Flow",
        "Transform",
        "Dense optical flow estimation using RAFT",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    visualize = GObject.Property(
        type=bool,
        default=True,
        nick="Visualize Flow",
        blurb="Overlay color-coded optical flow on the video frame",
        flags=GObject.ParamFlags.READWRITE,
    )

    colormap = GObject.Property(
        type=str,
        default="hsv",
        nick="Colormap",
        blurb="Colormap for flow visualization: hsv, jet, viridis, inferno",
        flags=GObject.ParamFlags.READWRITE,
    )

    def __init__(self):
        super().__init__()
        self.mgr.engine_name = "pyml_optical_flow_engine"
        EngineFactory.register(self.mgr.engine_name, OpticalFlowEngine)
        self.format_converter = FormatConverter()
        self._prev_frame = None

    @GObject.Property(type=str)
    def engine_name(self):
        """Machine Learning Engine (read-only for this element)."""
        return self.mgr.engine_name

    @engine_name.setter
    def engine_name(self, value):
        raise ValueError("'engine_name' is read-only for pyml_optical_flow")

    def do_transform_ip(self, buf):
        try:
            processor = MuxedBufferProcessor(
                self.logger, self.width, self.height, 30, 1
            )
            frames, _, num_sources, fmt = processor.extract_frames(buf, self.sinkpad)
            if frames is None:
                return Gst.FlowReturn.ERROR

            frame = frames[0] if frames.ndim == 4 else frames

            if self._prev_frame is None:
                self._prev_frame = frame.copy()
                return Gst.FlowReturn.OK

            flow = self._do_forward(self._prev_frame, frame)
            self._prev_frame = frame.copy()

            if flow is None:
                return Gst.FlowReturn.OK

            if self.visualize:
                self._apply_flow_vis(buf, flow, frame, fmt)

            return Gst.FlowReturn.OK

        except Exception as e:
            self.logger.error(f"Optical flow transform error: {e}")
            return Gst.FlowReturn.ERROR

    def _do_forward(self, prev_frame, curr_frame):
        if self.engine:
            return self.engine.do_forward(prev_frame, curr_frame)
        return None

    def _apply_flow_vis(self, buf, flow, frame, fmt):
        """Render flow as a color overlay and write back to buffer."""
        import cv2
        import numpy as np

        flow_vis = self._flow_to_color(flow)
        blended = cv2.addWeighted(frame, 0.5, flow_vis, 0.5, 0)
        output = self._convert_rgb_to_format(blended, fmt)
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

    @staticmethod
    def _flow_to_color(flow):
        """Convert optical flow (H, W, 2) to an RGB color image using HSV encoding."""
        import cv2
        import numpy as np

        fx, fy = flow[..., 0], flow[..., 1]
        mag = np.sqrt(fx**2 + fy**2)
        ang = np.arctan2(fy, fx)

        hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
        hsv[..., 0] = ((ang + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
        hsv[..., 1] = 255
        mag_norm = mag / (mag.max() + 1e-8)
        hsv[..., 2] = (mag_norm * 255).astype(np.uint8)

        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

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
    GObject.type_register(OpticalFlowTransform)
    __gstelementfactory__ = ("pyml_optical_flow", Gst.Rank.NONE, OpticalFlowTransform)
else:
    GlobalLogger().warning(
        "The 'pyml_optical_flow' element will not be registered because required modules are missing."
    )
