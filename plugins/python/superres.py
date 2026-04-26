# Super Resolution
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
    GlobalLogger().warning(f"The 'superres' element will not be available. Error {e}")


class SuperResEngine(PyTorchEngine):
    """
    PyTorch engine for image super-resolution using Real-ESRGAN.

    Supports model variants:
      real-esrgan-x4   (4x upscale, general purpose)
      real-esrgan-x2   (2x upscale)
    """

    def do_load_model(self, model_name, **kwargs):
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            scale = 4
            if "x2" in model_name:
                scale = 2

            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=scale,
            )

            model_url = (
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
                if scale == 4
                else "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
            )

            gpu_id = 0 if str(self.device) != "cpu" else None
            self.upsampler = RealESRGANer(
                scale=scale,
                model_path=model_url,
                model=model,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=False,
                gpu_id=gpu_id,
            )
            self._scale = scale
            self.logger.info(f"Real-ESRGAN model '{model_name}' (scale={scale}) loaded")
        except Exception as e:
            raise ValueError(f"Failed to load Real-ESRGAN model '{model_name}': {e}")

    def do_forward(self, frames):
        import cv2
        import numpy as np

        is_batch = isinstance(frames, np.ndarray) and frames.ndim == 4
        if not is_batch:
            frames = frames[np.newaxis]

        results = []
        for frame in frames:
            try:
                # Real-ESRGAN expects BGR input
                bgr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
                output, _ = self.upsampler.enhance(bgr, outscale=self._scale)
                # Convert back to RGB
                rgb_out = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                results.append(rgb_out)
            except Exception as e:
                self.logger.error(f"Super-resolution inference error: {e}")
                results.append(None)

        return results[0] if not is_batch else results


class SuperResTransform(VideoTransform):
    """
    GStreamer element for image super-resolution using Real-ESRGAN.

    Set model-name to a Real-ESRGAN variant: real-esrgan-x4 or real-esrgan-x2.

    The upscaled frame is resized back to the original buffer dimensions
    (in-place transform). This provides enhanced detail while maintaining
    pipeline compatibility. Use frame-stride to reduce compute load.
    """

    __gstmetadata__ = (
        "Super Resolution",
        "Transform",
        "Image super-resolution using Real-ESRGAN",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    scale_factor = GObject.Property(
        type=int,
        default=4,
        minimum=2,
        maximum=8,
        nick="Scale Factor",
        blurb="Upscaling factor (2 or 4)",
        flags=GObject.ParamFlags.READWRITE,
    )

    def __init__(self):
        super().__init__()
        self.mgr.engine_name = "pyml_superres_engine"
        EngineFactory.register(self.mgr.engine_name, SuperResEngine)
        self.format_converter = FormatConverter()

    @GObject.Property(type=str)
    def engine_name(self):
        """Machine Learning Engine (read-only for this element)."""
        return self.mgr.engine_name

    @engine_name.setter
    def engine_name(self, value):
        raise ValueError("'engine_name' is read-only for pyml_superres")

    def do_transform_ip(self, buf):
        try:
            processor = MuxedBufferProcessor(
                self.logger, self.width, self.height, 30, 1
            )
            frames, _, num_sources, fmt = processor.extract_frames(buf, self.sinkpad)
            if frames is None:
                return Gst.FlowReturn.ERROR

            frame = frames[0] if frames.ndim == 4 else frames
            upscaled = self._do_forward(frame)
            if upscaled is None:
                return Gst.FlowReturn.OK

            self._apply_superres(buf, upscaled, fmt)
            return Gst.FlowReturn.OK

        except Exception as e:
            self.logger.error(f"Super-resolution transform error: {e}")
            return Gst.FlowReturn.ERROR

    def _do_forward(self, frame):
        if self.engine:
            return self.engine.do_forward(frame)
        return None

    def _apply_superres(self, buf, upscaled, fmt):
        """Resize upscaled frame back to original dimensions and write to buffer."""
        import cv2
        import numpy as np

        # Resize back to original buffer dimensions for in-place compatibility
        resized = cv2.resize(
            upscaled, (self.width, self.height), interpolation=cv2.INTER_LANCZOS4
        )
        output = self._convert_rgb_to_format(resized, fmt)
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
    GObject.type_register(SuperResTransform)
    __gstelementfactory__ = ("pyml_superres", Gst.Rank.NONE, SuperResTransform)
else:
    GlobalLogger().warning(
        "The 'pyml_superres' element will not be registered because required modules are missing."
    )
