# pyml_inference — generic passthrough for testing ML engines
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
    import numpy as np
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstBase", "1.0")
    gi.require_version("GstVideo", "1.0")
    from gi.repository import Gst, GObject

    from video_transform import VideoTransform
    from utils.muxed_buffer_processor import MuxedBufferProcessor

except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(
        f"The 'pyml_inference' element will not be available. Error {e}"
    )


class GenericInferenceTransform(VideoTransform):
    """
    Generic passthrough element for testing any ML engine via the engine-name property.
    Runs do_forward() on each frame and logs the result. Buffer passes through unchanged.

    engine-name: pytorch (default), onnx, tensorflow, tflite, openvino

    Example:
      gst-launch-1.0 filesrc location=data/people.mp4 ! decodebin name=d \
        d. ! queue ! videoconvert ! videoscale \
        ! "video/x-raw,format=RGB,width=640,height=480" \
        ! pyml_inference engine-name=onnx model-name=yolo11m.onnx device=cpu \
        ! fakesink
    """

    __gstmetadata__ = (
        "Generic ML Inference",
        "Transform",
        "Passthrough element for testing ML engines; logs do_forward() output",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    def do_start(self):
        result = super().do_start()
        self.logger.info(
            f"pyml_inference started — engine={self.mgr.engine_name} "
            f"model={self.model_name} device={self.mgr.device}"
        )
        return result

    def do_transform_ip(self, buf):
        try:
            processor = MuxedBufferProcessor(
                self.logger, self.width, self.height, 30, 1
            )
            frames, _, num_sources, _ = processor.extract_frames(buf, self.sinkpad)
            if frames is None:
                return Gst.FlowReturn.ERROR

            frame = frames[0] if num_sources > 1 else frames

            if not self.engine:
                return Gst.FlowReturn.OK

            result = self.engine.do_forward(frame)
            if result is not None:
                self.logger.info(f"inference result: {result}")

            return Gst.FlowReturn.OK

        except Exception as e:
            self.logger.error(f"inference error: {e}")
            return Gst.FlowReturn.ERROR


if CAN_REGISTER_ELEMENT:
    GObject.type_register(GenericInferenceTransform)
    __gstelementfactory__ = ("pyml_inference", Gst.Rank.NONE, GenericInferenceTransform)
else:
    GlobalLogger().warning(
        "The 'pyml_inference' element will not be registered because required modules are missing."
    )
