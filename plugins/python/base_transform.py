# BaseTransform
# Copyright (C) 2024-2025 Collabora Ltd.
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

import gi
from engine.engine_factory import EngineFactory
from engine.engine_helper import EngineHelper

gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
from gi.repository import Gst, GObject, GstBase  # noqa: E402

from log.logger_factory import LoggerFactory  # noqa: E402


class BaseTransform(GstBase.BaseTransform):
    """
    Base class for GStreamer transform elements that perform
    inference with a machine learning model. This class manages shared properties
    and handles model loading and device management via MLEngine.
    """

    __gstmetadata__ = (
        "BaseTransform",
        "Transform",
        "Generic machine learning model transform element",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    def __init__(self):
        super().__init__()
        self.logger = LoggerFactory.get(LoggerFactory.LOGGER_TYPE_GST)
        self.engine_helper = EngineHelper(self.logger)
        self.kwargs = {}
        self.__batch_size = 1
        self.__frame_stride = 1
        self.__model_name = None
        self.__device_queue_id = 0

    @GObject.Property(type=str)
    def device(self):
        "Device to run the inference on (cpu, cuda, cuda:0, cuda:1, etc.)"
        return self.engine_helper.device

    @device.setter
    def device(self, value):
        self.engine_helper.set_device(value)

    @GObject.Property(type=int, default=1)
    def batch_size(self):
        "Number of items to process in a batch"
        return self.__batch_size

    @batch_size.setter
    def batch_size(self, value):
        self.__batch_size = value
        if self.engine_helper.engine:
            self.engine_helper.engine.batch_size = value

    @GObject.Property(type=int, default=1)
    def frame_stride(self):
        "How often to process a frame"
        return self.__frame_stride

    @frame_stride.setter
    def frame_stride(self, value):
        self.__frame_stride = value
        if self.engine_helper.engine:
            self.engine_helper.engine.frame_stride = value

    @GObject.Property(type=str)
    def model_name(self):
        "Name of the pre-trained model or local model path"
        return self.__model_name

    @model_name.setter
    def model_name(self, value):
        self.__model_name = value

    @GObject.Property(type=str)
    def engine_name(self):
        "Machine Learning Engine to use : pytorch, tflite, tensorflow, onnx or openvino, or custom engine name"
        return self.engine_helper.engine_name

    @engine_name.setter
    def engine_name(self, value):
        self.engine_helper.engine_name = value

    @GObject.Property(type=int, default=1)
    def device_queue_id(self):
        "ID of the DeviceQueue from the pool to use"
        return self.__device_queue_id

    @device_queue_id.setter
    def device_queue_id(self, value):
        self.__device_queue_id = value
        if self.engine_helper.engine:
            self.engine_helper.engine.device_queue_id = value

    def _initialize_engine_if_needed(self):
        self.engine_helper.initialize_engine_if_needed()

    def initialize_engine(self):
        if self.engine_helper.engine_name:
            self.engine_helper.initialize_engine()
            self.engine_helper.engine.batch_size = self.__batch_size
            self.engine_helper.engine.frame_stride = self.__frame_stride
            if self.__device_queue_id:
                self.engine_helper.engine.device_queue_id = self.__device_queue_id
        else:
            self.logger.error(
                f"Unsupported ML engine: {self.engine_helper.engine_name}"
            )

    def do_load_model(self):
        self._initialize_engine_if_needed()
        if self.engine_helper.engine is None:
            self.logger.error(
                f"Cannot load model {self.model_name}: engine not initialized"
            )
            return
        if self.model_name is None:
            self.logger.warning(f"Cannot load model as model name is not set")
            return
        self.engine_helper.load_model(self.model_name)

    def get_model(self):
        """Gets the model from the engine."""
        self._initialize_engine_if_needed()
        if self.engine_helper.engine is None:
            self.logger.error(
                f"Cannot get model {self.model_name}: engine not initialized"
            )
            return None
        """Gets the model from the engine."""
        if self.engine_helper.engine:
            return self.engine_helper.engine.get_model()
        return None

    def set_model(self, model):
        """Sets the model in the engine."""
        self._initialize_engine_if_needed()
        if self.engine_helper.engine is None:
            self.logger.error("Cannot load model: engine not initialized")
            return False
        self.engine_helper.engine.model = model
        self.logger.info("Model set successfully in the engine.")

