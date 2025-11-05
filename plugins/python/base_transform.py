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
        self.engine = None
        self.kwargs = {}
        self.__engine_name = EngineFactory.PYTORCH_ENGINE
        self.__batch_size = 1
        self.__frame_stride = 1
        self.__model_name = None
        self.__device_queue_id = 0

    @GObject.Property(type=str)
    def device(self):
        "Device to run the inference on (cpu, cuda, cuda:0, cuda:1, etc.)"
        if self.engine:
            return self.engine.get_device()

    @device.setter
    def device(self, value):
        self.__device = value
        # Only set the device if the engine is initialized
        if self.engine:
            self.engine.set_device(value)
            self.do_load_model()

    @GObject.Property(type=int, default=1)
    def batch_size(self):
        "Number of items to process in a batch"
        return self.__batch_size

    @batch_size.setter
    def batch_size(self, value):
        self.__batch_size = value
        if self.engine:
            self.engine.batch_size = value

    @GObject.Property(type=int, default=1)
    def frame_stride(self):
        "How often to process a frame"
        return self.__frame_stride

    @frame_stride.setter
    def frame_stride(self, value):
        self.__frame_stride = value
        if self.engine:
            self.engine.frame_stride = value

    @GObject.Property(type=str)
    def model_name(self):
        "Name of the pre-trained model or local model path"
        return self.__model_name

    @model_name.setter
    def model_name(self, value):
        self.__model_name = value
        self.do_load_model()

    @GObject.Property(type=str)
    def engine_name(self):
        "Machine Learning Engine to use : pytorch, tflite, tensorflow, onnx or openvino, or custom engine name"
        return self.__engine_name

    @engine_name.setter
    def engine_name(self, value):
        self.__engine_name = value
        if self.device:
            self.initialize_engine()
            self.do_load_model()

    @GObject.Property(type=int, default=1)
    def device_queue_id(self):
        "ID of the DeviceQueue from the pool to use"
        return self.__device_queue_id

    @device_queue_id.setter
    def device_queue_id(self, value):
        self.__device_queue_id = value
        if self.engine:
            self.engine.device_queue_id = value

    def _initialize_engine_if_needed(self):
        """Initialize the engine if it hasn't been initialized yet."""
        if not self.engine and self.engine_name:
            self.initialize_engine()

    def initialize_engine(self):
        """Initialize the machine learning engine based on the engine_name property."""
        if self.engine_name:
            self.engine = EngineFactory.create(self.engine_name, self.device)
            self.engine.batch_size = self.batch_size
            self.engine.frame_stride = self.frame_stride
            if self.device_queue_id:
                self.engine.device_queue_id = self.device_queue_id
            self.do_load_model()
        else:
            self.logger.error(f"Unsupported engine: {self.engine_name}")
            return

    def do_load_model(self):
        self._initialize_engine_if_needed()
        """Loads the model using the current engine."""
        if self.engine and self.model_name:
            self.engine.load_model(self.model_name, **self.kwargs)
        else:
            self.logger.warning(
                f"Engine is not present, unable to load the model {self.model_name}."
            )

    def get_model(self):
        """Gets the model from the engine."""
        self._initialize_engine_if_needed()
        if self.engine is None:
            self.logger.error("Cannot get model: engine not initialized")
            return None
        """Gets the model from the engine."""
        return self.engine.get_model()
