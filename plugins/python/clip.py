# CLIP / SigLIP zero-shot image classification
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
    import threading

    import gi
    import numpy as np

    gi.require_version("Gst", "1.0")
    gi.require_version("GstBase", "1.0")
    gi.require_version("GstVideo", "1.0")
    gi.require_version("GstAnalytics", "1.0")
    gi.require_version("GLib", "2.0")
    from gi.repository import Gst, GObject, GstAnalytics, GLib

    from video_transform import VideoTransform
    from utils.muxed_buffer_processor import MuxedBufferProcessor
    from engine.pytorch_engine import PyTorchEngine
    from engine.engine_factory import EngineFactory

except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(f"The 'clip' element will not be available. Error {e}")


class ClipEngine(PyTorchEngine):
    """
    PyTorch engine for CLIP and SigLIP zero-shot image classification.

    Works with any HuggingFace CLIP-compatible model:
      openai/clip-vit-base-patch32
      openai/clip-vit-large-patch14
      google/siglip-base-patch16-224
      google/siglip-large-patch16-384
    """

    def __init__(self):
        super().__init__()
        self._labels = []

    @property
    def clip_labels(self):
        return self._labels

    @clip_labels.setter
    def clip_labels(self, value):
        self._labels = value

    def do_load_model(self, model_name, **kwargs):
        try:
            from transformers import AutoProcessor, AutoModel

            self.image_processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.execute_with_stream(lambda: self.model.to(self.device))
            self.model.eval()
            self.logger.info(f"CLIP model '{model_name}' loaded on {self.device}")
        except Exception as e:
            raise ValueError(f"Failed to load CLIP model '{model_name}': {e}")

    def do_forward(self, frame):
        """
        Run zero-shot classification.

        Args:
            frame: RGB numpy array [H, W, 3]

        Returns:
            List of (label, probability) tuples sorted by probability descending,
            or None if no labels are set.
        """
        import torch
        from PIL import Image

        if not self._labels:
            self.logger.warning("No labels set — set the 'labels' property")
            return None

        try:
            pil_img = Image.fromarray(frame.astype(np.uint8))
            inputs = self.image_processor(
                text=self._labels,
                images=pil_img,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            # logits_per_image: [1, num_labels]
            probs = outputs.logits_per_image.softmax(dim=1)[0]
            results = [(label, prob.item()) for label, prob in zip(self._labels, probs)]
            results.sort(key=lambda x: x[1], reverse=True)
            return results

        except Exception as e:
            self.logger.error(f"CLIP inference error: {e}")
            return None


class CLIPTransform(VideoTransform):
    """
    GStreamer element for zero-shot image classification using CLIP or SigLIP.

    Classifies each frame against a user-defined set of text labels without
    any pre-training on those specific classes. Results are attached as
    GstAnalytics metadata and logged.

    Supported models (set via model-name property):
      openai/clip-vit-base-patch32       (default, ~600 MB)
      openai/clip-vit-large-patch14      (more accurate, ~1.7 GB)
      google/siglip-base-patch16-224     (SigLIP, better zero-shot accuracy)
      google/siglip-large-patch16-384    (SigLIP large)

    Example pipeline:
      gst-launch-1.0 filesrc location=data/people.mp4 ! decodebin \\
        ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 \\
        ! pyml_clip model-name=openai/clip-vit-base-patch32 device=cuda \\
                    labels="person, bicycle, car, dog, cat" top-k=3 \\
        ! fakesink
    """

    __gstmetadata__ = (
        "CLIP Zero-Shot Classifier",
        "Transform",
        "Zero-shot image classification using CLIP or SigLIP",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    top_k = GObject.Property(
        type=int,
        default=3,
        minimum=1,
        maximum=100,
        nick="Top K",
        blurb="Number of top labels to attach as metadata",
        flags=GObject.ParamFlags.READWRITE,
    )

    threshold = GObject.Property(
        type=float,
        default=0.0,
        minimum=0.0,
        maximum=1.0,
        nick="Threshold",
        blurb="Minimum probability to include a label in the metadata output",
        flags=GObject.ParamFlags.READWRITE,
    )

    def __init__(self):
        super().__init__()
        self.mgr.engine_name = "pyml_clip_engine"
        EngineFactory.register(self.mgr.engine_name, ClipEngine)
        self._labels_str = ""
        self._labels_list = []
        self._last_results = None
        # Background inference thread state
        self._infer_lock = threading.Lock()
        self._infer_event = threading.Event()
        self._pending_frame = None
        self._infer_thread = None
        self._running = False

    @GObject.Property(
        type=str,
        default="",
        nick="Labels",
        blurb="Comma-separated list of text labels to classify against, "
        "e.g. 'person, car, bicycle, dog, cat'",
        flags=GObject.ParamFlags.READWRITE,
    )
    def labels(self):
        return self._labels_str

    @labels.setter
    def labels(self, value):
        self._labels_str = value
        self._labels_list = [l.strip() for l in value.split(",") if l.strip()]
        if self.engine:
            self.engine.clip_labels = self._labels_list
        self.logger.info(f"Labels set to: {self._labels_list}")

    @GObject.Property(type=str)
    def engine_name(self):
        """Machine Learning Engine (read-only for this element)."""
        return self.mgr.engine_name

    @engine_name.setter
    def engine_name(self, value):
        raise ValueError("'engine_name' is read-only for pyml_clip")

    def do_start(self):
        result = super().do_start()
        # Push labels into the engine after it has been initialised
        if self.engine and self._labels_list:
            self.engine.clip_labels = self._labels_list
        # Start background inference thread
        self._running = True
        self._infer_thread = threading.Thread(
            target=self._inference_worker, daemon=True
        )
        self._infer_thread.start()
        return result

    def do_stop(self):
        self._running = False
        self._infer_event.set()  # Wake the thread so it sees _running=False
        self._infer_thread = None
        return True

    def _inference_worker(self):
        """Background thread: runs CLIP inference on the latest pending frame."""
        while self._running:
            self._infer_event.wait()
            self._infer_event.clear()
            if not self._running:
                break
            with self._infer_lock:
                frame = self._pending_frame
                self._pending_frame = None
            if frame is None or not self.engine:
                continue
            results = self.engine.do_forward(frame)
            if results is not None:
                with self._infer_lock:
                    self._last_results = results

    def do_transform_ip(self, buf):
        try:
            processor = MuxedBufferProcessor(
                self.logger, self.width, self.height, 30, 1
            )
            frames, _, num_sources, _ = processor.extract_frames(buf, self.sinkpad)
            if frames is None:
                return Gst.FlowReturn.ERROR

            # Use first frame for classification (batch not typical for CLIP)
            frame = frames[0] if num_sources > 1 else frames

            if self.engine:
                self.engine.clip_labels = self._labels_list

            # Post frame to background thread; never block the streaming thread
            with self._infer_lock:
                self._pending_frame = frame.copy()
            self._infer_event.set()

            with self._infer_lock:
                results = self._last_results

            if results is None:
                return Gst.FlowReturn.OK

            self._attach_metadata(buf, results)
            return Gst.FlowReturn.OK

        except Exception as e:
            self.logger.error(f"CLIP transform error: {e}")
            return Gst.FlowReturn.ERROR

    def _attach_metadata(self, buf, results):
        """Attach top-k classification results above threshold as GstAnalytics metadata."""
        meta = GstAnalytics.buffer_add_analytics_relation_meta(buf)
        if not meta:
            self.logger.error("Failed to add analytics relation metadata")
            return

        attached = 0
        for label, prob in results:
            if attached >= self.top_k:
                break
            if prob < self.threshold:
                break

            qk = GLib.quark_from_string(f"clip_{label.replace(' ', '_')}")
            ret, _ = meta.add_od_mtd(qk, 0, 0, self.width, self.height, prob)
            if ret:
                attached += 1
                self.logger.info(f"CLIP: {label} = {prob:.3f}")


if CAN_REGISTER_ELEMENT:
    GObject.type_register(CLIPTransform)
    __gstelementfactory__ = ("pyml_clip", Gst.Rank.NONE, CLIPTransform)
else:
    GlobalLogger().warning(
        "The 'pyml_clip' element will not be registered because required modules are missing."
    )
