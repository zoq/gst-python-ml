# Anomaly Detection
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
    GlobalLogger().warning(f"The 'anomaly' element will not be available. Error {e}")

# Header prefix for anomaly detection buffer metadata
ANOMALY_META_HEADER = b"GST-ANOMALY:"


class AnomalyEngine(PyTorchEngine):
    """
    PyTorch engine for anomaly detection using a PatchCore-like approach.

    Uses a pretrained feature extractor (WideResNet50 or ResNet) to extract
    patch-level features and compare them against a reference distribution.

    Supports torchvision backbone models:
      wide_resnet50_2
      resnet50
      resnet18
    """

    def do_load_model(self, model_name, **kwargs):
        try:
            import torch
            import torchvision.models as models

            model_fn = getattr(models, model_name, None)
            if model_fn is None:
                raise ValueError(f"Unknown backbone model: {model_name}")

            self.backbone = model_fn(weights="DEFAULT")
            # Remove the final FC layer to get feature maps
            self.feature_layers = torch.nn.Sequential(
                *list(self.backbone.children())[:-2]
            )
            self.execute_with_stream(lambda: self.feature_layers.to(self.device))
            self.feature_layers.eval()

            self.reference_features = None
            self._transform = None
            self.logger.info(f"Anomaly backbone '{model_name}' loaded on {self.device}")
        except Exception as e:
            raise ValueError(f"Failed to load anomaly backbone '{model_name}': {e}")

    def load_reference(self, reference_path):
        """Load precomputed reference features from a .npy file."""
        import numpy as np

        try:
            self.reference_features = np.load(reference_path)
            self.logger.info(
                f"Loaded reference features from '{reference_path}': "
                f"shape={self.reference_features.shape}"
            )
        except Exception as e:
            self.logger.warning(f"Failed to load reference features: {e}")

    def _get_transform(self):
        if self._transform is None:
            from torchvision import transforms

            self._transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
        return self._transform

    def do_forward(self, frames, threshold=0.5):
        import numpy as np
        import torch

        is_batch = isinstance(frames, np.ndarray) and frames.ndim == 4
        if not is_batch:
            frames = frames[np.newaxis]

        transform = self._get_transform()
        results = []
        for frame in frames:
            try:
                tensor = transform(frame.astype(np.uint8)).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    features = self.feature_layers(tensor)

                # Global average pool to get a feature vector
                feat_vec = features.mean(dim=[2, 3]).squeeze(0).cpu().numpy()

                # Compute anomaly score against reference distribution
                anomaly_score = 0.0
                if self.reference_features is not None:
                    distances = np.linalg.norm(
                        self.reference_features - feat_vec, axis=-1
                    )
                    anomaly_score = float(distances.min())

                # Generate a spatial anomaly heatmap from feature map distances
                feat_map = features.squeeze(0).cpu().numpy()
                heatmap = np.linalg.norm(feat_map, axis=0)
                heatmap = (heatmap - heatmap.min()) / (
                    heatmap.max() - heatmap.min() + 1e-8
                )

                is_anomaly = anomaly_score >= threshold

                results.append(
                    {
                        "score": anomaly_score,
                        "is_anomaly": is_anomaly,
                        "heatmap": heatmap,
                    }
                )
            except Exception as e:
                self.logger.error(f"Anomaly inference error on frame: {e}")
                results.append(
                    {
                        "score": 0.0,
                        "is_anomaly": False,
                        "heatmap": None,
                    }
                )

        return results[0] if not is_batch else results


class AnomalyTransform(VideoTransform):
    """
    GStreamer element for anomaly detection in video frames.

    Uses a pretrained feature extractor to compute patch-level anomaly scores
    against a reference distribution of normal frames.

    Set reference-path to a .npy file containing precomputed reference features
    from normal samples. When draw-heatmap=True (default), an anomaly heatmap
    is overlaid on frames that exceed the threshold.

    Anomaly scores are always attached as a GST-ANOMALY: memory chunk (JSON).
    """

    __gstmetadata__ = (
        "Anomaly Detection",
        "Transform",
        "Video anomaly detection using feature extraction and PatchCore scoring",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    threshold = GObject.Property(
        type=float,
        default=0.5,
        minimum=0.0,
        maximum=100.0,
        nick="Anomaly Threshold",
        blurb="Anomaly score threshold above which a frame is flagged",
        flags=GObject.ParamFlags.READWRITE,
    )

    reference_path = GObject.Property(
        type=str,
        default="",
        nick="Reference Path",
        blurb="Path to .npy file with reference feature vectors from normal frames",
        flags=GObject.ParamFlags.READWRITE,
    )

    draw_heatmap = GObject.Property(
        type=bool,
        default=True,
        nick="Draw Heatmap",
        blurb="Overlay anomaly heatmap on frames above threshold",
        flags=GObject.ParamFlags.READWRITE,
    )

    def __init__(self):
        super().__init__()
        self.mgr.engine_name = "pyml_anomaly_engine"
        EngineFactory.register(self.mgr.engine_name, AnomalyEngine)
        self.format_converter = FormatConverter()
        self._reference_loaded = False

    @GObject.Property(type=str)
    def engine_name(self):
        """Machine Learning Engine (read-only for this element)."""
        return self.mgr.engine_name

    @engine_name.setter
    def engine_name(self, value):
        raise ValueError("'engine_name' is read-only for pyml_anomaly")

    def do_transform_ip(self, buf):
        try:
            # Load reference features on first transform if path is set
            if not self._reference_loaded and self.reference_path and self.engine:
                self.engine.load_reference(self.reference_path)
                self._reference_loaded = True

            processor = MuxedBufferProcessor(
                self.logger, self.width, self.height, 30, 1
            )
            frames, _, num_sources, fmt = processor.extract_frames(buf, self.sinkpad)
            if frames is None:
                return Gst.FlowReturn.ERROR

            frame = frames[0] if frames.ndim == 4 else frames
            result = self._do_forward(frame)
            if result is None:
                return Gst.FlowReturn.OK

            self._apply_anomaly(buf, result, fmt, frame)
            return Gst.FlowReturn.OK

        except Exception as e:
            self.logger.error(f"Anomaly detection transform error: {e}")
            return Gst.FlowReturn.ERROR

    def _do_forward(self, frame):
        if self.engine:
            return self.engine.do_forward(frame, threshold=self.threshold)
        return None

    def _apply_anomaly(self, buf, result, fmt, frame):
        """Overlay heatmap on frame and append anomaly metadata."""
        import cv2
        import numpy as np

        is_anomaly = result.get("is_anomaly", False)
        heatmap = result.get("heatmap")

        # Draw heatmap overlay before appending read-only metadata memory
        if self.draw_heatmap and is_anomaly and heatmap is not None:
            H, W = frame.shape[:2]
            heatmap_resized = cv2.resize(heatmap, (W, H))
            heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

            overlay = cv2.addWeighted(frame, 0.6, heatmap_rgb, 0.4, 0)

            # Draw anomaly score text
            score = result.get("score", 0.0)
            text = f"ANOMALY: {score:.3f}"
            cv2.putText(
                overlay,
                text,
                (12, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 0, 0),
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

        # Append anomaly metadata (without the numpy heatmap)
        meta = {
            "score": result.get("score", 0.0),
            "is_anomaly": is_anomaly,
        }
        meta_bytes = ANOMALY_META_HEADER + json.dumps(meta).encode("utf-8")
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
    GObject.type_register(AnomalyTransform)
    __gstelementfactory__ = ("pyml_anomaly", Gst.Rank.NONE, AnomalyTransform)
else:
    GlobalLogger().warning(
        "The 'pyml_anomaly' element will not be registered because required modules are missing."
    )
