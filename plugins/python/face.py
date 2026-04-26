# Face
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
    import json
    import os

    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstBase", "1.0")
    gi.require_version("GstVideo", "1.0")
    gi.require_version("GstAnalytics", "1.0")
    gi.require_version("GLib", "2.0")
    from gi.repository import Gst, GObject, GstAnalytics, GLib

    from base_objectdetector import BaseObjectDetector
    from engine.pytorch_engine import PyTorchEngine
    from engine.engine_factory import EngineFactory

except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(f"The 'face' element will not be available. Error {e}")

# Header prefix for face identity buffer metadata
FACE_META_HEADER = b"GST-FACE:"


class FaceEngine(PyTorchEngine):
    """
    PyTorch engine for face detection and recognition using InsightFace.

    Supports InsightFace model packs:
      buffalo_l   (large, most accurate)
      buffalo_s   (small, fastest)
      buffalo_sc  (small with recognition)
    """

    def do_load_model(self, model_name, **kwargs):
        try:
            from insightface.app import FaceAnalysis

            self.app = FaceAnalysis(
                name=model_name,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            self.gallery = {}
            self.logger.info(f"InsightFace model '{model_name}' loaded")
        except Exception as e:
            raise ValueError(f"Failed to load InsightFace model '{model_name}': {e}")

    def load_gallery(self, gallery_path):
        """Load known face embeddings from a directory of images."""
        import numpy as np
        from PIL import Image

        if not gallery_path or not os.path.isdir(gallery_path):
            return

        self.gallery = {}
        for fname in os.listdir(gallery_path):
            fpath = os.path.join(gallery_path, fname)
            if not os.path.isfile(fpath):
                continue
            try:
                img = np.array(Image.open(fpath).convert("RGB"))
                faces = self.app.get(img)
                if faces:
                    name = os.path.splitext(fname)[0]
                    self.gallery[name] = faces[0].embedding
                    self.logger.info(f"Loaded gallery face: {name}")
            except Exception as e:
                self.logger.warning(f"Failed to load gallery image '{fname}': {e}")

    def do_forward(self, frames, threshold=0.5):
        import numpy as np

        is_batch = isinstance(frames, np.ndarray) and frames.ndim == 4
        if not is_batch:
            frames = frames[np.newaxis]

        results = []
        for frame in frames:
            try:
                faces = self.app.get(frame.astype(np.uint8))
                detections = []
                for face in faces:
                    bbox = face.bbox.astype(float).tolist()
                    score = float(face.det_score)
                    embedding = face.embedding

                    identity = "unknown"
                    best_sim = 0.0
                    if self.gallery and embedding is not None:
                        for name, gallery_emb in self.gallery.items():
                            sim = float(
                                np.dot(embedding, gallery_emb)
                                / (
                                    np.linalg.norm(embedding)
                                    * np.linalg.norm(gallery_emb)
                                    + 1e-8
                                )
                            )
                            if sim > best_sim:
                                best_sim = sim
                                if sim >= threshold:
                                    identity = name

                    detections.append(
                        {
                            "bbox": bbox,
                            "score": score,
                            "identity": identity,
                            "similarity": best_sim,
                        }
                    )
                results.append(detections)
            except Exception as e:
                self.logger.error(f"Face inference error on frame: {e}")
                results.append([])

        return results[0] if not is_batch else results


class FaceTransform(BaseObjectDetector):
    """
    GStreamer element for face detection and recognition using InsightFace.

    Set model-name to an InsightFace model pack, e.g.: buffalo_l

    Bounding boxes are attached as GstAnalytics od_mtd metadata.
    Face identities are appended as a GST-FACE: memory chunk (JSON).

    Set gallery-path to a directory of images (one face per image,
    filename = person name) for face recognition.
    """

    __gstmetadata__ = (
        "Face Detection",
        "Transform",
        "Face detection and recognition using InsightFace",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    gallery_path = GObject.Property(
        type=str,
        default="",
        nick="Gallery Path",
        blurb="Path to directory of known face images for recognition",
        flags=GObject.ParamFlags.READWRITE,
    )

    threshold = GObject.Property(
        type=float,
        default=0.5,
        minimum=0.0,
        maximum=1.0,
        nick="Recognition Threshold",
        blurb="Cosine similarity threshold for face recognition",
        flags=GObject.ParamFlags.READWRITE,
    )

    def __init__(self):
        super().__init__()
        self.mgr.engine_name = "pyml_face_engine"
        EngineFactory.register(self.mgr.engine_name, FaceEngine)
        self._gallery_loaded = False

    @GObject.Property(type=str)
    def engine_name(self):
        """Machine Learning Engine (read-only for this element)."""
        return self.mgr.engine_name

    @engine_name.setter
    def engine_name(self, value):
        raise ValueError("'engine_name' is read-only for pyml_face")

    def do_decode(self, buf, result, stream_idx=0):
        # Load gallery on first decode if gallery-path is set
        if not self._gallery_loaded and self.gallery_path and self.engine:
            self.engine.load_gallery(self.gallery_path)
            self._gallery_loaded = True

        if not result or len(result) == 0:
            self.logger.info(f"Stream {stream_idx}: no faces detected")
            return

        meta = GstAnalytics.buffer_add_analytics_relation_meta(buf)
        if not meta:
            self.logger.error("Failed to add analytics relation metadata")
            return

        face_data = []
        for i, det in enumerate(result):
            bbox = det["bbox"]
            x1, y1, x2, y2 = bbox
            score = det["score"]
            identity = det["identity"]
            similarity = det["similarity"]

            qk = GLib.quark_from_string(f"stream_{stream_idx}_face_{identity}")
            ret, _ = meta.add_od_mtd(
                qk,
                x1,
                y1,
                x2 - x1,
                y2 - y1,
                score,
            )
            if not ret:
                self.logger.error(f"Failed to add od_mtd for face {i}")
                continue

            face_data.append(
                {
                    "face_idx": i,
                    "bbox": bbox,
                    "score": score,
                    "identity": identity,
                    "similarity": similarity,
                }
            )

        # Append face identity metadata as a custom memory chunk
        if face_data:
            meta_bytes = FACE_META_HEADER + json.dumps(face_data).encode("utf-8")
            tmp = Gst.Buffer.new_allocate(None, len(meta_bytes), None)
            tmp.fill(0, meta_bytes)
            buf.append_memory(tmp.get_memory(0))
            self.logger.debug(
                f"Stream {stream_idx}: appended metadata for {len(face_data)} faces"
            )


if CAN_REGISTER_ELEMENT:
    GObject.type_register(FaceTransform)
    __gstelementfactory__ = ("pyml_face", Gst.Rank.NONE, FaceTransform)
else:
    GlobalLogger().warning(
        "The 'pyml_face' element will not be registered because required modules are missing."
    )
