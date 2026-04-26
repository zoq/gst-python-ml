# Embedding
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
    import struct

    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstBase", "1.0")
    gi.require_version("GstVideo", "1.0")
    from gi.repository import Gst, GObject

    from video_transform import VideoTransform
    from utils.format_converter import FormatConverter
    from engine.pytorch_engine import PyTorchEngine
    from engine.engine_factory import EngineFactory

except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(f"The 'embedding' element will not be available. Error {e}")

# Header prefix for embedding buffer metadata
EMBEDDING_META_HEADER = b"GST-EMBEDDING:"


class EmbeddingEngine(PyTorchEngine):
    """
    PyTorch engine for image/text embedding extraction.

    Supports CLIP and DINOv2 models via HuggingFace transformers:
      openai/clip-vit-large-patch14  (CLIP — image + text)
      facebook/dinov2-base           (DINOv2 — image only)
    """

    def __init__(self):
        super().__init__()
        self.processor = None
        self.tokenizer = None
        self.output_dim = 0
        self._is_clip = False

    def do_load_model(self, model_name, **kwargs):
        try:

            if "clip" in model_name.lower():
                self._load_clip(model_name)
            elif "dino" in model_name.lower():
                self._load_dinov2(model_name)
            else:
                # Default to CLIP-style loading
                self._load_clip(model_name)

            self.execute_with_stream(lambda: self.model.to(self.device))
            self.model.eval()
            self.logger.info(
                f"Embedding model '{model_name}' loaded on {self.device} "
                f"(dim={self.output_dim})"
            )
        except Exception as e:
            raise ValueError(f"Failed to load embedding model '{model_name}': {e}")

    def _load_clip(self, model_name):
        from transformers import CLIPModel, CLIPProcessor

        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self._is_clip = True
        # Determine output dim from config
        self.output_dim = self.model.config.projection_dim

    def _load_dinov2(self, model_name):
        from transformers import AutoModel, AutoImageProcessor

        self.model = AutoModel.from_pretrained(model_name)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self._is_clip = False
        self.output_dim = self.model.config.hidden_size

    def do_forward(self, frame, normalize=True):
        """
        Extract an embedding vector from a video frame.

        Args:
            frame: numpy RGB array (H, W, 3).
            normalize: if True, L2-normalize the embedding.

        Returns:
            numpy float32 array of shape (output_dim,), or None on failure.
        """
        import numpy as np
        import torch
        from PIL import Image

        try:
            pil_img = Image.fromarray(frame.astype(np.uint8))
            inputs = self.processor(images=pil_img, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                if self._is_clip:
                    emb = self.model.get_image_features(**inputs)
                else:
                    outputs = self.model(**inputs)
                    # Use CLS token embedding
                    emb = outputs.last_hidden_state[:, 0]

            emb = emb.squeeze(0).cpu().numpy().astype(np.float32)
            if normalize:
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
            return emb
        except Exception as e:
            self.logger.error(f"Embedding inference error: {e}")
            return None

    def do_text_embedding(self, text, normalize=True):
        """
        Extract a text embedding (CLIP only).

        Args:
            text: input string.
            normalize: if True, L2-normalize the embedding.

        Returns:
            numpy float32 array of shape (output_dim,), or None.
        """
        import numpy as np
        import torch

        if not self._is_clip:
            self.logger.warning("Text embeddings only supported for CLIP models")
            return None

        try:
            inputs = self.processor(text=[text], return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                emb = self.model.get_text_features(**inputs)
            emb = emb.squeeze(0).cpu().numpy().astype(np.float32)
            if normalize:
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
            return emb
        except Exception as e:
            self.logger.error(f"Text embedding error: {e}")
            return None


class EmbeddingTransform(VideoTransform):
    """
    GStreamer element for extracting frame embeddings for similarity search,
    clustering, or RAG.

    For each processed frame, extracts an embedding vector and appends it as a
    GST-EMBEDDING: memory chunk on the buffer. The chunk contains a JSON header
    followed by the raw float32 embedding bytes.

    Downstream elements can read the embedding:
      for i in range(buf.n_memory()):
          data = bytes(buf.peek_memory(i).map(Gst.MapFlags.READ).data)
          if data.startswith(b"GST-EMBEDDING:"):
              payload = data[14:]
              header_len = int.from_bytes(payload[:4], "little")
              header = json.loads(payload[4:4+header_len])
              dim = header["dim"]
              emb = np.frombuffer(payload[4+header_len:], dtype=np.float32)

    When a text property is set (CLIP only), a cosine similarity score between
    the text and image embeddings is also included in the JSON header.

    Use frame-stride to control processing frequency:
      pyml_embedding model-name=openai/clip-vit-large-patch14 frame-stride=5
    """

    __gstmetadata__ = (
        "Embedding Extractor",
        "Transform",
        "Extract frame embeddings using CLIP or DINOv2 for similarity search and RAG",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    normalize = GObject.Property(
        type=bool,
        default=True,
        nick="Normalize",
        blurb="L2-normalize the embedding vector",
        flags=GObject.ParamFlags.READWRITE,
    )

    text = GObject.Property(
        type=str,
        default=None,
        nick="Text Query",
        blurb="Optional text for computing text-image similarity (CLIP only)",
        flags=GObject.ParamFlags.READWRITE,
    )

    def __init__(self):
        super().__init__()
        self.mgr.engine_name = "pyml_embedding_engine"
        EngineFactory.register(self.mgr.engine_name, EmbeddingEngine)
        self.format_converter = FormatConverter()
        self._frame_count = 0
        self._text_embedding = None
        self._cached_text = None

    @GObject.Property(type=str)
    def engine_name(self):
        """Machine Learning Engine (read-only for this element)."""
        return self.mgr.engine_name

    @engine_name.setter
    def engine_name(self, value):
        raise ValueError("'engine_name' is read-only for pyml_embedding")

    @GObject.Property(type=int, default=0)
    def output_dim(self):
        """Embedding dimensionality (read-only, set after model load)."""
        if self.engine:
            return self.engine.output_dim
        return 0

    @output_dim.setter
    def output_dim(self, value):
        raise ValueError("'output_dim' is read-only")

    def _update_text_embedding(self):
        """Recompute cached text embedding when the text property changes."""
        if self.engine is None:
            return
        if self.text and self.text != self._cached_text:
            self._text_embedding = self.engine.do_text_embedding(
                self.text, normalize=self.normalize
            )
            self._cached_text = self.text
        elif not self.text:
            self._text_embedding = None
            self._cached_text = None

    def do_transform_ip(self, buf):
        import numpy as np

        try:
            self._frame_count += 1
            if self.frame_stride > 1 and (self._frame_count % self.frame_stride) != 1:
                return Gst.FlowReturn.OK

            if self.engine is None:
                return Gst.FlowReturn.OK

            success, map_info = buf.map(Gst.MapFlags.READ)
            if not success:
                self.logger.error("Failed to map video buffer for reading")
                return Gst.FlowReturn.ERROR

            try:
                frame = self.format_converter.to_rgb(
                    map_info.data, self.width, self.height, buf, self.sinkpad
                )
            finally:
                buf.unmap(map_info)

            if frame is None:
                return Gst.FlowReturn.ERROR

            emb = self.engine.do_forward(frame, normalize=self.normalize)
            if emb is None:
                return Gst.FlowReturn.OK

            # Update text embedding if needed
            self._update_text_embedding()

            # Build JSON header with dimension info and optional similarity score
            header = {"dim": int(emb.shape[0]), "dtype": "float32"}
            if self._text_embedding is not None:
                similarity = float(np.dot(emb, self._text_embedding))
                header["text"] = self.text
                header["similarity"] = round(similarity, 6)

            header_bytes = json.dumps(header).encode("utf-8")
            header_len = struct.pack("<I", len(header_bytes))
            emb_bytes = emb.tobytes()

            # Append embedding as a custom buffer memory chunk.
            # Format: HEADER_PREFIX + 4-byte header length + JSON header + raw float32
            # Use new_allocate+fill: PyGI hides the maxsize arg in new_wrapped
            # (it derives it from data length), so passing it explicitly shifts
            # all subsequent args and causes a GI assertion crash.
            payload = EMBEDDING_META_HEADER + header_len + header_bytes + emb_bytes
            tmp = Gst.Buffer.new_allocate(None, len(payload), None)
            tmp.fill(0, payload)
            buf.append_memory(tmp.get_memory(0))

            self.logger.debug(
                f"Embedding extracted: dim={emb.shape[0]}"
                + (
                    f" sim={header.get('similarity', 'N/A')}"
                    if self._text_embedding is not None
                    else ""
                )
            )

            return Gst.FlowReturn.OK

        except Exception as e:
            self.logger.error(f"Embedding transform error: {e}")
            return Gst.FlowReturn.ERROR


if CAN_REGISTER_ELEMENT:
    GObject.type_register(EmbeddingTransform)
    __gstelementfactory__ = ("pyml_embedding", Gst.Rank.NONE, EmbeddingTransform)
else:
    GlobalLogger().warning(
        "The 'pyml_embedding' element will not be registered because required modules are missing."
    )
