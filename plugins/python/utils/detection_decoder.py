# detection_decoder.py
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

import numpy as np


def nms(boxes, scores, iou_threshold):
    """Greedy NMS. Returns indices of kept boxes."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_threshold)[0] + 1]
    return keep


def _decode_anchor_free(output, conf_threshold=0.25, iou_threshold=0.45):
    """Decode anchor-free detection output [B, 4+nc, anchors] into a list of dicts.

    Supports any model that outputs [batch, 4+num_classes, num_anchors] with
    centre-xywh box encoding (e.g. YOLO v5/v8/v11, RT-DETR export variants).
    Each result dict has keys: boxes [N,4] xyxy, labels [N] int, scores [N] float.
    """
    # [B, 4+nc, anchors] -> [B, anchors, 4+nc]
    pred = np.transpose(output, (0, 2, 1))
    results = []
    for b in range(pred.shape[0]):
        p = pred[b]
        class_scores = p[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(p)), class_ids]
        mask = confidences > conf_threshold
        p, confidences, class_ids = p[mask], confidences[mask], class_ids[mask]
        if len(p) == 0:
            results.append({
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "labels": np.array([], dtype=int),
                "scores": np.array([], dtype=np.float32),
            })
            continue
        cx, cy, w, h = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
        keep = nms(boxes, confidences, iou_threshold)
        results.append({
            "boxes": boxes[keep],
            "labels": class_ids[keep].astype(int),
            "scores": confidences[keep],
        })
    return results


_DECODERS = {
    "anchor_free": _decode_anchor_free,
}


def decode(output, fmt, conf_threshold=0.25, iou_threshold=0.45):
    """Decode raw detection output using the named format.

    fmt: one of the keys in _DECODERS (e.g. 'anchor_free').
    Returns a list of per-batch dicts with boxes/labels/scores.
    """
    decoder = _DECODERS.get(fmt)
    if decoder is None:
        raise ValueError(f"Unknown detection format '{fmt}'. Known: {list(_DECODERS)}")
    return decoder(output, conf_threshold, iou_threshold)
