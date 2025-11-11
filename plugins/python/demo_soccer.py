# DemoSoccer
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

from log.global_logger import GlobalLogger

CAN_REGISTER_ELEMENT = True
try:
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstBase", "1.0")
    gi.require_version("GstVideo", "1.0")
    gi.require_version("GstAnalytics", "1.0")
    gi.require_version("GLib", "2.0")
    from gi.repository import Gst, GObject, GstAnalytics, GLib  # noqa: E402
    from base_objectdetector import BaseObjectDetector

    import numpy as np
    import time
    import cv2
    import os
    from collections import defaultdict, deque
    from ultralytics import YOLO
    from engine.pytorch_engine import PyTorchEngine
    from engine.engine_factory import EngineFactory

    try:
        import torch
    except Exception:
        torch = None

    BOT_OK = BYTE_OK = True
    try:
        from ultralytics.trackers.bot_sort import BOTSORT
    except Exception:
        BOT_OK = False
    try:
        from ultralytics.trackers.byte_tracker import BYTETracker
    except Exception:
        BYTE_OK = False

    CFG_OK = True
    try:
        from ultralytics.cfg import get_cfg
    except Exception:
        CFG_OK = False

    from ultralytics.engine.results import Boxes

except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(
        f"The 'yolo_advanced' element will not be available. Error {e}"
    )

COCO_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
}


def eye3():
    return np.eye(3, dtype=np.float32)


def estimate_global_motion(
    prev_gray,
    gray,
    gmc_mode,
    gmc_scale,
    gft_max_corners,
    gft_quality,
    gft_min_dist,
    lk_win,
    lk_levels,
    ransac_thresh,
    frame_idx,
    verbose=False,
):
    if gmc_mode == "off":
        if verbose:
            print(f"[frame {frame_idx}] GMC OFF → I")
        return eye3()

    def down(img):
        if gmc_scale == 1.0:
            return img
        w = max(2, int(img.shape[1] * gmc_scale))
        h = max(2, int(img.shape[0] * gmc_scale))
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    pg, cg = down(prev_gray), down(gray)
    pts_prev = cv2.goodFeaturesToTrack(
        pg,
        maxCorners=gft_max_corners,
        qualityLevel=gft_quality,
        minDistance=gft_min_dist,
    )
    if pts_prev is None or len(pts_prev) < 6:
        if verbose:
            print(f"[frame {frame_idx}] GMC: insufficient corners → I")
        return eye3()

    pts_curr, st, _ = cv2.calcOpticalFlowPyrLK(
        pg,
        cg,
        pts_prev,
        None,
        winSize=(lk_win, lk_win),
        maxLevel=lk_levels,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if pts_curr is None or st is None:
        if verbose:
            print(f"[frame {frame_idx}] GMC: LK failed → I")
        return eye3()

    m = st.reshape(-1).astype(bool)
    if m.sum() < (4 if gmc_mode == "homography" else 3):
        if verbose:
            print(f"[frame {frame_idx}] GMC: not enough inliers → I")
        return eye3()

    src = pts_prev[m]
    dst = pts_curr[m]
    if gmc_scale != 1.0:
        s = 1.0 / gmc_scale
        src *= s
        dst *= s

    if gmc_mode == "homography":
        H, _ = cv2.findHomography(
            src,
            dst,
            cv2.RANSAC,
            ransacReprojThreshold=ransac_thresh,
            maxIters=1000,
        )
        H = H.astype(np.float32) if H is not None else eye3()
    else:
        A, _ = cv2.estimateAffine2D(
            src, dst, ransacReprojThreshold=ransac_thresh, maxIters=1000
        )
        H = np.vstack([A, [0, 0, 1]]).astype(np.float32) if A is not None else eye3()

    avg = float(np.mean(np.linalg.norm(dst - src, axis=1))) if len(src) > 0 else 0.0
    if verbose:
        print(
            f"[frame {frame_idx}] GMC {gmc_mode} inliers={int(m.sum())} avg_motion={avg:.2f}px"
        )
    return H


def warp_points(points_xy, M):
    if not points_xy:
        return []
    P = np.c_[
        np.array(points_xy, dtype=np.float32), np.ones((len(points_xy), 1), np.float32)
    ]
    Q = (M @ P.T).T
    Q = Q[:, :2] / np.clip(Q[:, 2:3], 1e-6, None)
    return [tuple(q) for q in Q]


def classwise_keep(result, person_thr, ball_thr):
    if (
        result is None
        or result.boxes is None
        or len(result.boxes) == 0
        or torch is None
    ):
        return np.zeros((0, 6), np.float32), np.zeros((0, 6), np.float32)

    b = result.boxes
    xyxy = b.xyxy.cpu().numpy()
    conf = (
        b.conf.cpu().numpy() if b.conf is not None else np.ones((len(b),), np.float32)
    )
    cls = (
        b.cls.cpu().numpy().astype(int)
        if b.cls is not None
        else np.zeros((len(b),), np.int32)
    )

    keep_p = (cls == 0) & (conf >= person_thr)
    keep_b = (cls == 32) & (conf >= ball_thr)

    dets_p = np.c_[xyxy[keep_p], conf[keep_p], cls[keep_p]].astype(np.float32)
    dets_b = np.c_[xyxy[keep_b], conf[keep_b], cls[keep_b]].astype(np.float32)
    return dets_p, dets_b


def dets_to_boxes(dets_xyxy_conf_cls, frame_shape):
    if dets_xyxy_conf_cls is None or dets_xyxy_conf_cls.size == 0:
        import torch as _torch

        data = _torch.zeros((0, 6), dtype=_torch.float32)
        return Boxes(data, frame_shape)
    import torch as _torch

    data = _torch.from_numpy(dets_xyxy_conf_cls).to(_torch.float32)
    return Boxes(data, frame_shape)


def clamp_imgsz_for_device(imgsz, device_str):
    if device_str in ("cpu", "auto"):
        return min(imgsz, 1280)
    return imgsz


def expand_roi(xyxy, scale, W, H, min_side=256, max_side=1920):
    x1, y1, x2, y2 = map(float, xyxy)
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    w, h = (x2 - x1), (y2 - y1)
    side = max(w, h) * float(scale)
    side = max(min_side, min(side, max_side))
    x1n = max(0, int(cx - side * 0.5))
    y1n = max(0, int(cy - side * 0.5))
    x2n = min(W - 1, int(cx + side * 0.5))
    y2n = min(H - 1, int(cy + side * 0.5))
    return x1n, y1n, x2n, y2n


def _normalize_tracker_args(args, kind="byte"):
    def has(a):
        return hasattr(args, a) and getattr(args, a) is not None

    def setif(a, v):
        setattr(args, a, v)

    def copy_if_missing(target, *sources, default=None):
        if not has(target):
            for s in sources:
                if has(s):
                    setif(target, getattr(args, s))
                    return
            if default is not None:
                setif(target, default)

    copy_if_missing("track_high_thresh", "track_thresh", default=0.5)
    copy_if_missing("track_thresh", "track_high_thresh", default=0.5)
    copy_if_missing("track_low_thresh", default=0.1)
    copy_if_missing("new_track_thresh", default=0.4)
    copy_if_missing("match_thresh", default=0.8)
    copy_if_missing("asso_thresh", "match_thresh", default=0.8)
    copy_if_missing("track_buffer", default=30)
    copy_if_missing("frame_rate", default=30)
    copy_if_missing("fuse_score", default=True)
    copy_if_missing("fuse_score_coef", default=1.0)
    copy_if_missing("mot20", default=False)

    if kind == "botsort":
        copy_if_missing("with_reid", default=True)
        copy_if_missing("proximity_thresh", default=0.5)
        copy_if_missing("appearance_thresh", default=0.25)
        if has("cmc_method") and not has("gmc_method"):
            setif("gmc_method", getattr(args, "cmc_method"))
        copy_if_missing("gmc_method", default="sparseOptFlow")


class ByteTrackWrapper:
    """Ultralytics BYTETracker; load params from YAML using get_cfg, accept Boxes."""

    def __init__(self, yaml_path, frame_rate):
        if not BYTE_OK:
            raise RuntimeError("Ultralytics BYTETracker not available")
        if not CFG_OK:
            raise RuntimeError(
                "Ultralytics get_cfg not available; update ultralytics package."
            )
        args = get_cfg(yaml_path)
        args.frame_rate = int(frame_rate)
        _normalize_tracker_args(args, kind="byte")
        self.tracker = BYTETracker(args, frame_rate=int(frame_rate))

    def update(self, boxes: Boxes, frame):
        return self.tracker.update(boxes, frame)


class BoTSORTWrapper:
    """Ultralytics BOTSORT; load params via get_cfg, accept Boxes, safe ReID encoder."""

    def __init__(self, yaml_path, frame_rate, enable_reid=True):
        if not BOT_OK:
            raise RuntimeError("Ultralytics BOTSORT not available")
        if not CFG_OK:
            raise RuntimeError(
                "Ultralytics get_cfg not available; update ultralytics package."
            )
        args = get_cfg(yaml_path)
        args.frame_rate = int(frame_rate)
        args.with_reid = bool(enable_reid)
        _normalize_tracker_args(args, kind="botsort")
        self.tracker = BOTSORT(args, frame_rate=int(frame_rate))
        self._install_safe_encoder()

    def _install_safe_encoder(self):
        import torch as _torch

        def _safe_hsv_hist(img_bgr, bboxes):
            feats = []
            if img_bgr is None or bboxes is None:
                return feats

            if hasattr(bboxes, "detach"):
                bb = bboxes.detach().cpu().numpy()
            elif isinstance(bboxes, np.ndarray):
                bb = bboxes
            else:
                try:
                    bb = np.asarray(bboxes, dtype=np.float32)
                except Exception:
                    bb = None

            H, W = img_bgr.shape[:2]

            def process_one(b):
                b = np.asarray(b, dtype=np.float32).reshape(-1)
                x1, y1, x2, y2 = map(float, b[:4])
                x1i = max(0, min(int(x1), W - 1))
                y1i = max(0, min(int(y1), H - 1))
                x2i = max(0, min(int(x2), W - 1))
                y2i = max(0, min(int(y2), H - 1))
                if x2i <= x1i or y2i <= y1i:
                    return _torch.zeros(512, dtype=_torch.float32)
                crop = img_bgr[y1i:y2i, x1i:x2i]
                if crop.size == 0:
                    return _torch.zeros(512, dtype=_torch.float32)
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist(
                    [hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256]
                ).flatten()
                norm = np.linalg.norm(hist) + 1e-6
                hist = (hist / norm).astype(np.float32)
                return _torch.from_numpy(hist)

            if bb is not None:
                for b in bb:
                    feats.append(process_one(b))
            else:
                for b in bboxes:
                    feats.append(process_one(b))
            return feats

        self.tracker.encoder = lambda img, tlbrs: _safe_hsv_hist(img, tlbrs)

    def update(self, boxes: Boxes, frame):
        return self.tracker.update(boxes, frame)


def _callable_or_attr(obj, name):
    v = getattr(obj, name, None)
    if v is None:
        return None
    return v() if callable(v) else v


def tlbr_of(tr):
    a = _callable_or_attr(tr, "tlbr")
    if a is not None:
        a = a.tolist() if hasattr(a, "tolist") else a
        if len(a) == 4:
            return a
    a = _callable_or_attr(tr, "tlwh")
    if a is not None:
        a = a.tolist() if hasattr(a, "tolist") else a
        if len(a) == 4:
            x, y, w, h = a
            return [x, y, x + w, y + h]
    if hasattr(tr, "bbox"):
        b = tr.bbox
        return b.tolist() if hasattr(b, "tolist") else list(b)
    return None


class BallState:
    def __init__(self):
        self.cx = None
        self.cy = None
        self.vx = 0.0
        self.vy = 0.0
        self.frame = -1

    def predict(self, frame_idx, decay=0.85):
        if self.cx is None:
            return None
        return (self.cx + decay * self.vx, self.cy + decay * self.vy)

    def update_from_xyxy(self, xyxy, frame_idx):
        x1, y1, x2, y2 = map(float, xyxy)
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        if self.cx is not None and self.frame >= 0:
            dt = max(1, frame_idx - self.frame)
            self.vx = (cx - self.cx) / dt
            self.vy = (cy - self.cy) / dt
        self.cx, self.cy, self.frame = cx, cy, frame_idx

    def update_from_center(self, cx, cy, frame_idx):
        if self.cx is not None and self.frame >= 0:
            dt = max(1, frame_idx - self.frame)
            self.vx = (cx - self.cx) / dt
            self.vy = (cy - self.cy) / dt
        self.cx, self.cy, self.frame = float(cx), float(cy), int(frame_idx)


def _aspect_round_penalty(w, h):
    ar = w / max(h, 1e-6)
    roundness = np.exp(-((ar - 1.0) ** 2) / 0.15)
    return 1.0 - float(roundness)


def _size_penalty(w, h, H, W):
    s = max(w, h)
    tgt = 0.03 * min(H, W)
    return float(np.clip(abs(s - tgt) / (tgt + 1e-6), 0.0, 2.0)) * 0.5


def select_best_ball(
    dets_b,
    frame_shape,
    ball_state,
    frame_idx,
    w_conf=1.0,
    w_dist=0.015,
    w_size=0.5,
    w_round=0.4,
):
    if dets_b is None or len(dets_b) == 0:
        return None
    H, W = frame_shape
    pred = ball_state.predict(frame_idx)
    scores = []
    for d in dets_b:
        x1, y1, x2, y2, conf, _ = d
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        w, h = (x2 - x1), (y2 - y1)
        s_conf = float(conf)
        if pred is None:
            dist_pen = 0.0
        else:
            px, py = pred
            dist = np.hypot(cx - px, cy - py)
            dist_pen = float(dist / (0.5 * (H + W)))
        size_pen = _size_penalty(w, h, H, W)
        round_pen = _aspect_round_penalty(w, h)
        s = (
            w_conf * s_conf
            - w_dist * dist_pen
            - w_size * size_pen
            - w_round * round_pen
        )
        scores.append(s)
    if not scores:
        return None
    return int(np.argmax(scores))


def nms_class(dets, iou_thr=0.5):
    if dets is None or len(dets) == 0:
        return dets
    boxes = dets[:, :4].copy()
    scores = dets[:, 4].copy()
    order = scores.argsort()[::-1]
    keep = []

    def iou(a, b):
        xx1 = np.maximum(a[0], b[0])
        yy1 = np.maximum(a[1], b[1])
        xx2 = np.minimum(a[2], b[2])
        yy2 = np.minimum(a[3], b[3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (area_a + area_b - inter + 1e-6)

    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = np.array([iou(boxes[i], boxes[j]) for j in order[1:]])
        remain = np.where(ious <= iou_thr)[0]
        order = order[remain + 1]
    return dets[keep]


def add_trail_point(seq: deque, x: int, y: int, k: int, densify=True, max_gap=5):
    if len(seq) > 0 and densify:
        _, _, k_prev = seq[-1]
        gap = k - k_prev
        if 1 < gap <= max_gap:
            x_prev, y_prev, _ = seq[-1]
            for t in range(1, gap):
                alpha = t / gap
                xi = int(round((1 - alpha) * x_prev + alpha * x))
                yi = int(round((1 - alpha) * y_prev + alpha * y))
                seq.append((xi, yi, k_prev + t))
    seq.append((int(x), int(y), int(k)))


def iou_xyxy(a, b):
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    xx1 = max(ax1, bx1)
    yy1 = max(ay1, by1)
    xx2 = min(ax2, bx2)
    yy2 = min(ay2, by2)
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    denom = area_a + area_b - inter + 1e-6
    return float(inter / denom)


def lerp(a, b, t):
    return a * (1.0 - t) + b * t


def gate_accept(
    center,
    cand_box,
    trail_seq,
    last_shown_box,
    recent_speed,
    frame_shape,
    args,
    pred=None,
    from_track=True,
):
    if center is None:
        return False

    H, W = frame_shape
    hard_cap = float(args.ball_max_jump_rel) * float(min(H, W))

    if len(trail_seq) == 0:
        dist_prev = 0.0
    else:
        x_prev, y_prev, _ = trail_seq[-1]
        dist_prev = float(np.hypot(center[0] - x_prev, center[1] - y_prev))

    if dist_prev > hard_cap:
        return False

    base_gate = max(
        float(args.ball_gate_min), float(args.ball_gate_rel) * float(min(H, W))
    )
    gate_px = base_gate if from_track else base_gate * 1.25
    pass_prev = (len(trail_seq) == 0) or (dist_prev <= gate_px)

    pred_ok = False
    if getattr(args, "ball_gate_use_pred", False) and pred is not None:
        d_pred = float(np.hypot(center[0] - pred[0], center[1] - pred[1]))
        pred_ok = d_pred <= gate_px * 1.25

    if not from_track:
        return pass_prev or pred_ok

    iou_ok = (last_shown_box is None) or (
        iou_xyxy(cand_box, last_shown_box) >= float(args.ball_min_iou)
    )

    speed_ok = True
    if recent_speed is not None and recent_speed > 0:
        speed_ok = dist_prev <= float(args.ball_speed_mult) * float(recent_speed + 1e-6)

    return (pass_prev and iou_ok and speed_ok) or pred_ok


def safe_int_pair(wx, wy, W, H):
    if wx is None or wy is None:
        return None
    if not (np.isfinite(wx) and np.isfinite(wy)):
        return None
    try:
        xi = int(round(float(wx)))
        yi = int(round(float(wy)))
    except Exception:
        return None
    if abs(xi) > 10 * W or abs(yi) > 10 * H:
        return None
    return xi, yi


class YoloAdvancedEngine(PyTorchEngine):
    def __init__(self, device=None, **kwargs):
        super().__init__(device=device)
        # Then set self.device_str = device if device else 'auto'
        self.device_str = device if device else "auto"
        self.det_model = None
        self.fb_model = None
        self.people_tracker = None
        self.ball_tracker = None
        self.ball_state = BallState()
        self.single_ball_trail = deque(maxlen=kwargs.get("trail", 200))
        self.cum_H_history = [eye3()]
        self.cum_H = eye3()
        self.prev_gray = None
        self.last_ball_xyxy = None
        self.last_shown_box = None
        self.recent_speed = None
        self.ema_cxcy = None
        self.coast_streak = 0
        self.det_reject_streak = 0
        self.dropped_by_gate = 0
        self.coast_used = 0
        self.frame_idx = 0
        self.frame_rate = kwargs.get("frame_rate", 30.0)
        # Set all params from kwargs
        self.device_str = kwargs.get("device", "auto")
        self.imgsz = kwargs.get("imgsz", 1280)
        self.conf = kwargs.get("conf", 0.25)
        self.iou = kwargs.get("iou", 0.45)
        self.classes = kwargs.get("classes", [0, 32])
        self.person_conf_keep = kwargs.get("person_conf_keep", 0.25)
        self.ball_conf_keep = kwargs.get("ball_conf_keep", 0.04)
        self.ball_mode = kwargs.get("ball_mode", True)
        self.hires_fallback = kwargs.get("hires_fallback", True)
        self.hires_imgsz = kwargs.get("hires_imgsz", 1536)
        self.fallback_every = kwargs.get("fallback_every", 6)
        self.fallback_tiles = kwargs.get("fallback_tiles", False)
        self.tile_size = kwargs.get("tile_size", 1280)
        self.tile_overlap = kwargs.get("tile_overlap", 256)
        self.fallback_budget_ms = kwargs.get("fallback_budget_ms", 300)
        self.ball_roi_boost = kwargs.get("ball_roi_boost", False)
        self.roi_scale = kwargs.get("roi_scale", 2.5)
        self.roi_min = kwargs.get("roi_min", 256)
        self.roi_max = kwargs.get("roi_max", 1920)
        self.tracker_people = kwargs.get("tracker_people", "botsort_people_reid.yaml")
        self.tracker_ball = kwargs.get("tracker_ball", "bytetrack_ball.yaml")
        self.people_reid = kwargs.get("people_reid", True)
        self.trail = kwargs.get("trail", 200)
        self.gmc = kwargs.get("gmc", "affine")
        self.gmc_scale = kwargs.get("gmc_scale", 0.5)
        self.gft_max_corners = kwargs.get("gft_max_corners", 400)
        self.gft_quality = kwargs.get("gft_quality", 0.01)
        self.gft_min_dist = kwargs.get("gft_min_dist", 8)
        self.lk_win = kwargs.get("lk_win", 21)
        self.lk_levels = kwargs.get("lk_levels", 3)
        self.ransac_thresh = kwargs.get("ransac_thresh", 3.0)
        self.ball_gate_rel = kwargs.get("ball_gate_rel", 0.06)
        self.ball_gate_min = kwargs.get("ball_gate_min", 12)
        self.ball_gate_use_pred = kwargs.get("ball_gate_use_pred", False)
        self.ball_min_iou = kwargs.get("ball_min_iou", 0.20)
        self.ball_max_jump_rel = kwargs.get("ball_max_jump_rel", 0.12)
        self.ball_speed_mult = kwargs.get("ball_speed_mult", 3.0)
        self.ball_smooth_ema = kwargs.get("ball_smooth_ema", 0.0)
        self.det_override_conf = kwargs.get("det_override_conf", 0.28)
        self.det_override_after = kwargs.get("det_override_after", 2)
        self.reacquire_frames = kwargs.get("reacquire_frames", 6)
        self.ball_coast = kwargs.get("ball_coast", False)
        self.coast_max = kwargs.get("coast_max", 6)
        self.coast_decay = kwargs.get("coast_decay", 0.90)
        self.verbose = kwargs.get("verbose", False)

    def do_load_model(self, model_name, **kwargs):
        try:
            # YOLO load unchanged...
            self.det_model = YOLO(f"{model_name}.pt")
            self.execute_with_stream(lambda: self.det_model.to(self.device))
            self.logger.info(
                f"YOLO primary model '{model_name}' loaded on {self.device}"
            )

            if self.hires_fallback:
                self.fb_model = YOLO(f"{model_name}.pt")
                self.execute_with_stream(lambda: self.fb_model.to(self.device))

            self.model = self.det_model  # Alias for base compat

            # Trackers with fallback
            if self.tracker_people:
                try:
                    self.people_tracker = BoTSORTWrapper(
                        self.tracker_people, self.frame_rate, self.people_reid
                    )
                    self.logger.info(
                        f"People tracker loaded from {self.tracker_people}"
                    )
                except Exception as te:
                    self.logger.warning(f"People tracker failed ({te}); disabling.")
                    self.people_tracker = None

            if self.tracker_ball:
                try:
                    self.ball_tracker = ByteTrackWrapper(
                        self.tracker_ball, self.frame_rate
                    )
                    self.logger.info(f"Ball tracker loaded from {self.tracker_ball}")
                except Exception as te:
                    self.logger.warning(f"Ball tracker failed ({te}); disabling.")
                    self.ball_tracker = None

            # ... kwargs update unchanged ...
            return self.tracker_people and self.tracker_ball

        except Exception as e:
            self.logger.error(f"Core model load failed: {e}")
            return False  # No raise—let base handle

    def do_forward(self, frames):
        is_batch = isinstance(frames, np.ndarray) and frames.ndim == 4
        if is_batch:
            frame_bgr = frames[0]  # Assume single for stateful; extend if needed
        else:
            frame_bgr = np.array(frames, copy=True)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        Hh, Ww = frame_bgr.shape[:2]

        # GMC
        if self.prev_gray is not None:
            H = estimate_global_motion(
                self.prev_gray,
                gray,
                self.gmc,
                self.gmc_scale,
                self.gft_max_corners,
                self.gft_quality,
                self.gft_min_dist,
                self.lk_win,
                self.lk_levels,
                self.ransac_thresh,
                self.frame_idx,
                self.verbose,
            )
            self.cum_H = H @ self.cum_H
        self.cum_H_history.append(self.cum_H.copy())
        self.prev_gray = gray

        # Detection
        det_res = self.execute_with_stream(
            lambda: self.det_model.predict(
                frame_bgr,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                classes=self.classes,
                device=self.device_str if self.device_str != "auto" else None,
                verbose=False,
            )[0]
        )

        dets_p, dets_b = classwise_keep(
            det_res, self.person_conf_keep, self.ball_conf_keep
        )

        dets_b = nms_class(dets_b, iou_thr=0.35)
        best_idx = select_best_ball(
            dets_b, frame_bgr.shape[:2], self.ball_state, self.frame_idx
        )
        if best_idx is not None:
            dets_b = dets_b[[best_idx]]
            self.ball_state.update_from_xyxy(dets_b[0, :4], self.frame_idx)
            self.last_ball_xyxy = dets_b[0, :4].astype(int).tolist()
        else:
            dets_b = dets_b[:0]

        # Fallback logic
        if (
            self.ball_mode
            and self.hires_fallback
            and dets_b.shape[0] == 0
            and self.frame_idx % max(1, self.fallback_every) == 0
        ):
            clamp_imgsz = clamp_imgsz_for_device(self.hires_imgsz, self.device_str)
            collected = []

            if (
                self.ball_roi_boost
                and self.last_ball_xyxy is not None
                and self.fb_model is not None
            ):
                x1r, y1r, x2r, y2r = expand_roi(
                    self.last_ball_xyxy,
                    self.roi_scale,
                    Ww,
                    Hh,
                    min_side=self.roi_min,
                    max_side=self.roi_max,
                )
                crop = frame_bgr[y1r:y2r, x1r:x2r]
                pred = self.execute_with_stream(
                    lambda: self.fb_model.predict(
                        crop,
                        imgsz=min(max(x2r - x1r, y2r - y1r), clamp_imgsz),
                        conf=max(self.ball_conf_keep, 0.02),
                        iou=max(self.iou, 0.50),
                        classes=[32],
                        device=self.device_str if self.device_str != "auto" else None,
                        verbose=False,
                    )[0]
                )
                if pred.boxes is not None and len(pred.boxes) > 0:
                    b = pred.boxes
                    xyxy = b.xyxy.cpu().numpy()
                    conf = (
                        b.conf.cpu().numpy()
                        if b.conf is not None
                        else np.ones((len(b),), np.float32)
                    )
                    cls = np.full((len(b),), 32, dtype=np.float32)
                    xyxy[:, [0, 2]] += x1r
                    xyxy[:, [1, 3]] += y1r
                    collected.append(np.c_[xyxy, conf, cls])

            if self.fb_model is not None and not collected:
                pred = self.execute_with_stream(
                    lambda: self.fb_model.predict(
                        frame_bgr,
                        imgsz=clamp_imgsz,
                        conf=max(self.ball_conf_keep, 0.02),
                        iou=max(self.iou, 0.50),
                        classes=[32],
                        device=self.device_str if self.device_str != "auto" else None,
                        verbose=False,
                    )[0]
                )
                if pred.boxes is not None and len(pred.boxes) > 0:
                    b = pred.boxes
                    xyxy = b.xyxy.cpu().numpy()
                    conf = (
                        b.conf.cpu().numpy()
                        if b.conf is not None
                        else np.ones((len(b),), np.float32)
                    )
                    cls = np.full((len(b),), 32, dtype=np.float32)
                    collected.append(np.c_[xyxy, conf, cls])

            if collected:
                dets_b = np.vstack(collected).astype(np.float32)
                dets_b = nms_class(dets_b, iou_thr=0.35)
                best_idx = select_best_ball(
                    dets_b, frame_bgr.shape[:2], self.ball_state, self.frame_idx
                )
                if best_idx is not None:
                    dets_b = dets_b[[best_idx]]
                    self.ball_state.update_from_xyxy(dets_b[0, :4], self.frame_idx)
                    self.last_ball_xyxy = dets_b[0, :4].astype(int).tolist()
                else:
                    dets_b = dets_b[:0]

        frame_shape = frame_bgr.shape[:2]
        boxes_p = dets_to_boxes(dets_p, frame_shape)
        boxes_b = dets_to_boxes(dets_b, frame_shape)
        tracks_p = (
            self.people_tracker.update(boxes_p, frame_bgr)
            if self.people_tracker
            else []
        )
        tracks_b = (
            self.ball_tracker.update(boxes_b, frame_bgr) if self.ball_tracker else []
        )

        # Ball candidate selection and gating
        cand_box = None
        ball_center_candidate = None
        cand_conf = None
        from_track = False

        if len(tracks_b) >= 1:
            tr = tracks_b[0]
            tb = tlbr_of(tr)
            if tb is not None:
                x1, y1, x2, y2 = map(int, tb)
                cand_box = [x1, y1, x2, y2]
                ball_center_candidate = ((x1 + x2) // 2, (y1 + y2) // 2)
                from_track = True
                cand_conf = None
        elif dets_b.shape[0] == 1:
            x1, y1, x2, y2 = map(int, dets_b[0, :4])
            cand_box = [x1, y1, x2, y2]
            ball_center_candidate = ((x1 + x2) // 2, (y1 + y2) // 2)
            cand_conf = float(dets_b[0, 4])
            from_track = False

        pred_pos = (
            self.ball_state.predict(self.frame_idx) if self.ball_gate_use_pred else None
        )
        accept = gate_accept(
            ball_center_candidate,
            cand_box,
            self.single_ball_trail,
            self.last_shown_box,
            self.recent_speed,
            frame_bgr.shape[:2],
            self,  # Use self as args
            pred=pred_pos,
            from_track=from_track,
        )

        # Det override logic
        if (not accept) and (ball_center_candidate is not None) and (not from_track):
            self.det_reject_streak += 1
            gap_frames = (
                (self.frame_idx - self.single_ball_trail[-1][2])
                if len(self.single_ball_trail)
                else 9999
            )
            Hmin = float(min(Hh, Ww))
            base_gate = max(float(self.ball_gate_min), float(self.ball_gate_rel) * Hmin)

            if cand_conf is not None and cand_conf >= float(self.det_override_conf):
                x_prev, y_prev = (
                    (self.single_ball_trail[-1][0], self.single_ball_trail[-1][1])
                    if self.single_ball_trail
                    else (ball_center_candidate[0], ball_center_candidate[1])
                )
                dist_prev = float(
                    np.hypot(
                        ball_center_candidate[0] - x_prev,
                        ball_center_candidate[1] - y_prev,
                    )
                )
                if (
                    (self.det_reject_streak >= int(self.det_override_after))
                    or (gap_frames >= int(self.reacquire_frames))
                    or (dist_prev <= 2.5 * base_gate)
                ):
                    accept = True  # force accept the detection
        else:
            self.det_reject_streak = 0

        did_update_trail = False

        if accept and ball_center_candidate is not None:
            cx_raw, cy_raw = ball_center_candidate
            alpha = float(self.ball_smooth_ema)
            if 0.0 < alpha <= 1.0:
                if self.ema_cxcy is None:
                    self.ema_cxcy = (float(cx_raw), float(cy_raw))
                else:
                    self.ema_cxcy = (
                        lerp(self.ema_cxcy[0], float(cx_raw), alpha),
                        lerp(self.ema_cxcy[1], float(cy_raw), alpha),
                    )
                cx, cy = int(round(self.ema_cxcy[0])), int(round(self.ema_cxcy[1]))
            else:
                cx, cy = cx_raw, cy_raw

            if from_track:
                x1, y1, x2, y2 = cand_box
                # No drawing here, as it's engine

            # Update trail
            if (
                len(self.single_ball_trail) >= 1
                and self.single_ball_trail[-1][2] <= self.frame_idx - 2
            ):
                x_prev, y_prev, k_prev = self.single_ball_trail[-1]
                if self.frame_idx - k_prev == 2:
                    add_trail_point(
                        self.single_ball_trail,
                        (x_prev + cx) // 2,
                        (y_prev + cy) // 2,
                        k_prev + 1,
                        densify=False,
                    )

            add_trail_point(
                self.single_ball_trail, cx, cy, self.frame_idx, densify=True, max_gap=5
            )
            did_update_trail = True

            if from_track:
                self.ball_state.update_from_xyxy(cand_box, self.frame_idx)
            else:
                if cand_box is not None:
                    self.ball_state.update_from_center(cx, cy, self.frame_idx)

            self.last_shown_box = (
                cand_box if cand_box is not None else self.last_shown_box
            )

            if len(self.single_ball_trail) >= 2:
                x0, y0, _ = self.single_ball_trail[-2]
                step = float(np.hypot(cx - x0, cy - y0))
                self.recent_speed = (
                    step
                    if self.recent_speed is None
                    else 0.8 * self.recent_speed + 0.2 * step
                )

            self.coast_streak = 0
            self.det_reject_streak = 0

        else:
            if ball_center_candidate is not None:
                self.dropped_by_gate += 1

            if (
                self.ball_coast
                and len(self.single_ball_trail) > 0
                and self.coast_streak < int(self.coast_max)
            ):
                pred = self.ball_state.predict(
                    self.frame_idx, decay=float(self.coast_decay)
                )
                if pred is not None and np.all(np.isfinite(pred)):
                    px, py = int(round(pred[0])), int(round(pred[1]))
                    x_prev, y_prev, _ = self.single_ball_trail[-1]
                    hard_cap = float(self.ball_max_jump_rel) * float(min(Hh, Ww))
                    if (
                        0 <= px < Ww
                        and 0 <= py < Hh
                        and float(np.hypot(px - x_prev, py - y_prev)) <= 1.25 * hard_cap
                    ):
                        add_trail_point(
                            self.single_ball_trail,
                            px,
                            py,
                            self.frame_idx,
                            densify=False,
                        )
                        self.ball_state.update_from_center(px, py, self.frame_idx)
                        if len(self.single_ball_trail) >= 2:
                            step = float(np.hypot(px - x_prev, py - y_prev))
                            self.recent_speed = (
                                step
                                if self.recent_speed is None
                                else 0.8 * self.recent_speed + 0.2 * step
                            )
                        self.coast_streak += 1
                        self.coast_used += 1
                        did_update_trail = True
            else:
                self.coast_streak = 0

        self.frame_idx += 1

        # Return result for decode
        class AdvancedResult:
            def __init__(self, tracks_p, tracks_b, ball_trail, boxes):
                self.tracks_p = tracks_p
                self.tracks_b = tracks_b
                self.ball_trail = list(ball_trail)
                self.boxes = boxes

        return AdvancedResult(tracks_p, tracks_b, self.single_ball_trail, det_res.boxes)


class DemoSoccer(BaseObjectDetector):
    """
    GStreamer element for advanced YOLO inference focused on person and ball tracking with fallback and gating.
    """

    __gstmetadata__ = (
        "YOLOBall",
        "Transform",
        "Advanced YOLO for person/ball detection, dual tracking, and ball trail stabilization",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    def __init__(self):
        super().__init__()
        self.mgr.engine_name = "demo_engine"
        EngineFactory.register(self.engine_name, YoloAdvancedEngine)

        # Resolve YAMLs
        script_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_base = os.path.join(script_dir, "data", "soccer")
        self.logger.info(f"YAML base resolved to: {yaml_base}")
        self.__tracker_people = os.path.join(yaml_base, "botsort_people_reid.yaml")
        self.__tracker_ball = os.path.join(yaml_base, "bytetrack_ball.yaml")

        # Check if YAML files exist
        if not os.path.exists(self.__tracker_people):
            raise FileNotFoundError(
                f"People tracker YAML file not found: {self.__tracker_people}"
            )
        if not os.path.exists(self.__tracker_ball):
            raise FileNotFoundError(
                f"Ball tracker YAML file not found: {self.__tracker_ball}"
            )

        # Defaults (all params)
        self.__model = "yolo11x"
        self.__device = "auto"
        self.__imgsz = 1280
        self.__conf = 0.25
        self.__iou = 0.45
        self.__classes = [0, 32]
        self.__person_conf_keep = 0.25
        self.__ball_conf_keep = 0.04
        self.__ball_mode = True
        self.__hires_fallback = True
        self.__hires_imgsz = 1536
        self.__fallback_every = 6
        self.__fallback_tiles = False
        self.__tile_size = 1280
        self.__tile_overlap = 256
        self.__fallback_budget_ms = 300
        self.__ball_roi_boost = False
        self.__roi_scale = 2.5
        self.__roi_min = 256
        self.__roi_max = 1920
        self.__people_reid = True
        self.__trail = 200
        self.__gmc = "affine"
        self.__gmc_scale = 0.5
        self.__gft_max_corners = 400
        self.__gft_quality = 0.01
        self.__gft_min_dist = 8
        self.__lk_win = 21
        self.__lk_levels = 3
        self.__ransac_thresh = 3.0
        self.__ball_gate_rel = 0.06
        self.__ball_gate_min = 12
        self.__ball_gate_use_pred = False
        self.__ball_min_iou = 0.20
        self.__ball_max_jump_rel = 0.12
        self.__ball_speed_mult = 3.0
        self.__ball_smooth_ema = 0.0
        self.__det_override_conf = 0.28
        self.__det_override_after = 2
        self.__reacquire_frames = 6
        self.__ball_coast = False
        self.__coast_max = 6
        self.__coast_decay = 0.90
        self.__verbose = False
        self.__frame_rate = 30.0

    # make engine_name read only
    @GObject.Property(type=str)
    def engine_name(self):
        """Machine Learning Engine (read-only in this class)."""
        return self.mgr.engine_name

    @engine_name.setter
    def engine_name(self, value):
        raise ValueError(
            "The 'engine_name' property cannot be set in this derived class."
        )

    # Properties - all of them
    @GObject.Property(type=str, default="yolo11x")
    def model(self):
        return self.__model

    @model.setter
    def model(self, value):
        self.__model = value
        if self.engine:
            self.engine.do_load_model(value)

    @GObject.Property(type=str, default="auto")
    def device(self):
        return self.__device

    @device.setter
    def device(self, value):
        self.__device = value
        if self.engine:
            self.engine.device_str = value

    @GObject.Property(type=int, default=1280)
    def imgsz(self):
        return self.__imgsz

    @imgsz.setter
    def imgsz(self, value):
        self.__imgsz = value
        if self.engine:
            self.engine.imgsz = value

    @GObject.Property(type=float, default=0.25)
    def conf(self):
        return self.__conf

    @conf.setter
    def conf(self, value):
        self.__conf = value
        if self.engine:
            self.engine.conf = value

    @GObject.Property(type=float, default=0.45)
    def iou(self):
        return self.__iou

    @iou.setter
    def iou(self, value):
        self.__iou = value
        if self.engine:
            self.engine.iou = value

    @GObject.Property(type=str, default="0 32")
    def classes(self):
        return " ".join(map(str, self.__classes))

    @classes.setter
    def classes(self, value):
        self.__classes = [int(c.strip()) for c in value.split()]
        if self.engine:
            self.engine.classes = self.__classes

    @GObject.Property(type=float, default=0.25)
    def person_conf_keep(self):
        return self.__person_conf_keep

    @person_conf_keep.setter
    def person_conf_keep(self, value):
        self.__person_conf_keep = value
        if self.engine:
            self.engine.person_conf_keep = value

    @GObject.Property(type=float, default=0.04)
    def ball_conf_keep(self):
        return self.__ball_conf_keep

    @ball_conf_keep.setter
    def ball_conf_keep(self, value):
        self.__ball_conf_keep = value
        if self.engine:
            self.engine.ball_conf_keep = value

    @GObject.Property(type=bool, default=True)
    def ball_mode(self):
        return self.__ball_mode

    @ball_mode.setter
    def ball_mode(self, value):
        self.__ball_mode = value
        if self.engine:
            self.engine.ball_mode = value

    @GObject.Property(type=bool, default=True)
    def hires_fallback(self):
        return self.__hires_fallback

    @hires_fallback.setter
    def hires_fallback(self, value):
        self.__hires_fallback = value
        if self.engine:
            self.engine.hires_fallback = value

    @GObject.Property(type=int, default=1536)
    def hires_imgsz(self):
        return self.__hires_imgsz

    @hires_imgsz.setter
    def hires_imgsz(self, value):
        self.__hires_imgsz = value
        if self.engine:
            self.engine.hires_imgsz = value

    @GObject.Property(type=int, default=6)
    def fallback_every(self):
        return self.__fallback_every

    @fallback_every.setter
    def fallback_every(self, value):
        self.__fallback_every = value
        if self.engine:
            self.engine.fallback_every = value

    @GObject.Property(type=bool, default=False)
    def fallback_tiles(self):
        return self.__fallback_tiles

    @fallback_tiles.setter
    def fallback_tiles(self, value):
        self.__fallback_tiles = value
        if self.engine:
            self.engine.fallback_tiles = value

    @GObject.Property(type=int, default=1280)
    def tile_size(self):
        return self.__tile_size

    @tile_size.setter
    def tile_size(self, value):
        self.__tile_size = value
        if self.engine:
            self.engine.tile_size = value

    @GObject.Property(type=int, default=256)
    def tile_overlap(self):
        return self.__tile_overlap

    @tile_overlap.setter
    def tile_overlap(self, value):
        self.__tile_overlap = value
        if self.engine:
            self.engine.tile_overlap = value

    @GObject.Property(type=int, default=300)
    def fallback_budget_ms(self):
        return self.__fallback_budget_ms

    @fallback_budget_ms.setter
    def fallback_budget_ms(self, value):
        self.__fallback_budget_ms = value
        if self.engine:
            self.engine.fallback_budget_ms = value

    @GObject.Property(type=bool, default=False)
    def ball_roi_boost(self):
        return self.__ball_roi_boost

    @ball_roi_boost.setter
    def ball_roi_boost(self, value):
        self.__ball_roi_boost = value
        if self.engine:
            self.engine.ball_roi_boost = value

    @GObject.Property(type=float, default=2.5)
    def roi_scale(self):
        return self.__roi_scale

    @roi_scale.setter
    def roi_scale(self, value):
        self.__roi_scale = value
        if self.engine:
            self.engine.roi_scale = value

    @GObject.Property(type=int, default=256)
    def roi_min(self):
        return self.__roi_min

    @roi_min.setter
    def roi_min(self, value):
        self.__roi_min = value
        if self.engine:
            self.engine.roi_min = value

    @GObject.Property(type=int, default=1920)
    def roi_max(self):
        return self.__roi_max

    @roi_max.setter
    def roi_max(self, value):
        self.__roi_max = value
        if self.engine:
            self.engine.roi_max = value

    @GObject.Property(type=str, default="botsort_people_reid.yaml")
    def tracker_people(self):
        return self.__tracker_people

    @tracker_people.setter
    def tracker_people(self, value):
        self.__tracker_people = value
        if self.engine:
            self.engine.tracker_people = value
            self.engine.people_tracker = BoTSORTWrapper(
                value, self.engine.frame_rate, self.people_reid
            )

    @GObject.Property(type=str, default="bytetrack_ball.yaml")
    def tracker_ball(self):
        return self.__tracker_ball

    @tracker_ball.setter
    def tracker_ball(self, value):
        self.__tracker_ball = value
        if self.engine:
            self.engine.tracker_ball = value
            self.engine.ball_tracker = ByteTrackWrapper(value, self.engine.frame_rate)

    @GObject.Property(type=bool, default=True)
    def people_reid(self):
        return self.__people_reid

    @people_reid.setter
    def people_reid(self, value):
        self.__people_reid = value
        if self.engine:
            self.engine.people_reid = value
            # Reinit tracker if needed

    @GObject.Property(type=int, default=200)
    def trail(self):
        return self.__trail

    @trail.setter
    def trail(self, value):
        self.__trail = value
        if self.engine:
            self.engine.trail = value
            self.engine.single_ball_trail = deque(maxlen=value)

    @GObject.Property(type=str, default="affine")
    def gmc(self):
        return self.__gmc

    @gmc.setter
    def gmc(self, value):
        self.__gmc = value
        if self.engine:
            self.engine.gmc = value

    @GObject.Property(type=float, default=0.5)
    def gmc_scale(self):
        return self.__gmc_scale

    @gmc_scale.setter
    def gmc_scale(self, value):
        self.__gmc_scale = value
        if self.engine:
            self.engine.gmc_scale = value

    @GObject.Property(type=int, default=400)
    def gft_max_corners(self):
        return self.__gft_max_corners

    @gft_max_corners.setter
    def gft_max_corners(self, value):
        self.__gft_max_corners = value
        if self.engine:
            self.engine.gft_max_corners = value

    @GObject.Property(type=float, default=0.01)
    def gft_quality(self):
        return self.__gft_quality

    @gft_quality.setter
    def gft_quality(self, value):
        self.__gft_quality = value
        if self.engine:
            self.engine.gft_quality = value

    @GObject.Property(type=int, default=8)
    def gft_min_dist(self):
        return self.__gft_min_dist

    @gft_min_dist.setter
    def gft_min_dist(self, value):
        self.__gft_min_dist = value
        if self.engine:
            self.engine.gft_min_dist = value

    @GObject.Property(type=int, default=21)
    def lk_win(self):
        return self.__lk_win

    @lk_win.setter
    def lk_win(self, value):
        self.__lk_win = value
        if self.engine:
            self.engine.lk_win = value

    @GObject.Property(type=int, default=3)
    def lk_levels(self):
        return self.__lk_levels

    @lk_levels.setter
    def lk_levels(self, value):
        self.__lk_levels = value
        if self.engine:
            self.engine.lk_levels = value

    @GObject.Property(type=float, default=3.0)
    def ransac_thresh(self):
        return self.__ransac_thresh

    @ransac_thresh.setter
    def ransac_thresh(self, value):
        self.__ransac_thresh = value
        if self.engine:
            self.engine.ransac_thresh = value

    @GObject.Property(type=float, default=0.06)
    def ball_gate_rel(self):
        return self.__ball_gate_rel

    @ball_gate_rel.setter
    def ball_gate_rel(self, value):
        self.__ball_gate_rel = value
        if self.engine:
            self.engine.ball_gate_rel = value

    @GObject.Property(type=int, default=12)
    def ball_gate_min(self):
        return self.__ball_gate_min

    @ball_gate_min.setter
    def ball_gate_min(self, value):
        self.__ball_gate_min = value
        if self.engine:
            self.engine.ball_gate_min = value

    @GObject.Property(type=bool, default=False)
    def ball_gate_use_pred(self):
        return self.__ball_gate_use_pred

    @ball_gate_use_pred.setter
    def ball_gate_use_pred(self, value):
        self.__ball_gate_use_pred = value
        if self.engine:
            self.engine.ball_gate_use_pred = value

    @GObject.Property(type=float, default=0.20)
    def ball_min_iou(self):
        return self.__ball_min_iou

    @ball_min_iou.setter
    def ball_min_iou(self, value):
        self.__ball_min_iou = value
        if self.engine:
            self.engine.ball_min_iou = value

    @GObject.Property(type=float, default=0.12)
    def ball_max_jump_rel(self):
        return self.__ball_max_jump_rel

    @ball_max_jump_rel.setter
    def ball_max_jump_rel(self, value):
        self.__ball_max_jump_rel = value
        if self.engine:
            self.engine.ball_max_jump_rel = value

    @GObject.Property(type=float, default=3.0)
    def ball_speed_mult(self):
        return self.__ball_speed_mult

    @ball_speed_mult.setter
    def ball_speed_mult(self, value):
        self.__ball_speed_mult = value
        if self.engine:
            self.engine.ball_speed_mult = value

    @GObject.Property(type=float, default=0.0)
    def ball_smooth_ema(self):
        return self.__ball_smooth_ema

    @ball_smooth_ema.setter
    def ball_smooth_ema(self, value):
        self.__ball_smooth_ema = value
        if self.engine:
            self.engine.ball_smooth_ema = value

    @GObject.Property(type=float, default=0.28)
    def det_override_conf(self):
        return self.__det_override_conf

    @det_override_conf.setter
    def det_override_conf(self, value):
        self.__det_override_conf = value
        if self.engine:
            self.engine.det_override_conf = value

    @GObject.Property(type=int, default=2)
    def det_override_after(self):
        return self.__det_override_after

    @det_override_after.setter
    def det_override_after(self, value):
        self.__det_override_after = value
        if self.engine:
            self.engine.det_override_after = value

    @GObject.Property(type=int, default=6)
    def reacquire_frames(self):
        return self.__reacquire_frames

    @reacquire_frames.setter
    def reacquire_frames(self, value):
        self.__reacquire_frames = value
        if self.engine:
            self.engine.reacquire_frames = value

    @GObject.Property(type=bool, default=False)
    def ball_coast(self):
        return self.__ball_coast

    @ball_coast.setter
    def ball_coast(self, value):
        self.__ball_coast = value
        if self.engine:
            self.engine.ball_coast = value

    @GObject.Property(type=int, default=6)
    def coast_max(self):
        return self.__coast_max

    @coast_max.setter
    def coast_max(self, value):
        self.__coast_max = value
        if self.engine:
            self.engine.coast_max = value

    @GObject.Property(type=float, default=0.90)
    def coast_decay(self):
        return self.__coast_decay

    @coast_decay.setter
    def coast_decay(self, value):
        self.__coast_decay = value
        if self.engine:
            self.engine.coast_decay = value

    @GObject.Property(type=bool, default=False)
    def verbose(self):
        return self.__verbose

    @verbose.setter
    def verbose(self, value):
        self.__verbose = value
        if self.engine:
            self.engine.verbose = value

    @GObject.Property(type=float, default=30.0)
    def frame_rate(self):
        return self.__frame_rate

    @frame_rate.setter
    def frame_rate(self, value):
        self.__frame_rate = value
        if self.engine:
            self.engine.frame_rate = value

    def set_model(self):
        """Override: Create engine first, then load with all properties as kwargs."""
        if self.engine is None:
            self.initialize_engine()
        if self.engine is None:
            self.logger.error("Cannot load model: engine not initialized")
            return False
        # Sync all properties to kwargs for engine load
        kwargs = {
            "classes": self.classes,
            "imgsz": self.imgsz,
            "conf": self.conf,
            "iou": self.iou,
            "person_conf_keep": self.person_conf_keep,
            "ball_conf_keep": self.ball_conf_keep,
            "ball_mode": self.ball_mode,
            "hires_fallback": self.hires_fallback,
            "hires_imgsz": self.hires_imgsz,
            "fallback_every": self.fallback_every,
            "fallback_tiles": self.fallback_tiles,
            "tile_size": self.tile_size,
            "tile_overlap": self.tile_overlap,
            "fallback_budget_ms": self.fallback_budget_ms,
            "ball_roi_boost": self.ball_roi_boost,
            "roi_scale": self.roi_scale,
            "roi_min": self.roi_min,
            "roi_max": self.roi_max,
            "tracker_people": self.tracker_people,
            "tracker_ball": self.tracker_ball,
            "people_reid": self.people_reid,
            "trail": self.trail,
            "gmc": self.gmc,
            "gmc_scale": self.gmc_scale,
            "gft_max_corners": self.gft_max_corners,
            "gft_quality": self.gft_quality,
            "gft_min_dist": self.gft_min_dist,
            "lk_win": self.lk_win,
            "lk_levels": self.lk_levels,
            "ransac_thresh": self.ransac_thresh,
            "ball_gate_rel": self.ball_gate_rel,
            "ball_gate_min": self.ball_gate_min,
            "ball_gate_use_pred": self.ball_gate_use_pred,
            "ball_min_iou": self.ball_min_iou,
            "ball_max_jump_rel": self.ball_max_jump_rel,
            "ball_speed_mult": self.ball_speed_mult,
            "ball_smooth_ema": self.ball_smooth_ema,
            "det_override_conf": self.det_override_conf,
            "det_override_after": self.det_override_after,
            "reacquire_frames": self.reacquire_frames,
            "ball_coast": self.ball_coast,
            "coast_max": self.coast_max,
            "coast_decay": self.coast_decay,
            "verbose": self.verbose,
            "frame_rate": self.frame_rate,
        }
        # Call engine's do_load_model with current model name + synced kwargs
        return self.engine.do_load_model(self.model, **kwargs)

    def do_decode(self, buf, result, stream_idx=0):
        self.logger.debug(
            f"Decoding advanced YOLO result for buffer {hex(id(buf))}, stream {stream_idx}: {result}"
        )
        tracks_p = result.tracks_p
        tracks_b = result.tracks_b
        ball_trail = result.ball_trail
        boxes = result.boxes

        meta = GstAnalytics.buffer_add_analytics_relation_meta(buf)
        if not meta:
            self.logger.error(
                f"Stream {stream_idx} - Failed to add analytics relation metadata"
            )
            return

        self.logger.debug(
            f"Stream {stream_idx} - Attaching metadata for {len(tracks_p) + len(tracks_b)} tracks"
        )

        # Person tracks (unchanged)
        for tr in tracks_p:
            box = tlbr_of(tr)
            if not box:
                continue
            x1, y1, x2, y2 = box
            score = 1.0  # Track confidence
            track_id = getattr(tr, "track_id", 0)
            label_num = 0  # person
            class_name = COCO_CLASSES.get(label_num, f"unknown_{label_num}")
            qk_string = f"stream_{stream_idx}_person_id_{track_id}"
            qk = GLib.quark_from_string(qk_string)
            ret, od_mtd = meta.add_od_mtd(
                qk,
                x1,
                y1,
                x2 - x1,
                y2 - y1,
                score,
            )
            if not ret:
                self.logger.error(
                    f"Stream {stream_idx} - Failed to add person detection metadata"
                )
                continue
            self.logger.debug(
                f"Stream {stream_idx} - Added person od_mtd: id={track_id}, x1={x1}, y1={y1}, w={x2-x1}, h={y2-y1}, score={score}"
            )

            ret, tracking_mtd = meta.add_tracking_mtd(
                track_id, Gst.util_get_timestamp()
            )
            if not ret:
                self.logger.error(
                    f"Stream {stream_idx} - Failed to add person tracking metadata"
                )
                continue
            ret = GstAnalytics.RelationMeta.set_relation(
                meta, GstAnalytics.RelTypes.RELATE_TO, od_mtd.id, tracking_mtd.id
            )
            if not ret:
                self.logger.error(
                    f"Stream {stream_idx} - Failed to relate person od and tracking metadata"
                )
            else:
                self.logger.debug(
                    f"Stream {stream_idx} - Linked person od_mtd {od_mtd.id} to tracking_mtd {tracking_mtd.id}"
                )

        # Ball tracks (unchanged)
        for i, tr in enumerate(tracks_b):
            box = tlbr_of(tr)
            if not box:
                continue
            x1, y1, x2, y2 = box
            score = 1.0
            track_id = getattr(tr, "track_id", 0)
            label_num = 32  # sports ball
            class_name = COCO_CLASSES.get(label_num, f"unknown_{label_num}")
            qk_string = f"stream_{stream_idx}_ball_id_{track_id}"
            qk = GLib.quark_from_string(qk_string)
            ret, od_mtd = meta.add_od_mtd(
                qk,
                x1,
                y1,
                x2 - x1,
                y2 - y1,
                score,
            )
            if not ret:
                self.logger.error(
                    f"Stream {stream_idx} - Failed to add ball detection metadata"
                )
                continue
            self.logger.debug(
                f"Stream {stream_idx} - Added ball od_mtd: id={track_id}, x1={x1}, y1={y1}, w={x2-x1}, h={y2-y1}, score={score}"
            )

            ret, tracking_mtd = meta.add_tracking_mtd(
                track_id, Gst.util_get_timestamp()
            )
            if not ret:
                self.logger.error(
                    f"Stream {stream_idx} - Failed to add ball tracking metadata"
                )
                continue
            ret = GstAnalytics.RelationMeta.set_relation(
                meta, GstAnalytics.RelTypes.RELATE_TO, od_mtd.id, tracking_mtd.id
            )
            if not ret:
                self.logger.error(
                    f"Stream {stream_idx} - Failed to relate ball od and tracking metadata"
                )
            else:
                self.logger.debug(
                    f"Stream {stream_idx} - Linked ball od_mtd {od_mtd.id} to tracking_mtd {tracking_mtd.id}"
                )

        # Ball trail - attach as custom GstStructure meta (fixed API, uncommented)
        if ball_trail:
            structure = Gst.Structure.new_empty("ball-trail")
            trail_data = [(int(x), int(y), int(k)) for x, y, k in ball_trail]
            structure.set_value("points", trail_data)
            structure.set_value("length", len(ball_trail))

            # Correct API: Use Gst.Buffer.add_meta with GstMeta for structure
            try:
                # Get generic meta API (GstMeta for any custom data)
                meta_api = Gst.Meta.get_api(
                    Gst.StructureMeta
                )  # Or Gst.Meta.api_type_get_tag(Gst.StructureMeta) if available
                if meta_api:
                    # Create custom meta with structure
                    meta = Gst.Meta.new(buf, meta_api, structure)
                    if meta:
                        Gst.Buffer.add_meta(buf, meta_api, meta)
                        self.logger.debug(
                            f"Stream {stream_idx} - Added ball trail meta with {len(ball_trail)} points"
                        )
                    else:
                        raise ValueError("Failed to create meta")
                else:
                    raise AttributeError("Meta API not available")
            except (AttributeError, ValueError, TypeError) as e:
                self.logger.warning(
                    f"Ball trail meta attachment failed ({e}); logging instead"
                )
                self.logger.info(
                    f"Ball trail for stream {stream_idx}: {trail_data[:5]}... (length {len(ball_trail)})"
                )

        # Fallback to original boxes if needed (unchanged)
        if boxes is not None and len(boxes) > 0:
            # Add any non-person/ball or raw detections if desired
            pass  # Optional

        attached_meta = GstAnalytics.buffer_get_analytics_relation_meta(buf)
        if attached_meta:
            count = GstAnalytics.relation_get_length(attached_meta)
            self.logger.info(
                f"Stream {stream_idx} - Advanced metadata attached to buffer {hex(id(buf))}: {count} relations, ball trail: {len(ball_trail)}"
            )
        else:
            self.logger.error(
                f"Stream {stream_idx} - Metadata not attached to buffer after adding"
            )


if CAN_REGISTER_ELEMENT:
    GObject.type_register(DemoSoccer)
    __gstelementfactory__ = ("demo_soccer", Gst.Rank.NONE, DemoSoccer)
else:
    GlobalLogger().warning(
        "The 'demo_soccer' element will not be registered because required modules are missing."
    )
