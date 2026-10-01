"""YOLO-based cat detector."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

# COCO class id for cat
CAT_CLASS_ID = 15


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    def clip(self, width: int, height: int) -> Detection:
        return Detection(
            x1=max(0, min(self.x1, width - 1)),
            y1=max(0, min(self.y1, height - 1)),
            x2=max(0, min(self.x2, width)),
            y2=max(0, min(self.y2, height)),
            confidence=self.confidence,
        )


class CatDetector:
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf: float = 0.45,
        device: str | None = None,
    ) -> None:
        self.model = YOLO(model_name)
        self.conf = conf
        self.device = device

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        height, width = frame_bgr.shape[:2]
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf,
            classes=[CAT_CLASS_ID],
            verbose=False,
            device=self.device,
        )
        detections: list[Detection] = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None:
            return detections

        for box in boxes:
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0].item())
            det = Detection(
                x1=int(xyxy[0]),
                y1=int(xyxy[1]),
                x2=int(xyxy[2]),
                y2=int(xyxy[3]),
                confidence=conf,
            ).clip(width, height)
            if det.area > 0:
                detections.append(det)
        return detections


def pick_primary(detections: list[Detection], frame_shape: tuple[int, ...]) -> Detection | None:
    """Prefer the largest box; break ties by closeness to frame center."""
    if not detections:
        return None
    height, width = frame_shape[:2]
    cx, cy = width / 2.0, height / 2.0

    def score(det: Detection) -> tuple[float, float]:
        dx = det.center[0] - cx
        dy = det.center[1] - cy
        dist2 = dx * dx + dy * dy
        return (float(det.area), -dist2)

    return max(detections, key=score)


def crop_detection(
    frame_bgr: np.ndarray,
    det: Detection,
    pad_ratio: float = 0.08,
) -> np.ndarray | None:
    height, width = frame_bgr.shape[:2]
    bw = det.x2 - det.x1
    bh = det.y2 - det.y1
    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)
    x1 = max(0, det.x1 - pad_x)
    y1 = max(0, det.y1 - pad_y)
    x2 = min(width, det.x2 + pad_x)
    y2 = min(height, det.y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame_bgr[y1:y2, x1:x2].copy()
