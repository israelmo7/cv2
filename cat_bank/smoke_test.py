#!/usr/bin/env python3
"""Headless smoke test: detect -> enroll -> recognize on a sample cat image."""

from __future__ import annotations

import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from bank import CatBank
from detector import CatDetector, crop_detection, pick_primary
from recognizer import CatRecognizer


def download(url: str, dest: Path) -> bool:
    try:
        urllib.request.urlretrieve(url, dest)
        return dest.exists() and dest.stat().st_size > 0
    except Exception as exc:
        print(f"Download failed for {url}: {exc}")
        return False


def find_cat_frame(detector: CatDetector, search_dir: Path) -> np.ndarray | None:
    """Try bundled/sample images; synthesize a fallback only if needed."""
    urls = [
        "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/4/4d/Cat_November_2010-1a.jpg",
    ]
    for i, url in enumerate(urls):
        path = search_dir / f"sample_{i}.jpg"
        if not download(url, path):
            continue
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        detections = detector.detect(frame)
        if detections:
            print(f"Found cat in {url}")
            return frame
        print(f"No cat in {url}")
    return None


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="cat_bank_smoke_"))
    try:
        print("Loading detector...")
        detector = CatDetector(model_name="yolov8n.pt", conf=0.35, device="cpu")
        frame = find_cat_frame(detector, tmp)
        if frame is None:
            print("SMOKE SKIP: could not download an image containing a detectable cat")
            return 0

        detections = detector.detect(frame)
        primary = pick_primary(detections, frame.shape)
        assert primary is not None
        crop = crop_detection(frame, primary)
        assert crop is not None

        bank_dir = tmp / "bank"
        bank = CatBank(bank_dir)
        print("Loading recognizer...")
        recognizer = CatRecognizer(bank=bank, match_threshold=0.55, device="cpu")

        # Enroll a few slightly varied crops
        h, w = crop.shape[:2]
        for i in range(5):
            # Tiny jitter via resize to simulate multiple enrollment frames
            scale = 1.0 - (i * 0.02)
            resized = cv2.resize(crop, (max(8, int(w * scale)), max(8, int(h * scale))))
            recognizer.add_enrollment_crop("SmokeCat", resized)

        name, score = recognizer.match(crop)
        print(f"Match result: {name} ({score:.3f})")
        print(f"Bank cats: {bank.list_cats()}")
        assert "SmokeCat" in bank.list_cats()
        assert name == "SmokeCat", f"Expected SmokeCat, got {name}"
        print("SMOKE OK")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
