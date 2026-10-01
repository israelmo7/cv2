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
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; CatBankSmoke/1.0)"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            dest.write_bytes(response.read())
        return dest.exists() and dest.stat().st_size > 0
    except Exception as exc:
        print(f"Download failed for {url}: {exc}")
        return False


def find_cat_frame(detector: CatDetector, search_dir: Path) -> np.ndarray | None:
    """Download sample images until YOLO finds a cat."""
    urls = [
        "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg",
        "https://cdn.pixabay.com/photo/2017/02/20/18/03/cat-2083492_640.jpg",
        "https://images.pexels.com/photos/45201/kitty-cat-kitten-pet-45201.jpeg?auto=compress&cs=tinysrgb&w=640",
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
        recognizer = CatRecognizer(bank=bank, match_threshold=0.85, device="cpu")

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

        # Annotated proof image for walkthrough artifacts
        annotated = frame.copy()
        cv2.rectangle(
            annotated,
            (primary.x1, primary.y1),
            (primary.x2, primary.y2),
            (80, 220, 80),
            2,
        )
        cv2.putText(
            annotated,
            f"{name} {score:.2f}",
            (primary.x1, max(24, primary.y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (80, 220, 80),
            2,
            cv2.LINE_AA,
        )
        out_dir = Path("/opt/cursor/artifacts")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "smoke_recognize_smokecat.jpg"
        cv2.imwrite(str(out_path), annotated)
        print(f"Wrote {out_path}")
        print("SMOKE OK")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
