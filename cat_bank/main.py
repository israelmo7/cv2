#!/usr/bin/env python3
"""Live webcam cat bank: detect cats with YOLO, enroll by name, recognize later."""

from __future__ import annotations

import argparse
import sys
import time
from enum import Enum, auto
from pathlib import Path

import cv2
import numpy as np

from bank import CatBank, normalize_name
from detector import CatDetector, crop_detection, pick_primary
from recognizer import CatRecognizer

WINDOW = "Cat Bank"


class Mode(Enum):
    RECOGNIZE = auto()
    ENROLL_NAME = auto()
    ENROLL_CAPTURE = auto()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live YOLO cat bank")
    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index (default 0), video path, or image path",
    )
    parser.add_argument(
        "--bank",
        default=str(Path(__file__).resolve().parent / "cats_bank"),
        help="Cat bank directory",
    )
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics YOLO weights")
    parser.add_argument("--conf", type=float, default=0.45, help="YOLO confidence")
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.72,
        help="Min cosine similarity to accept a name",
    )
    parser.add_argument(
        "--enroll-count",
        type=int,
        default=8,
        help="Number of crops to save during enrollment",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device for YOLO/embeddings (e.g. cpu, cuda, 0)",
    )
    return parser.parse_args()


def open_capture(source: str) -> tuple[cv2.VideoCapture | None, np.ndarray | None, bool]:
    """Return (capture, static_image, is_image)."""
    path = Path(source)
    if path.exists() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        image = cv2.imread(str(path))
        if image is None:
            raise SystemExit(f"Could not read image: {source}")
        return None, image, True

    camera_index: int | str
    if source.isdigit():
        camera_index = int(source)
    else:
        camera_index = source

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {source}")
    return cap, None, False


def draw_hud(
    frame: np.ndarray,
    mode: Mode,
    name_buffer: str,
    status: str,
    overlays_on: bool,
    bank_names: list[str],
) -> None:
    lines = [
        "e enroll | d toggle overlays | l list bank | q/x quit",
        f"mode: {mode.name} | overlays: {'on' if overlays_on else 'off'}",
    ]
    if mode == Mode.ENROLL_NAME:
        lines.append(f"type name: {name_buffer}_  (Enter confirm, Esc cancel)")
    if status:
        lines.append(status)
    if bank_names:
        preview = ", ".join(bank_names[:6])
        if len(bank_names) > 6:
            preview += ", ..."
        lines.append(f"bank: {preview}")

    y = 24
    for line in lines:
        cv2.putText(
            frame,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        y += 22


def draw_detection(
    frame: np.ndarray,
    det,
    label: str,
    color: tuple[int, int, int],
) -> None:
    cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), color, 2)
    text_origin = (det.x1, max(20, det.y1 - 8))
    cv2.putText(
        frame,
        label,
        text_origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        label,
        text_origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


def main() -> int:
    args = parse_args()
    bank = CatBank(args.bank)
    print(f"Loading YOLO model: {args.model}")
    detector = CatDetector(model_name=args.model, conf=args.conf, device=args.device)
    print("Loading embedding model (ResNet18)...")
    recognizer = CatRecognizer(
        bank=bank,
        match_threshold=args.match_threshold,
        device=args.device,
    )

    cap, static_image, is_image = open_capture(args.source)
    print("Controls: e=enroll  d=overlays  l=list  q/x=quit")
    if bank.list_cats():
        print("Cats in bank:", ", ".join(bank.list_cats()))
    else:
        print("Cat bank is empty — press e while a cat is visible to enroll.")

    mode = Mode.RECOGNIZE
    name_buffer = ""
    enroll_name = ""
    enroll_saved = 0
    enroll_target = max(1, args.enroll_count)
    last_enroll_save = 0.0
    status = ""
    overlays_on = True
    label_memory: dict[tuple[int, int, int, int], tuple[str, float, float]] = {}

    while True:
        if is_image:
            assert static_image is not None
            frame = static_image.copy()
        else:
            assert cap is not None
            ok, frame = cap.read()
            if not ok or frame is None:
                print("End of stream / failed to read frame")
                break

        display = frame.copy()
        detections = detector.detect(frame)
        primary = pick_primary(detections, frame.shape)
        now = time.time()

        if mode == Mode.ENROLL_CAPTURE and primary is not None:
            if now - last_enroll_save >= 0.18:
                crop = crop_detection(frame, primary)
                if crop is not None and crop.size > 0:
                    path = recognizer.add_enrollment_crop(enroll_name, crop)
                    enroll_saved += 1
                    last_enroll_save = now
                    status = f"Saved {enroll_saved}/{enroll_target}: {path.name}"
            if enroll_saved >= enroll_target:
                status = f"Enrolled {enroll_name} ({enroll_saved} crops)"
                print(status)
                mode = Mode.RECOGNIZE
                enroll_name = ""
                enroll_saved = 0

        if overlays_on:
            for det in detections:
                crop = crop_detection(frame, det)
                label = "cat"
                color = (80, 200, 80)
                if mode == Mode.RECOGNIZE and crop is not None:
                    key = (det.x1 // 20, det.y1 // 20, det.x2 // 20, det.y2 // 20)
                    cached = label_memory.get(key)
                    if cached and now - cached[2] < 0.35:
                        name, score = cached[0], cached[1]
                    else:
                        name, score = recognizer.match(crop)
                        label_memory[key] = (name, score, now)
                    if name == "Unknown":
                        label = f"Unknown {score:.2f}"
                        color = (40, 180, 255)
                    else:
                        label = f"{name} {score:.2f}"
                        color = (80, 220, 80)
                elif mode in (Mode.ENROLL_NAME, Mode.ENROLL_CAPTURE) and det is primary:
                    label = f"enrolling: {enroll_name or name_buffer or '?'}"
                    color = (0, 200, 255)
                draw_detection(display, det, label, color)

        # Drop stale label cache entries
        label_memory = {
            k: v for k, v in label_memory.items() if now - v[2] < 1.0
        }

        draw_hud(
            display,
            mode,
            name_buffer,
            status,
            overlays_on,
            bank.list_cats(),
        )
        cv2.imshow(WINDOW, display)

        key = cv2.waitKey(1 if not is_image else 30) & 0xFF
        if key == 255:
            if is_image and mode == Mode.RECOGNIZE:
                # Keep showing the image until user acts
                continue
            continue

        if mode == Mode.ENROLL_NAME:
            if key in (27,):  # Esc
                mode = Mode.RECOGNIZE
                name_buffer = ""
                status = "Enrollment cancelled"
            elif key in (13, 10):  # Enter
                enroll_name = normalize_name(name_buffer)
                if not enroll_name:
                    status = "Name cannot be empty"
                elif primary is None:
                    status = "No cat detected — enrollment cancelled"
                    mode = Mode.RECOGNIZE
                    name_buffer = ""
                else:
                    bank.ensure_cat(enroll_name)
                    mode = Mode.ENROLL_CAPTURE
                    enroll_saved = 0
                    last_enroll_save = 0.0
                    status = f"Capturing {enroll_target} crops for {enroll_name}"
                    print(status)
                    name_buffer = ""
            elif key in (8, 127):  # Backspace
                name_buffer = name_buffer[:-1]
            elif 32 <= key <= 126:
                if len(name_buffer) < 32:
                    name_buffer += chr(key)
            continue

        if key in (ord("q"), ord("x")):
            break
        if key == ord("d"):
            overlays_on = not overlays_on
            status = f"Overlays {'on' if overlays_on else 'off'}"
        elif key == ord("l"):
            names = bank.list_cats()
            status = "Bank empty" if not names else "Bank: " + ", ".join(names)
            print(status)
        elif key == ord("e"):
            if primary is None:
                status = "No cat visible to enroll"
            else:
                mode = Mode.ENROLL_NAME
                name_buffer = ""
                status = "Type a name, then Enter"
        elif is_image and key == ord("n"):
            # Convenience: advance is a no-op for single image; kept for symmetry
            pass

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
