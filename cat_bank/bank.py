"""Named cat bank: store enrollment crops and metadata."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

SAFE_NAME = re.compile(r"[^a-zA-Z0-9_\- ]+")


def normalize_name(name: str) -> str:
    cleaned = SAFE_NAME.sub("", name.strip())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def slugify(name: str) -> str:
    return re.sub(r"\s+", "_", normalize_name(name).lower())


@dataclass
class CatProfile:
    name: str
    folder: str
    images: list[str]


class CatBank:
    def __init__(self, root: str | Path = "cats_bank") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        self._index: dict[str, dict] = self._load_index()

    def _load_index(self) -> dict[str, dict]:
        if not self.index_path.exists():
            return {"cats": {}}
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"cats": {}}
        if "cats" not in data or not isinstance(data["cats"], dict):
            return {"cats": {}}
        return data

    def save_index(self) -> None:
        self.index_path.write_text(
            json.dumps(self._index, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def list_cats(self) -> list[str]:
        names = sorted(self._index.get("cats", {}).keys(), key=str.lower)
        return names

    def get_profile(self, name: str) -> CatProfile | None:
        key = normalize_name(name)
        meta = self._index.get("cats", {}).get(key)
        if not meta:
            return None
        folder = self.root / meta["folder"]
        images = sorted(str(p) for p in folder.glob("*.jpg"))
        return CatProfile(name=key, folder=meta["folder"], images=images)

    def ensure_cat(self, name: str) -> CatProfile:
        key = normalize_name(name)
        if not key:
            raise ValueError("Cat name cannot be empty")
        cats = self._index.setdefault("cats", {})
        if key not in cats:
            folder = slugify(key)
            (self.root / folder).mkdir(parents=True, exist_ok=True)
            cats[key] = {
                "folder": folder,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            self.save_index()
        profile = self.get_profile(key)
        assert profile is not None
        return profile

    def save_crop(self, name: str, crop_bgr: np.ndarray) -> Path:
        profile = self.ensure_cat(name)
        folder = self.root / profile.folder
        folder.mkdir(parents=True, exist_ok=True)
        existing = list(folder.glob("*.jpg"))
        next_id = len(existing) + 1
        path = folder / f"{next_id:03d}.jpg"
        # Avoid overwrite if files were deleted out of order
        while path.exists():
            next_id += 1
            path = folder / f"{next_id:03d}.jpg"
        ok = cv2.imwrite(str(path), crop_bgr)
        if not ok:
            raise OSError(f"Failed to write crop to {path}")
        cats = self._index.setdefault("cats", {})
        cats[profile.name]["updated_at"] = datetime.now(timezone.utc).isoformat()
        cats[profile.name]["image_count"] = len(list(folder.glob("*.jpg")))
        self.save_index()
        return path

    def iter_reference_images(self) -> list[tuple[str, Path]]:
        refs: list[tuple[str, Path]] = []
        for name in self.list_cats():
            profile = self.get_profile(name)
            if not profile:
                continue
            for image_path in profile.images:
                refs.append((name, Path(image_path)))
        return refs
