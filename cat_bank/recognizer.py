"""Embedding-based named cat recognition against the cat bank."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

from bank import CatBank


class CatRecognizer:
    def __init__(
        self,
        bank: CatBank,
        match_threshold: float = 0.72,
        device: str | None = None,
    ) -> None:
        self.bank = bank
        self.match_threshold = match_threshold
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        weights = models.ResNet18_Weights.DEFAULT
        backbone = models.resnet18(weights=weights)
        self.model = nn.Sequential(*list(backbone.children())[:-1]).to(self.device)
        self.model.eval()
        self.preprocess = transforms.Compose(
            [
                transforms.ToPILImage(),
                weights.transforms(),
            ]
        )

        # name -> (N, D) float32 embeddings
        self._gallery: dict[str, np.ndarray] = {}
        self.rebuild()

    @torch.inference_mode()
    def embed(self, image_bgr: np.ndarray) -> np.ndarray:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Empty image for embedding")
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.preprocess(rgb).unsqueeze(0).to(self.device)
        features = self.model(tensor).flatten(1)
        features = nn.functional.normalize(features, p=2, dim=1)
        return features.squeeze(0).cpu().numpy().astype(np.float32)

    def rebuild(self) -> None:
        gallery: dict[str, list[np.ndarray]] = {}
        for name, path in self.bank.iter_reference_images():
            image = cv2.imread(str(path))
            if image is None:
                continue
            try:
                emb = self.embed(image)
            except Exception:
                continue
            gallery.setdefault(name, []).append(emb)

        self._gallery = {
            name: np.stack(vectors, axis=0)
            for name, vectors in gallery.items()
            if vectors
        }

    def known_names(self) -> list[str]:
        return sorted(self._gallery.keys(), key=str.lower)

    def match(self, crop_bgr: np.ndarray) -> tuple[str, float]:
        """Return (name, score). Name is 'Unknown' when below threshold."""
        if not self._gallery:
            return "Unknown", 0.0

        query = self.embed(crop_bgr)
        best_name = "Unknown"
        best_score = -1.0
        for name, vectors in self._gallery.items():
            # Max cosine similarity against that cat's references
            scores = vectors @ query
            score = float(np.max(scores))
            if score > best_score:
                best_score = score
                best_name = name

        if best_score < self.match_threshold:
            return "Unknown", best_score
        return best_name, best_score

    def add_enrollment_crop(self, name: str, crop_bgr: np.ndarray) -> Path:
        path = self.bank.save_crop(name, crop_bgr)
        emb = self.embed(crop_bgr)
        if name in self._gallery:
            self._gallery[name] = np.vstack([self._gallery[name], emb[None, :]])
        else:
            self._gallery[name] = emb[None, :]
        return path
