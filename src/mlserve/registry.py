"""Loading model versions from disk.

A service that serves "the model" and cannot say which one is a service you
cannot debug. Every bundle is a folder under models/<version>/ holding the
pipeline and the metadata written at training time. The registry can list
versions, load any of them, and switch the active one without a restart.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

REQUIRED_METADATA = {"version", "features", "classes", "metrics", "created_at"}


class ModelNotFound(Exception):
    pass


class InvalidBundle(Exception):
    pass


@dataclass
class ModelBundle:
    version: str
    pipeline: object
    metadata: dict

    @property
    def features(self) -> list[str]:
        return list(self.metadata["features"])

    @property
    def classes(self) -> list[str]:
        return list(self.metadata["classes"])

    def predict(self, rows: list[list[float]]) -> tuple[list[str], list[dict[str, float]]]:
        matrix = np.asarray(rows, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.features):
            raise ValueError(f"expected {len(self.features)} features, got {matrix.shape[-1]}")
        probabilities = self.pipeline.predict_proba(matrix)
        labels = [self.classes[int(i)] for i in probabilities.argmax(axis=1)]
        # strict: one probability per class. A row of the wrong width means the
        # loaded model does not match the class list recorded beside it.
        spread = [{c: float(round(p, 6)) for c, p in zip(self.classes, row, strict=True)}
                  for row in probabilities]
        return labels, spread


class ModelRegistry:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._cache: dict[str, ModelBundle] = {}
        self._active: str | None = None

    def versions(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.name for p in self.root.iterdir() if (p / "model.joblib").exists())

    def latest(self) -> str:
        versions = self.versions()
        if not versions:
            raise ModelNotFound(f"no model versions under {self.root}")
        return versions[-1]

    def load(self, version: str | None = None) -> ModelBundle:
        version = version or self.latest()
        if version in self._cache:
            return self._cache[version]
        folder = self.root / version
        if not (folder / "model.joblib").exists():
            raise ModelNotFound(f"model version {version!r} not found in {self.root}")
        try:
            metadata = json.loads((folder / "metadata.json").read_text())
        except FileNotFoundError as exc:
            raise InvalidBundle(f"{version} has no metadata.json") from exc
        missing = REQUIRED_METADATA - metadata.keys()
        if missing:
            raise InvalidBundle(f"{version} metadata is missing {sorted(missing)}")
        bundle = ModelBundle(version, joblib.load(folder / "model.joblib"), metadata)
        self._cache[version] = bundle
        return bundle

    @property
    def active(self) -> ModelBundle:
        return self.load(self._active)

    def activate(self, version: str) -> ModelBundle:
        bundle = self.load(version)   # raises before anything changes
        self._active = version
        return bundle
