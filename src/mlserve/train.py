"""Train the model that the service serves.

Wine quality is a real, small, public dataset that ships with scikit-learn, so
the service has something meaningful to predict without downloading anything.
Training writes a versioned bundle: the pipeline, the feature order, the class
names and the metrics it was accepted on. The service refuses to load a bundle
that is missing any of them.
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODELS = Path(__file__).resolve().parents[2] / "models"


def train(version: str | None = None, seed: int = 42, output_dir: Path | None = None) -> dict:
    data = load_wine()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.25, stratify=data.target, random_state=seed
    )
    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=250, min_samples_leaf=2, random_state=seed)),
    ])
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_macro": float(f1_score(y_test, predictions, average="macro")),
        "cv_accuracy_mean": float(np.mean(cross_val_score(pipeline, data.data, data.target, cv=5))),
        "test_samples": int(len(y_test)),
    }
    version = version or datetime.now(timezone.utc).strftime("%Y.%m.%d.%H%M")
    out = (output_dir or MODELS) / version
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out / "model.joblib")
    (out / "metadata.json").write_text(json.dumps({
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "RandomForestClassifier(n_estimators=250, min_samples_leaf=2) on StandardScaler",
        "dataset": "scikit-learn wine (178 samples, 13 features, 3 classes)",
        "features": list(data.feature_names),
        "classes": list(data.target_names),
        "metrics": metrics,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
    }, indent=2))
    (out / "classification_report.txt").write_text(
        classification_report(y_test, predictions, target_names=data.target_names)
    )
    return {"version": version, "path": str(out), **metrics}
