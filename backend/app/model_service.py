from __future__ import annotations

import threading
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.features import build_matrix


def severity(prob: float, alert: bool) -> str:
    if not alert:
        return "none"
    return "high" if prob >= 0.8 else "medium" if prob >= 0.4 else "low"


class ModelService:
    """Loads the model bundle once. Scoring is read-only, so it is safe across threads."""

    def __init__(self, path: Path):
        self._lock = threading.Lock()
        self.load(path)

    def load(self, path: Path) -> None:
        bundle = joblib.load(path)   # only ever load artifacts you trained yourself (pickle)
        with self._lock:
            self.bundle = bundle

    @property
    def version(self) -> str:
        return self.bundle["version"]

    @property
    def info(self) -> dict:
        b = self.bundle
        return {"version": b["version"], "strategy": b["strategy"], "threshold": b["threshold"],
                "fpr_budget": b["fpr_budget"], "n_features": len(b["features"])}

    def score(self, df: pd.DataFrame):
        b = self.bundle
        raw = b["model"].predict_proba(build_matrix(df))[:, 1]
        prob = b["calibrator"].predict(raw)
        alert = raw >= b["threshold"]
        return raw, np.asarray(prob), alert
