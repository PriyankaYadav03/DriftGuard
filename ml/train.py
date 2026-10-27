"""Train DriftGuard V1 and compare imbalance-handling strategies.

    python -m ml.train --rows 400000 --prevalence 0.02

All strategies use the same model family (HistGradientBoosting), the same time-ordered
split and the same test set, so the only thing that changes is how imbalance is handled.
Resampling is applied to the TRAINING block only. Validation and test keep the natural
attack rate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression

from . import data as D
from .evaluate import (curves, metrics_at, precision_at_prevalence,
                       threshold_for_fpr, threshold_free)
from .features import MODEL_FEATURES, build_matrix

ROOT = Path(__file__).resolve().parent.parent


def make_model(seed: int, class_weight=None):
    return HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.1, max_leaf_nodes=31, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=15, class_weight=class_weight,
        random_state=seed)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "flows_large.csv"))
    ap.add_argument("--cic", nargs="*", help="CIC-IDS2017 CSV files (overrides --data)")
    ap.add_argument("--rows", type=int, default=400_000, help="rows to generate if --data is missing")
    ap.add_argument("--prevalence", type=float, default=0.02, help="attack share after thinning")
    ap.add_argument("--target-fpr", type=float, default=0.01, help="false-positive budget for thresholds")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(ROOT / "artifacts"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # ---- data -------------------------------------------------------------
    source = "cic-ids2017" if args.cic else "synthetic"
    if args.cic:
        df = D.load_cicids(args.cic)
        data_hash = "multi-file"
    else:
        path = Path(args.data)
        if not path.exists():
            import sys; sys.path.insert(0, str(ROOT))
            from data_gen import generate_flows
            path.parent.mkdir(parents=True, exist_ok=True)
            generate_flows(args.rows, args.seed).to_csv(path, index=False)
            print(f"generated {args.rows:,} synthetic flows -> {path}")
        df = D.load_flows(path)
        data_hash = sha256(path)
    raw_prev = float(df["label"].mean())
    df = D.ensure_time_order(df)
    df = D.set_prevalence(df, args.prevalence, args.seed)
    train, val, test = D.temporal_split(df)
    print(f"rows={len(df):,} attack_rate={df['label'].mean():.2%} "
          f"(raw file {raw_prev:.2%}) train/val/test={len(train):,}/{len(val):,}/{len(test):,}")

    Xtr, ytr = build_matrix(train), train["label"].to_numpy()
    Xva, yva = build_matrix(val), val["label"].to_numpy()
    Xte, yte = build_matrix(test), test["label"].to_numpy()
    fpr_budget = args.target_fpr

    # ---- strategies ---------------------------------------------------------
    fitted: dict[str, tuple] = {}   # name -> (model, fit_seconds, n_train_rows)

    def fit(name, X, y, class_weight=None):
        t = time.time()
        m = make_model(args.seed, class_weight).fit(X, y)
        fitted[name] = (m, time.time() - t, len(y))
        print(f"  fitted {name:<12} rows={len(y):>8,} attacks={int(y.sum()):>6,} {time.time()-t:5.1f}s")

    print("training strategies")
    fit("baseline", Xtr, ytr)
    fit("weighted", Xtr, ytr, class_weight="balanced")
    Xu, yu = RandomUnderSampler(sampling_strategy=0.25, random_state=args.seed).fit_resample(Xtr, ytr)
    fit("undersample", Xu, yu)
    Xs, ys = SMOTE(sampling_strategy=0.25, random_state=args.seed).fit_resample(Xtr, ytr)
    fit("smote", Xs, ys)

    # (label, fitted model, threshold mode)
    runs = [
        ("Baseline, threshold 0.5", "baseline", "fixed"),
        ("Baseline + tuned threshold", "baseline", "tuned"),
        ("Class weights, threshold 0.5", "weighted", "fixed"),
        ("Class weights + tuned threshold", "weighted", "tuned"),
        ("Undersample benign + tuned threshold", "undersample", "tuned"),
        ("SMOTE + tuned threshold", "smote", "tuned"),
    ]
    rows, scores = [], {}
    for label, key, mode in runs:
        model = fitted[key][0]
        s_va = model.predict_proba(Xva)[:, 1]
        s_te = scores.setdefault(key, model.predict_proba(Xte)[:, 1])
        thr = 0.5 if mode == "fixed" else threshold_for_fpr(yva, s_va, fpr_budget)
        m_te = metrics_at(yte, s_te, thr)
        tf = threshold_free(yte, s_te, fpr_budget)
        val_recall = metrics_at(yva, s_va, threshold_for_fpr(yva, s_va, fpr_budget))["recall"]
        rows.append({"strategy": label, "model_key": key, "threshold_mode": mode,
                     **{k: round(v, 5) if isinstance(v, float) else v for k, v in {**m_te, **tf}.items()},
                     "val_recall_at_budget": round(val_recall, 5),
                     "fit_seconds": round(fitted[key][1], 1), "train_rows": fitted[key][2]})

    # ---- choose the final model on VALIDATION data only ---------------------
    tuned = [r for r in rows if r["threshold_mode"] == "tuned"]
    best = max(tuned, key=lambda r: (r["val_recall_at_budget"], r["pr_auc"]))
    model = fitted[best["model_key"]][0]
    s_va = model.predict_proba(Xva)[:, 1]
    s_te = scores[best["model_key"]]
    thr = threshold_for_fpr(yva, s_va, fpr_budget)
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(s_va, yva)
    print(f"selected: {best['strategy']} (val recall@{fpr_budget:.0%}FPR={best['val_recall_at_budget']:.3f})")

    final = metrics_at(yte, s_te, thr)
    final.update(threshold_free(yte, s_te, fpr_budget))
    tpr, fpr = final["recall"], final["fpr"]
    sweep = [{"prevalence": p, "precision": round(precision_at_prevalence(tpr, fpr, p), 4)}
             for p in (0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001)]

    pred = s_te >= thr
    per_type = []
    for name, g in test.assign(pred=pred).groupby("attack_type"):
        if name == "Benign" or name.upper() == "BENIGN":
            continue
        per_type.append({"attack_type": name, "flows": int(len(g)), "detected": int(g["pred"].sum()),
                         "recall": round(float(g["pred"].mean()), 4)})
    per_type.sort(key=lambda r: -r["flows"])

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(Xte), min(20_000, len(Xte)), replace=False)
    imp = permutation_importance(model, Xte.iloc[idx], yte[idx], scoring="average_precision",
                                 n_repeats=3, random_state=args.seed, n_jobs=2)
    importance = sorted(({"feature": f, "importance": round(float(v), 5)}
                         for f, v in zip(MODEL_FEATURES, imp.importances_mean)),
                        key=lambda r: -r["importance"])[:12]

    version = datetime.now(timezone.utc).strftime("v1-%Y%m%d-%H%M%S")
    bundle = {"model": model, "calibrator": calibrator, "threshold": float(thr),
              "features": MODEL_FEATURES, "version": version, "strategy": best["strategy"],
              "fpr_budget": fpr_budget}
    joblib.dump(bundle, out / "model.joblib", compress=3)

    metrics = {
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selected_strategy": best["strategy"],
        "fpr_budget": fpr_budget,
        "data": {"source": source, "hash": data_hash, "rows": len(df),
                 "attack_rate": round(float(df["label"].mean()), 5),
                 "raw_file_attack_rate": round(raw_prev, 5),
                 "split": {"train": len(train), "val": len(val), "test": len(test)},
                 "time_split_is_meaningful": source != "synthetic"},
        "test": final,
        "comparison": rows,
        "precision_vs_prevalence": sweep,
        "per_attack_type": per_type,
        "feature_importance": importance,
        "curves": curves(yte, s_te),
        "caveats": (["Synthetic data: results show the pipeline works, not real-world accuracy."]
                    if source == "synthetic" else []),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))

    replay = test.iloc[:20_000]
    replay.to_csv(out / "replay.csv", index=False)
    print(f"saved model + metrics + replay sample to {out}")
    print(f"TEST  PR-AUC={final['pr_auc']:.3f} recall={final['recall']:.3f} "
          f"precision={final['precision']:.3f} FPR={final['fpr']:.4f}")


if __name__ == "__main__":
    main()
