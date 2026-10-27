import numpy as np
import pandas as pd
import pytest

import data_gen
from ml import data as D
from ml.evaluate import metrics_at, precision_at_prevalence, recall_at_fpr, threshold_for_fpr
from ml.features import MODEL_FEATURES, build_matrix
from ml.schema import FEATURES


def test_schema_matches_generator():
    assert FEATURES == data_gen.FEATURES


def test_features_handle_inf_nan_and_keep_order():
    df = data_gen.generate_flows(50, seed=1)
    df.loc[0, "flow_bytes_per_s"] = np.inf
    df.loc[1, "fwd_iat_mean_ms"] = np.nan
    X = build_matrix(df)
    assert list(X.columns) == MODEL_FEATURES
    assert np.isfinite(X.to_numpy()).all()


def test_set_prevalence_thins_attacks_only():
    df = D.ensure_time_order(data_gen.generate_flows(20_000, seed=2))
    out = D.set_prevalence(df, 0.02, seed=0)
    assert (out["label"] == 0).sum() == (df["label"] == 0).sum()
    assert abs(out["label"].mean() - 0.02) < 0.002
    assert out["timestamp"].is_monotonic_increasing


def test_temporal_split_is_ordered_and_disjoint():
    df = D.ensure_time_order(data_gen.generate_flows(1000, seed=3))
    tr, va, te = D.temporal_split(df)
    assert tr["timestamp"].max() < va["timestamp"].min() <= va["timestamp"].max() < te["timestamp"].min()
    assert len(tr) + len(va) + len(te) == 1000


def test_threshold_for_fpr_respects_budget():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(10_000), np.ones(200)].astype(int)
    s = np.r_[rng.beta(2, 8, 10_000), rng.beta(6, 3, 200)]
    thr = threshold_for_fpr(y, s, 0.01)
    assert metrics_at(y, s, thr)["fpr"] <= 0.01
    assert 0 < recall_at_fpr(y, s, 0.01) <= 1


def test_precision_at_prevalence_exact():
    assert precision_at_prevalence(0.8, 0.01, 0.5) == pytest.approx(0.8 * 0.5 / (0.8 * 0.5 + 0.01 * 0.5))
    assert precision_at_prevalence(0.8, 0.01, 0.001) < 0.1   # rare attacks -> mostly false alarms


def test_cic_loader_maps_columns_and_units(tmp_path):
    from ml.data import CIC_MAP
    row = {src: 10 for src in CIC_MAP}
    row.update({"Flow Duration": 5000, "Label": "DoS Hulk"})
    r2 = dict(row, Label="BENIGN")
    cols = {f" {k}" if i % 2 else k: v for i, (k, v) in enumerate(row.items())}   # CIC pads names with spaces
    cols2 = {f" {k}" if i % 2 else k: v for i, (k, v) in enumerate(r2.items())}
    cols[" Label"] = cols.pop("Label", "DoS Hulk"); cols2[" Label"] = cols2.pop("Label", "BENIGN")
    k = next(k for k in cols2 if k.strip() == "Total Fwd Packets")
    cols2[k] = 99     # make the rows distinct so de-dup keeps both
    p = tmp_path / "cic.csv"
    pd.DataFrame([cols, cols2]).to_csv(p, index=False)
    out = D.load_cicids([p])
    assert list(out["label"]) == [1, 0]
    assert out["flow_duration_ms"].iloc[0] == pytest.approx(5.0)   # 5000 us -> 5 ms
