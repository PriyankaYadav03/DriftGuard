"""Dataset loading, imbalance shaping and the time-ordered split."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .schema import FEATURES

# CIC-IDS2017 (CICFlowMeter) column -> DriftGuard column, plus unit scale.
# CIC reports durations / IATs in microseconds; we use milliseconds.
CIC_MAP = {
    "Destination Port": ("dst_port", 1.0),
    "Protocol": ("protocol", 1.0),
    "Flow Duration": ("flow_duration_ms", 1e-3),
    "Total Fwd Packets": ("fwd_packets", 1.0),
    "Total Backward Packets": ("bwd_packets", 1.0),
    "Total Length of Fwd Packets": ("fwd_bytes", 1.0),
    "Total Length of Bwd Packets": ("bwd_bytes", 1.0),
    "Packet Length Mean": ("pkt_len_mean", 1.0),
    "Packet Length Std": ("pkt_len_std", 1.0),
    "Flow Bytes/s": ("flow_bytes_per_s", 1.0),
    "Flow Packets/s": ("flow_pkts_per_s", 1.0),
    "Fwd IAT Mean": ("fwd_iat_mean_ms", 1e-3),
    "SYN Flag Count": ("syn_flags", 1.0),
    "ACK Flag Count": ("ack_flags", 1.0),
    "RST Flag Count": ("rst_flags", 1.0),
    "PSH Flag Count": ("psh_flags", 1.0),
    "Init_Win_bytes_forward": ("init_win_bytes", 1.0),
    "Active Mean": ("active_mean_ms", 1e-3),
    "Idle Mean": ("idle_mean_ms", 1e-3),
}


def load_flows(path: str | Path) -> pd.DataFrame:
    """Load a DriftGuard-format CSV (as written by data_gen.py)."""
    df = pd.read_csv(path)
    missing = [c for c in FEATURES + ["label"] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    if "attack_type" not in df.columns:
        df["attack_type"] = np.where(df["label"] == 1, "Attack", "Benign")
    return df


def load_cicids(paths: list[str | Path]) -> pd.DataFrame:
    """Load CIC-IDS2017 CSV files into the DriftGuard schema.

    NOTE: written against the published column names but NOT yet tested on the real
    files (no network access when this was built). Run it once on your copy and
    inspect `df.describe()` before trusting results. The 'MachineLearningCSV'
    release has no Protocol/Timestamp columns; 'TrafficLabelling' has both.
    """
    frames = []
    for p in paths:
        raw = pd.read_csv(p, encoding="latin-1", low_memory=False)
        raw.columns = [c.strip() for c in raw.columns]
        frames.append(raw)
    raw = pd.concat(frames, ignore_index=True)

    out = pd.DataFrame(index=raw.index)
    for src, (dst, scale) in CIC_MAP.items():
        if src in raw.columns:
            out[dst] = pd.to_numeric(raw[src], errors="coerce") * scale
        elif dst == "protocol":
            out[dst] = 6  # assume TCP when the column is absent
        else:
            raise ValueError(f"CIC file is missing expected column '{src}'")
    out["attack_type"] = raw["Label"].astype(str).str.strip().str.replace("�", "-", regex=False)
    out["label"] = (out["attack_type"].str.upper() != "BENIGN").astype(int)
    if "Timestamp" in raw.columns:
        out["timestamp"] = pd.to_datetime(raw["Timestamp"], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
    out = out[out["flow_duration_ms"] >= 0]
    out = out.drop_duplicates(subset=FEATURES + ["label"])  # CIC has many exact duplicates
    return out.reset_index(drop=True)


def ensure_time_order(df: pd.DataFrame) -> pd.DataFrame:
    """Return df sorted by time. Synthetic data has no clock, so row order is used.

    For the synthetic generator rows are i.i.d., so a time split is statistically the
    same as a random split. It only becomes meaningful on real data (or in V2, once
    drift is injected into the generator).
    """
    df = df.copy()
    if "timestamp" in df.columns and df["timestamp"].notna().all():
        return df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    df["timestamp"] = pd.Timestamp("2026-01-01") + pd.to_timedelta(np.arange(len(df)), unit="s")
    return df


def set_prevalence(df: pd.DataFrame, prevalence: float, seed: int = 0) -> pd.DataFrame:
    """Thin out attack rows (keeping time order) so attacks make up ~`prevalence` of rows."""
    if not 0 < prevalence < 1:
        raise ValueError("prevalence must be in (0, 1)")
    rng = np.random.default_rng(seed)
    n_benign = int((df["label"] == 0).sum())
    n_attack_target = int(round(n_benign * prevalence / (1 - prevalence)))
    attack_idx = df.index[df["label"] == 1].to_numpy()
    if n_attack_target >= len(attack_idx):
        return df
    keep = rng.choice(attack_idx, n_attack_target, replace=False)
    mask = (df["label"] == 0) | df.index.isin(keep)
    return df[mask].reset_index(drop=True)


def temporal_split(df: pd.DataFrame, train=0.6, val=0.2):
    """Contiguous train / validation / test blocks in time order."""
    n = len(df)
    a, b = int(n * train), int(n * (train + val))
    return df.iloc[:a].copy(), df.iloc[a:b].copy(), df.iloc[b:].copy()
