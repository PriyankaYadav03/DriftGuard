"""Feature engineering. Used by BOTH training and the API so they cannot drift apart."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import FEATURES

ENGINEERED = [
    "total_packets", "total_bytes", "fwd_bwd_pkt_ratio", "fwd_bwd_byte_ratio",
    "syn_ratio", "ack_ratio", "rst_ratio", "psh_ratio", "is_udp", "well_known_port",
]
MODEL_FEATURES = FEATURES + ENGINEERED


def build_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Raw flow table -> model input. Missing/inf values are made safe here."""
    out = df[FEATURES].astype("float64").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    pk = (out["fwd_packets"] + out["bwd_packets"]).clip(lower=1)
    out["total_packets"] = out["fwd_packets"] + out["bwd_packets"]
    out["total_bytes"] = out["fwd_bytes"] + out["bwd_bytes"]
    out["fwd_bwd_pkt_ratio"] = out["fwd_packets"] / (out["bwd_packets"] + 1.0)
    out["fwd_bwd_byte_ratio"] = out["fwd_bytes"] / (out["bwd_bytes"] + 1.0)
    out["syn_ratio"] = out["syn_flags"] / pk
    out["ack_ratio"] = out["ack_flags"] / pk
    out["rst_ratio"] = out["rst_flags"] / pk
    out["psh_ratio"] = out["psh_flags"] / pk
    out["is_udp"] = (out["protocol"] == 17).astype("float64")
    out["well_known_port"] = (out["dst_port"] < 1024).astype("float64")
    return out[MODEL_FEATURES]
