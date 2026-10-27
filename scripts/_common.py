import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ml.schema import FEATURES  # noqa: E402

INT_COLS = {"dst_port", "protocol", "fwd_packets", "bwd_packets", "syn_flags", "ack_flags",
            "rst_flags", "psh_flags", "init_win_bytes"}


def load_replay(path=None, attack_rate=None, seed=0) -> pd.DataFrame:
    df = pd.read_csv(path or ROOT / "artifacts" / "replay.csv")
    if attack_rate is not None:   # demo only: oversample attacks so the dashboard has something to show
        rng = np.random.default_rng(seed)
        att, ben = df[df.label == 1], df[df.label == 0]
        n_att = int(len(df) * attack_rate)
        df = pd.concat([ben.sample(len(df) - n_att, replace=True, random_state=seed),
                        att.sample(n_att, replace=True, random_state=seed)]).sample(frac=1, random_state=seed)
    return df.reset_index(drop=True)


def to_payload(df: pd.DataFrame, start=0, with_labels=False) -> list[dict]:
    out = []
    for i, r in enumerate(df[FEATURES + ["label"]].to_dict("records")):
        f = {k: (int(r[k]) if k in INT_COLS else float(r[k])) for k in FEATURES}
        f["flow_id"] = f"f{start + i}"
        if with_labels:
            f["label"] = int(r["label"])
        out.append(f)
    return out
