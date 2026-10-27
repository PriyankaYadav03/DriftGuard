"""
Synthetic network-flow generator for DriftGuard.

Produces CICFlowMeter-style flow records: one benign class and five attack
families. It exists so the whole pipeline (train, evaluate, dashboard) runs end
to end with no multi-gigabyte download.

IMPORTANT: this is not real traffic. The metrics it produces show that the
pipeline works, not how well a detector would do on a real network. To use a
real labeled dataset (CIC-IDS2017, NSL-KDD, UNSW-NB15 ...) pass it to
`train.py --data`.

Design notes
- Attack families overlap with benign traffic on purpose (slow DoS looks like an
  idle session, brute force looks like SSH, web attacks look like web browsing).
- STEALTH_RATE of attack flows are drawn from the benign distribution but keep
  an attack label, so recall has a ceiling below 100% like it would in practice.

Usage:
    python data_gen.py --rows 40000 --seed 7 --out data/flows.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [
    "dst_port", "protocol", "flow_duration_ms",
    "fwd_packets", "bwd_packets", "fwd_bytes", "bwd_bytes",
    "pkt_len_mean", "pkt_len_std",
    "flow_bytes_per_s", "flow_pkts_per_s", "fwd_iat_mean_ms",
    "syn_flags", "ack_flags", "rst_flags", "psh_flags",
    "init_win_bytes", "active_mean_ms", "idle_mean_ms",
]

CLASS_MIX = {
    "Benign": 0.80,
    "DoS": 0.06,
    "PortScan": 0.05,
    "BruteForce": 0.035,
    "WebAttack": 0.025,
    "Botnet": 0.03,
}
STEALTH_RATE = 0.08  # share of attack flows that look exactly like benign ones


def _arr(x, n: int) -> np.ndarray:
    """Broadcast a scalar or length-n array to a float array of length n."""
    return np.broadcast_to(np.asarray(x, dtype=float), (n,)).copy()


def _build(rng, n, *, port, proto, dur_ms, fwd_pk, bwd_pk, fwd_sz, bwd_sz,
           syn_frac, ack_p, rst_p, psh_frac, win, active_mu, idle_mu, idle_p):
    """Turn per-class draws into the full feature table (derived fields included)."""
    port = np.clip(_arr(port, n), 1, 65535).astype(int)
    proto = _arr(proto, n).astype(int)
    tcp = proto == 6

    fwd_pk = np.maximum(1, np.round(_arr(fwd_pk, n))).astype(int)
    bwd_pk = np.maximum(0, np.round(_arr(bwd_pk, n))).astype(int)
    fwd_sz = np.clip(_arr(fwd_sz, n), 40, 1500)
    bwd_sz = np.clip(_arr(bwd_sz, n), 40, 1500)
    dur_ms = np.maximum(_arr(dur_ms, n), 0.05)

    fwd_bytes = fwd_pk * fwd_sz
    bwd_bytes = bwd_pk * bwd_sz
    total_pk = fwd_pk + bwd_pk
    total_bytes = fwd_bytes + bwd_bytes
    secs = dur_ms / 1000.0

    mean_len = total_bytes / total_pk
    spread = np.where(bwd_pk > 0, np.abs(fwd_sz - bwd_sz) * 0.5, 0.0)
    std_len = spread + mean_len * rng.uniform(0.05, 0.4, n)
    fwd_iat = dur_ms / np.maximum(fwd_pk - 1, 1) * rng.uniform(0.8, 1.2, n)

    def flag(count, p):
        p = np.clip(_arr(p, n), 0, 1)
        return np.where(tcp, rng.binomial(count, p), 0)

    syn = flag(fwd_pk, syn_frac)
    ack = flag(total_pk, ack_p)
    rst = flag(total_pk, rst_p)
    psh = flag(total_pk, psh_frac)

    idle = np.where(rng.random(n) < idle_p, rng.lognormal(idle_mu, 1.2, n), 0.0)
    active = rng.lognormal(active_mu, 1.0, n)

    return pd.DataFrame({
        "dst_port": port,
        "protocol": proto,
        "flow_duration_ms": dur_ms.round(3),
        "fwd_packets": fwd_pk,
        "bwd_packets": bwd_pk,
        "fwd_bytes": fwd_bytes.round(0).astype(int),
        "bwd_bytes": bwd_bytes.round(0).astype(int),
        "pkt_len_mean": mean_len.round(2),
        "pkt_len_std": std_len.round(2),
        "flow_bytes_per_s": (total_bytes / secs).round(2),
        "flow_pkts_per_s": (total_pk / secs).round(2),
        "fwd_iat_mean_ms": fwd_iat.round(3),
        "syn_flags": syn,
        "ack_flags": ack,
        "rst_flags": rst,
        "psh_flags": psh,
        "init_win_bytes": np.where(tcp, _arr(win, n), 0).astype(int),
        "active_mean_ms": active.round(3),
        "idle_mean_ms": idle.round(3),
    })[FEATURES]


# --------------------------------------------------------------------------- #
# Traffic families
# --------------------------------------------------------------------------- #

def _benign(rng, n):
    ports = np.array([443, 80, 53, 22, 25, 8080, 3389, 123, 445, 993, 5432])
    p = np.array([.34, .16, .16, .05, .03, .04, .03, .06, .04, .04, .05])
    port = rng.choice(ports, n, p=p / p.sum())
    proto = np.where(np.isin(port, [53, 123]), 17, 6)
    udp = proto == 17

    dur = np.where(udp, rng.lognormal(1.5, 1.0, n), rng.lognormal(5.0, 1.9, n))
    fwd_pk = np.where(udp, 1, 1 + rng.poisson(rng.lognormal(1.6, 1.0, n)))
    bwd_pk = np.where(udp, rng.integers(0, 2, n), rng.poisson(rng.lognormal(1.8, 1.0, n)))
    return _build(
        rng, n, port=port, proto=proto, dur_ms=dur, fwd_pk=fwd_pk, bwd_pk=bwd_pk,
        fwd_sz=rng.lognormal(5.2, 0.7, n), bwd_sz=rng.lognormal(6.0, 0.9, n),
        syn_frac=0.08, ack_p=rng.uniform(0.6, 0.95, n), rst_p=0.01, psh_frac=0.3,
        win=rng.choice([8192, 29200, 64240, 65535], n, p=[.1, .25, .4, .25]),
        active_mu=3.0, idle_mu=5.5, idle_p=0.25,
    )


def _dos(rng, n):
    flood = rng.random(n) < 0.7  # the rest are slow, slowloris-style
    port = rng.choice([80, 443, 8080], n, p=[.5, .4, .1])
    return _build(
        rng, n, port=port, proto=6,
        dur_ms=np.where(flood, rng.lognormal(3.3, 1.2, n), rng.lognormal(9.5, 0.8, n)),
        fwd_pk=np.where(flood, 15 + rng.poisson(rng.lognormal(3.2, 0.8, n)), 4 + rng.poisson(8, n)),
        bwd_pk=np.where(flood, rng.poisson(0.8, n), rng.poisson(2, n)),
        fwd_sz=np.where(flood, rng.lognormal(4.2, 0.35, n), rng.lognormal(4.8, 0.5, n)),
        bwd_sz=rng.lognormal(4.5, 0.5, n),
        syn_frac=np.where(flood, 0.6, 0.15), ack_p=rng.uniform(0.1, 0.7, n),
        rst_p=0.12, psh_frac=np.where(flood, 0.05, 0.1),
        win=rng.choice([0, 1024, 8192], n, p=[.4, .3, .3]),
        active_mu=2.0, idle_mu=4.0, idle_p=0.1,
    )


def _portscan(rng, n):
    common = rng.random(n) < 0.25  # some probes hit well-known ports
    port = np.where(common, rng.choice([22, 80, 443, 445, 3389, 8080], n),
                    rng.integers(1, 65536, n))
    return _build(
        rng, n, port=port, proto=6,
        dur_ms=rng.lognormal(0.6, 1.3, n),
        fwd_pk=rng.choice([1, 2, 3], n, p=[.6, .3, .1]),
        bwd_pk=rng.choice([0, 1], n, p=[.5, .5]),
        fwd_sz=rng.lognormal(3.95, 0.12, n), bwd_sz=rng.lognormal(3.95, 0.12, n),
        syn_frac=1.0, ack_p=rng.uniform(0.0, 0.2, n), rst_p=0.3, psh_frac=0.0,
        win=rng.choice([1024, 2048, 4096, 29200], n, p=[.4, .2, .2, .2]),
        active_mu=0.2, idle_mu=3.0, idle_p=0.0,
    )


def _bruteforce(rng, n):
    port = rng.choice([22, 21, 3389, 445], n, p=[.55, .2, .2, .05])
    return _build(
        rng, n, port=port, proto=6,
        dur_ms=rng.lognormal(6.4, 0.7, n),
        fwd_pk=8 + rng.poisson(rng.lognormal(1.8, 0.5, n)),
        bwd_pk=8 + rng.poisson(rng.lognormal(1.8, 0.5, n)),
        fwd_sz=rng.lognormal(4.7, 0.3, n), bwd_sz=rng.lognormal(4.9, 0.4, n),
        syn_frac=0.1, ack_p=rng.uniform(0.7, 0.95, n), rst_p=0.04, psh_frac=0.55,
        win=rng.choice([29200, 64240, 65535], n),
        active_mu=3.0, idle_mu=4.5, idle_p=0.1,
    )


def _webattack(rng, n):
    port = rng.choice([80, 443, 8080], n, p=[.45, .4, .15])
    return _build(
        rng, n, port=port, proto=6,
        dur_ms=rng.lognormal(5.6, 1.0, n),
        fwd_pk=3 + rng.poisson(rng.lognormal(1.5, 0.6, n)),
        bwd_pk=rng.poisson(rng.lognormal(1.5, 0.6, n)),
        fwd_sz=rng.lognormal(6.0, 0.7, n), bwd_sz=rng.lognormal(6.3, 0.8, n),
        syn_frac=0.1, ack_p=rng.uniform(0.6, 0.95, n), rst_p=0.02, psh_frac=0.45,
        win=rng.choice([29200, 64240, 65535], n),
        active_mu=3.0, idle_mu=5.0, idle_p=0.2,
    )


def _botnet(rng, n):
    port = rng.choice([8080, 4444, 6667, 443, 8443], n, p=[.25, .15, .15, .3, .15])
    return _build(
        rng, n, port=port, proto=6,
        dur_ms=rng.lognormal(6.2, 0.25, n),
        fwd_pk=4 + rng.poisson(2, n), bwd_pk=3 + rng.poisson(2, n),
        fwd_sz=rng.lognormal(4.9, 0.15, n), bwd_sz=rng.lognormal(5.3, 0.2, n),
        syn_frac=0.15, ack_p=rng.uniform(0.7, 0.95, n), rst_p=0.01, psh_frac=0.3,
        win=rng.choice([64240, 65535], n),
        active_mu=3.0, idle_mu=8.0, idle_p=0.9,
    )


_GENERATORS = {
    "Benign": _benign, "DoS": _dos, "PortScan": _portscan,
    "BruteForce": _bruteforce, "WebAttack": _webattack, "Botnet": _botnet,
}


def generate_flows(n_rows: int = 40_000, seed: int = 7) -> pd.DataFrame:
    """Return a shuffled table of FEATURES + `attack_type` + `label` (0 benign, 1 malicious)."""
    rng = np.random.default_rng(seed)
    names = list(CLASS_MIX)
    counts = rng.multinomial(n_rows, [CLASS_MIX[k] for k in names])

    parts = []
    for name, n in zip(names, counts):
        if n == 0:
            continue
        df = _GENERATORS[name](rng, int(n))
        if name != "Benign":
            stealth = rng.random(n) < STEALTH_RATE
            if stealth.any():
                mimic = _benign(rng, int(stealth.sum()))
                df.loc[stealth, FEATURES] = mimic[FEATURES].to_numpy()
        df["attack_type"] = name
        df["label"] = 0 if name == "Benign" else 1
        parts.append(df)

    out = pd.concat(parts, ignore_index=True)
    out = out.iloc[rng.permutation(len(out))].reset_index(drop=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=40_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/flows.csv")
    args = ap.parse_args()

    df = generate_flows(args.rows, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df):,} flows to {out}")
    print(df["attack_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
