"""Replay held-out flows into a running API so the dashboard has live data.

    python scripts/simulate.py --rate 300 --attack-rate 0.05 --seconds 120
"""
import argparse
import time

import httpx

from _common import load_replay, to_payload

ap = argparse.ArgumentParser()
ap.add_argument("--api", default="http://localhost:8000")
ap.add_argument("--rate", type=float, default=200, help="flows per second")
ap.add_argument("--batch", type=int, default=100)
ap.add_argument("--attack-rate", type=float, default=None, help="oversample attacks (demo only)")
ap.add_argument("--seconds", type=float, default=60)
ap.add_argument("--api-key", default=None)
a = ap.parse_args()

df = load_replay(attack_rate=a.attack_rate)
headers = {"X-API-Key": a.api_key} if a.api_key else {}
sent = alerts = 0
t0 = time.time()
with httpx.Client(base_url=a.api, headers=headers, timeout=30) as c:
    pos = 0
    while time.time() - t0 < a.seconds:
        chunk = df.iloc[pos:pos + a.batch]
        if chunk.empty:
            pos = 0
            continue
        r = c.post("/predict", json={"flows": to_payload(chunk, pos)})
        r.raise_for_status()
        sent += r.json()["count"]; alerts += r.json()["alerts"]; pos += a.batch
        time.sleep(max(0, sent / a.rate - (time.time() - t0)))
print(f"sent {sent:,} flows, {alerts:,} alerts in {time.time()-t0:.0f}s")
