"""Measure API throughput and latency.

    python scripts/loadtest.py --concurrency 8 --batch 500 --seconds 20
"""
import argparse
import asyncio
import statistics
import time

import httpx

from _common import load_replay, to_payload

ap = argparse.ArgumentParser()
ap.add_argument("--api", default="http://localhost:8000")
ap.add_argument("--concurrency", type=int, default=8)
ap.add_argument("--batch", type=int, default=500)
ap.add_argument("--seconds", type=float, default=20)
a = ap.parse_args()

payload = {"flows": to_payload(load_replay().iloc[: a.batch])}
lat, flows, errors = [], 0, 0


async def worker(client, stop):
    global flows, errors
    while time.time() < stop:
        t = time.perf_counter()
        try:
            r = await client.post("/predict", json=payload)
            r.raise_for_status()
            lat.append(time.perf_counter() - t); flows += a.batch
        except Exception:
            errors += 1


async def main():
    async with httpx.AsyncClient(base_url=a.api, timeout=60) as c:
        await c.post("/predict", json=payload)   # warm-up
        t0 = time.time(); stop = t0 + a.seconds
        await asyncio.gather(*[worker(c, stop) for _ in range(a.concurrency)])
        dur = time.time() - t0
    q = statistics.quantiles(lat, n=100) if len(lat) >= 100 else [max(lat)] * 99
    print(f"batch={a.batch} concurrency={a.concurrency} duration={dur:.1f}s")
    print(f"throughput : {flows/dur:,.0f} flows/s  ({len(lat)/dur:.1f} req/s)")
    print(f"latency/req: p50={statistics.median(lat)*1000:.0f}ms p95={q[94]*1000:.0f}ms p99={q[98]*1000:.0f}ms")
    print(f"errors     : {errors}")

asyncio.run(main())
