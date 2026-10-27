# DriftGuard

ML network-intrusion detection with a live dashboard. **V1 (this repo): static model.**
V2 (planned): drift monitoring, safe retraining, rollback.

```
flows ──► FastAPI /predict ──► model (HistGradientBoosting + calibration) ──► DB ──► React dashboard
                                   ▲
                    python -m ml.train  (time-ordered split, imbalance comparison, metrics.json)
```

## Run it (local, no Docker)
```bash
pip install -r requirements-ml.txt -r backend/requirements.txt
python -m ml.train                      # ~35 s; writes artifacts/{model.joblib,metrics.json,replay.csv}
PYTHONPATH=.:backend uvicorn app.main:app --port 8000 --workers 2
cd frontend && npm install && npm run dev          # http://localhost:5173
python scripts/simulate.py --rate 250 --attack-rate 0.05   # stream demo traffic
pytest                                  # 13 tests
```
Docker: `python -m ml.train && docker compose up --build` → dashboard on :8080 (see status below).

## API
| | |
|---|---|
| `POST /predict` | batch of flows (≤5000) → score, calibrated probability, alert, severity; stored |
| `GET /predictions?alerts_only=` | recent scored flows |
| `GET /stats?minutes=` | volumes, alerts/min, probability histogram, top alerted ports |
| `GET /model/metrics`, `/model/info` | evaluation results, deployed model |
| `GET /health` | DB + model status |
Set `API_KEY` to require an `X-API-Key` header (the dashboard's nginx adds it, so browsers never hold the key).

## Design decisions
- **Imbalance:** training data is thinned to 2% attacks. Strategies compared on one split: baseline, class
  weights, undersampling, SMOTE, each with/without a threshold tuned to a 1% false-positive budget. Resampling touches
  the training block only. Validation/test keep the natural rate. Probabilities are isotonic-calibrated for display.
- **Train/serve parity:** `ml/features.py` is the only feature code, used by both training and the API.
- **Dataset:** `ml/data.py` has a CIC-IDS2017 loader (`python -m ml.train --cic file1.csv file2.csv`).

## Status and limits (read before quoting numbers)
- **Results are on synthetic data** (`data_gen.py`). They show the pipeline works, not real-world accuracy.
  Strategy differences are small here (PR-AUC 0.851–0.861); threshold tuning mattered most.
- The CIC-IDS2017 loader is **untested on the real files** (no download access when built). Check `df.describe()` first.
- On the synthetic data a time-ordered split equals a random split (rows are i.i.d.); it matters on real data and in V2.
- **Throughput:** ~14k flows/s (batch 500, 4 clients), p95 ≈ 210 ms, on a 2-core machine that also ran the load generator,
  with SQLite. Batch 50: ~2.4k flows/s. Run `scripts/loadtest.py` on your hardware. This is flow-level scoring, not packet capture.
- **Docker/Postgres files are written but untested** (no Docker here); SQL is portable but unverified on Postgres.
- Not in V1: message queue (Kafka/Redis Streams), auth beyond a shared key, retention/partitioning, multi-class attack-type prediction, drift handling.
- The model file is a pickle: only load artifacts you trained yourself.
