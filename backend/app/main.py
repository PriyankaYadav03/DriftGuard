from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import Integer, func, insert, select, text

from ml.schema import FEATURES

from .config import settings
from .db import Prediction, SessionLocal, engine, init_db, now
from .model_service import ModelService, severity
from .schemas import FlowResult, PredictRequest, PredictResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    app.state.model = ModelService(settings.model_path)
    yield


app = FastAPI(title="DriftGuard API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins,
                   allow_methods=["*"], allow_headers=["*"])


def require_key(x_api_key: str | None = Header(None)):
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(401, "invalid or missing API key")


def _score_and_store(flows, model: ModelService):
    df = pd.DataFrame([f.model_dump() for f in flows])
    raw, prob, alert = model.score(df)
    ts = now()
    version = model.version
    rows = []
    feats = df[FEATURES].to_dict("records")
    for i, f in enumerate(flows):
        rows.append({"ts": ts, "minute": int(ts // 60), "flow_id": f.flow_id, "score": float(raw[i]),
                     "probability": float(prob[i]), "is_alert": bool(alert[i]),
                     "model_version": version, "features": feats[i], "true_label": f.label})
    with SessionLocal() as s:
        s.execute(insert(Prediction), rows)     # one bulk insert per batch
        s.commit()
    return raw, prob, alert


@app.get("/health")
def health():
    try:
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False
    return {"status": "ok" if db_ok else "degraded", "db": db_ok, "model": app.state.model.version}


@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(require_key)])
async def predict(req: PredictRequest):
    if len(req.flows) > settings.max_batch:
        raise HTTPException(413, f"batch too large (max {settings.max_batch})")
    t = time.perf_counter()
    model: ModelService = app.state.model
    raw, prob, alert = await run_in_threadpool(_score_and_store, req.flows, model)
    results = [FlowResult(index=i, flow_id=f.flow_id, score=round(float(raw[i]), 5),
                          probability=round(float(prob[i]), 5), is_alert=bool(alert[i]),
                          severity=severity(float(prob[i]), bool(alert[i])))
               for i, f in enumerate(req.flows)]
    return PredictResponse(model_version=model.version, count=len(results), alerts=int(alert.sum()),
                           latency_ms=round((time.perf_counter() - t) * 1000, 2), results=results)


@app.get("/predictions", dependencies=[Depends(require_key)])
def predictions(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0),
                alerts_only: bool = False):
    q = select(Prediction).order_by(Prediction.id.desc()).limit(limit).offset(offset)
    if alerts_only:
        q = q.where(Prediction.is_alert.is_(True))
    with SessionLocal() as s:
        rows = s.scalars(q).all()
    return [{"id": r.id, "ts": r.ts, "flow_id": r.flow_id, "score": round(r.score, 5),
             "probability": round(r.probability, 5), "is_alert": r.is_alert,
             "severity": severity(r.probability, r.is_alert), "model_version": r.model_version,
             "dst_port": r.features.get("dst_port"), "protocol": r.features.get("protocol"),
             "features": r.features, "true_label": r.true_label} for r in rows]


@app.get("/stats", dependencies=[Depends(require_key)])
def stats(minutes: int = Query(30, ge=1, le=1440)):
    cutoff = int(now() // 60) - minutes
    with SessionLocal() as s:
        total, alerts = s.execute(select(func.count(Prediction.id),
                                         func.coalesce(func.sum(Prediction.is_alert.cast(Integer)), 0))).one()
        series = s.execute(
            select(Prediction.minute, func.count(Prediction.id),
                   func.sum(Prediction.is_alert.cast(Integer)))
            .where(Prediction.minute >= cutoff).group_by(Prediction.minute).order_by(Prediction.minute)).all()
        window_total = sum(r[1] for r in series)
        window_alerts = sum(int(r[2] or 0) for r in series)
        hist_rows = s.execute(select(func.floor(Prediction.probability * 10), func.count(Prediction.id))
                              .where(Prediction.minute >= cutoff).group_by(func.floor(Prediction.probability * 10))).all()
        top_ports = s.execute(
            select(Prediction.features["dst_port"].as_integer(), func.count(Prediction.id))
            .where(Prediction.is_alert.is_(True), Prediction.minute >= cutoff)
            .group_by(Prediction.features["dst_port"].as_integer())
            .order_by(func.count(Prediction.id).desc()).limit(8)).all()
    hist = [0] * 10
    for b, c in hist_rows:
        hist[min(int(b), 9)] += c
    return {
        "total_flows": int(total), "total_alerts": int(alerts or 0),
        "window_minutes": minutes, "window_flows": window_total, "window_alerts": window_alerts,
        "window_alert_rate": round(window_alerts / window_total, 5) if window_total else 0.0,
        "per_minute": [{"minute": int(m) * 60, "flows": int(c), "alerts": int(a or 0)} for m, c, a in series],
        "probability_histogram": [{"bin": f"{i/10:.1f}-{(i+1)/10:.1f}", "count": c} for i, c in enumerate(hist)],
        "top_alert_ports": [{"port": int(p), "alerts": int(c)} for p, c in top_ports if p is not None],
    }


@app.get("/model/info", dependencies=[Depends(require_key)])
def model_info():
    return app.state.model.info


@app.get("/model/metrics", dependencies=[Depends(require_key)])
def model_metrics():
    if not settings.metrics_path.exists():
        raise HTTPException(404, "metrics.json not found; run `python -m ml.train`")
    return json.loads(settings.metrics_path.read_text())

