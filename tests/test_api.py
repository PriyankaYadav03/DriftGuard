from conftest import GOOD_FLOW, SCAN_FLOW


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_batch_and_store(client):
    r = client.post("/predict", json={"flows": [dict(GOOD_FLOW, flow_id="a"), dict(SCAN_FLOW, flow_id="b")]})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2 and len(body["results"]) == 2
    by_id = {x["flow_id"]: x for x in body["results"]}
    assert by_id["b"]["probability"] > by_id["a"]["probability"]   # scan looks worse than normal HTTPS
    assert 0 <= by_id["a"]["probability"] <= 1
    stored = client.get("/predictions?limit=10").json()
    assert {"a", "b"} <= {p["flow_id"] for p in stored}


def test_rejects_bad_input(client):
    assert client.post("/predict", json={"flows": []}).status_code == 422
    assert client.post("/predict", json={"flows": [dict(GOOD_FLOW, dst_port=70000)]}).status_code == 422
    assert client.post("/predict", json={"flows": [dict(GOOD_FLOW, fwd_bytes=-1)]}).status_code == 422
    assert client.post("/predict", json={"flows": [dict(GOOD_FLOW, surprise=1)]}).status_code == 422


def test_batch_limit(client, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "max_batch", 2)
    r = client.post("/predict", json={"flows": [GOOD_FLOW] * 3})
    assert r.status_code == 413


def test_stats_and_metrics(client):
    client.post("/predict", json={"flows": [GOOD_FLOW, SCAN_FLOW, SCAN_FLOW]})
    s = client.get("/stats?minutes=60").json()
    assert s["total_flows"] >= 3 and s["window_flows"] >= 3
    assert sum(b["count"] for b in s["probability_histogram"]) == s["window_flows"]
    assert client.get("/predictions?alerts_only=true").status_code == 200
    m = client.get("/model/metrics").json()
    assert "comparison" in m and "test" in m
    assert client.get("/model/info").json()["version"].startswith("v1-")


def test_api_key_enforced(client, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "api_key", "secret")
    assert client.get("/stats").status_code == 401
    assert client.get("/stats", headers={"X-API-Key": "secret"}).status_code == 200
