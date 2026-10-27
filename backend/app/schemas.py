from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Flow(BaseModel):
    """One network flow in CICFlowMeter-style features."""
    model_config = ConfigDict(extra="forbid")
    flow_id: str | None = Field(None, max_length=64)
    label: int | None = Field(None, ge=0, le=1, description="ground truth, if already known")

    dst_port: int = Field(ge=0, le=65535)
    protocol: int = Field(ge=0, le=255)
    flow_duration_ms: float = Field(ge=0)
    fwd_packets: int = Field(ge=0)
    bwd_packets: int = Field(ge=0)
    fwd_bytes: float = Field(ge=0)
    bwd_bytes: float = Field(ge=0)
    pkt_len_mean: float = Field(ge=0)
    pkt_len_std: float = Field(ge=0)
    flow_bytes_per_s: float = Field(ge=0)
    flow_pkts_per_s: float = Field(ge=0)
    fwd_iat_mean_ms: float = Field(ge=0)
    syn_flags: int = Field(ge=0)
    ack_flags: int = Field(ge=0)
    rst_flags: int = Field(ge=0)
    psh_flags: int = Field(ge=0)
    init_win_bytes: int = Field(ge=0)
    active_mean_ms: float = Field(ge=0)
    idle_mean_ms: float = Field(ge=0)


class PredictRequest(BaseModel):
    flows: list[Flow] = Field(min_length=1)


class FlowResult(BaseModel):
    index: int
    flow_id: str | None
    score: float
    probability: float
    is_alert: bool
    severity: str   # none | low | medium | high


class PredictResponse(BaseModel):
    model_version: str
    count: int
    alerts: int
    latency_ms: float
    results: list[FlowResult]
