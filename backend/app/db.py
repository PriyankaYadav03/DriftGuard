from __future__ import annotations

import time

from sqlalchemy import (JSON, Boolean, Float, Index, Integer, String, create_engine)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from .config import settings

_is_sqlite = settings.database_url.startswith("sqlite")
engine = create_engine(
    settings.database_url, pool_pre_ping=True,
    connect_args={"check_same_thread": False} if _is_sqlite else {},
)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[float] = mapped_column(Float, index=True)            # unix seconds, server time
    minute: Mapped[int] = mapped_column(Integer)                    # ts // 60, for cheap bucketing
    flow_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    score: Mapped[float] = mapped_column(Float)                     # raw model score
    probability: Mapped[float] = mapped_column(Float)               # calibrated attack probability
    is_alert: Mapped[bool] = mapped_column(Boolean)
    model_version: Mapped[str] = mapped_column(String(40))
    features: Mapped[dict] = mapped_column(JSON)
    true_label: Mapped[int | None] = mapped_column(Integer, nullable=True)  # filled when labels arrive
    __table_args__ = (Index("ix_pred_minute", "minute"), Index("ix_pred_alert_ts", "is_alert", "ts"))


def init_db() -> None:
    Base.metadata.create_all(engine)


def now() -> float:
    return time.time()
