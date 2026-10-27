import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class Settings:
    database_url: str = os.getenv("DATABASE_URL", f"sqlite:///{ROOT / 'driftguard.db'}")
    model_path: Path = Path(os.getenv("MODEL_PATH", ROOT / "artifacts" / "model.joblib"))
    metrics_path: Path = Path(os.getenv("METRICS_PATH", ROOT / "artifacts" / "metrics.json"))
    api_key: str | None = os.getenv("API_KEY") or None      # unset = auth disabled (dev only)
    max_batch: int = int(os.getenv("MAX_BATCH", "5000"))
    cors_origins: list[str] = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")


settings = Settings()
