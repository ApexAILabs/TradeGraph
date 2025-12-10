from typing import List, Optional
import os

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", case_sensitive=False, extra="ignore"
    )

    openai_api_key: str = Field(default="")
    finnhub_api_key: str = Field(default="")
    alpha_vantage_api_key: Optional[str] = Field(default=None)
    financial_data_api_key: Optional[str] = Field(default=None)

    log_level: str = Field(default="INFO")
    max_concurrent_agents: int = Field(default=5)
    analysis_timeout_seconds: int = Field(default=30)

    news_sources: List[str] = Field(
        default_factory=lambda: [
            "bloomberg",
            "reuters",
            "yahoo-finance",
            "marketwatch",
            "cnbc",
        ]
    )
    analysis_depth: str = Field(default="detailed")
    default_portfolio_size: float = Field(default=100000.0)
    news_db_path: str = Field(default="tradegraph.duckdb")


settings = Settings()


def refresh_openai_api_key() -> None:
    """Reload API keys from the environment at runtime."""

    load_dotenv(override=True)
    for env_var, attr in (
        ("OPENAI_API_KEY", "openai_api_key"),
        ("FINNHUB_API_KEY", "finnhub_api_key"),
    ):
        value = (os.getenv(env_var) or "").strip()
        if value:
            setattr(settings, attr, value)
