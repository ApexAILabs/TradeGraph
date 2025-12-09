from typing import List, Optional
import os

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    openai_api_key: str = Field("", env="OPENAI_API_KEY")
    finnhub_api_key: str = Field("", env="FINNHUB_API_KEY")
    alpha_vantage_api_key: Optional[str] = Field(None, env="ALPHA_VANTAGE_API_KEY")
    financial_data_api_key: Optional[str] = Field(None, env="FINANCIAL_DATA_API_KEY")

    log_level: str = Field("INFO", env="LOG_LEVEL")
    max_concurrent_agents: int = Field(5, env="MAX_CONCURRENT_AGENTS")
    analysis_timeout_seconds: int = Field(30, env="ANALYSIS_TIMEOUT_SECONDS")

    news_sources: List[str] = Field(
        default_factory=lambda: [
            "bloomberg",
            "reuters",
            "yahoo-finance",
            "marketwatch",
            "cnbc",
        ],
        env="NEWS_SOURCES",
    )
    analysis_depth: str = Field("detailed", env="ANALYSIS_DEPTH")
    default_portfolio_size: float = Field(100000.0, env="DEFAULT_PORTFOLIO_SIZE")

    model_config = {"env_file": ".env", "case_sensitive": False, "extra": "ignore"}

    @classmethod
    def get_news_sources_list(cls, v: str) -> List[str]:
        if isinstance(v, str):
            return [source.strip() for source in v.split(",")]
        return v


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
