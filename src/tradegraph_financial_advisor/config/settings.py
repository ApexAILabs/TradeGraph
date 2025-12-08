from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    openai_api_key: str = Field("", env="OPENAI_API_KEY")
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

    # Database Configuration
    neo4j_uri: str = Field("bolt://localhost:7687", env="NEO4J_URI")
    neo4j_user: str = Field("neo4j", env="NEO4J_USER")
    neo4j_password: str = Field("password", env="NEO4J_PASSWORD")
    duckdb_path: str = Field("tradegraph.duckdb", env="DUCKDB_PATH")

    model_config = {"env_file": ".env", "case_sensitive": False}

    @classmethod
    def get_news_sources_list(cls, v: str) -> List[str]:
        if isinstance(v, str):
            return [source.strip() for source in v.split(",")]
        return v


settings = Settings()
