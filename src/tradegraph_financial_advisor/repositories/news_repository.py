"""DuckDB-backed persistence for scraped news articles."""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import duckdb
from dateutil import parser as date_parser
from loguru import logger
from pydantic import ValidationError

from ..config.settings import settings
from ..models.financial_data import NewsArticle


class NewsRepository:
    """Simple repository that stores news articles inside DuckDB."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None) -> None:
        path = Path(db_path or settings.news_db_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = path
        self._schema_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._schema_lock:
            with duckdb.connect(str(self.db_path)) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS news_articles (
                        symbol TEXT,
                        title TEXT,
                        url TEXT,
                        summary TEXT,
                        content TEXT,
                        source TEXT,
                        published_at TIMESTAMP,
                        scraped_at TIMESTAMP,
                        symbols TEXT,
                        sentiment TEXT,
                        impact_score DOUBLE
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_news_articles_symbol_title_url
                    ON news_articles(symbol, title, url);
                    """
                )

    def record_articles(
        self, articles: Sequence[Union[NewsArticle, Dict[str, Any]]]
    ) -> int:
        rows: List[tuple] = []
        scraped_at = datetime.utcnow()
        for article in articles:
            model = self._coerce_article(article)
            if not model:
                continue
            primary_symbol = model.symbols[0] if model.symbols else None
            rows.append(
                (
                    primary_symbol,
                    model.title.strip(),
                    model.url,
                    (model.summary or "").strip() or None,
                    model.content,
                    model.source,
                    self._normalize_datetime(model.published_at),
                    scraped_at,
                    ",".join(model.symbols),
                    self._normalize_sentiment(model.sentiment),
                    model.impact_score,
                )
            )

        if not rows:
            return 0

        insert_sql = """
            INSERT INTO news_articles (
                symbol,
                title,
                url,
                summary,
                content,
                source,
                published_at,
                scraped_at,
                symbols,
                sentiment,
                impact_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, title, url) DO UPDATE SET
                summary=excluded.summary,
                content=excluded.content,
                source=excluded.source,
                published_at=excluded.published_at,
                scraped_at=excluded.scraped_at,
                symbols=excluded.symbols,
                sentiment=excluded.sentiment,
                impact_score=excluded.impact_score;
        """

        with self._write_lock:
            try:
                with duckdb.connect(str(self.db_path)) as conn:
                    conn.executemany(insert_sql, rows)
            except Exception as exc:  # pragma: no cover - disk/config issues
                logger.warning(f"Failed to persist news articles: {exc}")
                return 0

        return len(rows)

    def fetch_recent_articles(self, limit: int = 50) -> List[Dict[str, Any]]:
        query = (
            "SELECT symbol, title, url, summary, source, published_at, scraped_at, symbols, sentiment, impact_score "
            "FROM news_articles ORDER BY COALESCE(published_at, scraped_at) DESC LIMIT ?"
        )
        with duckdb.connect(str(self.db_path)) as conn:
            rows = conn.execute(query, [limit]).fetchall()
            columns = [desc[0] for desc in conn.description]

        return [dict(zip(columns, row)) for row in rows]

    def _coerce_article(
        self, article: Union[NewsArticle, Dict[str, Any], None]
    ) -> Optional[NewsArticle]:
        if article is None:
            return None
        if isinstance(article, NewsArticle):
            return article
        if isinstance(article, dict):
            try:
                return NewsArticle(**article)
            except ValidationError as exc:
                logger.warning(f"Invalid news article payload skipped: {exc}")
                return None
        logger.warning("Unsupported news article type {}", type(article))
        return None

    @staticmethod
    def _normalize_datetime(value: Any) -> Optional[datetime]:
        if isinstance(value, datetime):
            return value
        if not value:
            return None
        if isinstance(value, str):
            try:
                return date_parser.parse(value)
            except (ValueError, TypeError):
                return None
        return None

    @staticmethod
    def _normalize_sentiment(value: Any) -> Optional[str]:
        if value is None:
            return None
        if hasattr(value, "value"):
            return str(value.value)
        return str(value)


__all__ = ["NewsRepository"]
