from datetime import datetime, timezone

from tradegraph_financial_advisor.models.financial_data import NewsArticle
from tradegraph_financial_advisor.repositories import NewsRepository


def _sample_article(**overrides):
    base = {
        "title": "Sample headline",
        "url": "https://example.com/story",
        "content": "Detailed article body",
        "summary": "Summary",
        "source": "ExampleWire",
        "published_at": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "symbols": ["AAPL"],
    }
    base.update(overrides)
    return NewsArticle(**base)


def test_news_repository_upsert(tmp_path):
    repo = NewsRepository(db_path=tmp_path / "news.duckdb")

    first = _sample_article()
    repo.record_articles([first])

    # Update summary to ensure UPSERT semantics
    updated = _sample_article(summary="Updated summary")
    repo.record_articles([updated])

    rows = repo.fetch_recent_articles(limit=10)
    assert len(rows) == 1
    assert rows[0]["summary"] == "Updated summary"
    assert rows[0]["symbol"] == "AAPL"


def test_news_repository_handles_invalid_articles(tmp_path):
    repo = NewsRepository(db_path=tmp_path / "news.duckdb")

    inserted = repo.record_articles([None, {"title": "missing fields"}])

    assert inserted == 0
    assert repo.fetch_recent_articles(limit=10) == []
