import duckdb
from neo4j import GraphDatabase
import logging
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from ..config.settings import settings

logger = logging.getLogger(__name__)

class DBManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DBManager, cls).__new__(cls)
            cls._instance.duckdb_conn = None
            cls._instance.neo4j_driver = None
            cls._instance._initialize_duckdb()
            cls._instance._initialize_neo4j()
        return cls._instance

    def _initialize_duckdb(self):
        try:
            self.duckdb_conn = duckdb.connect(settings.duckdb_path)
            # Create tables if they don't exist
            self.duckdb_conn.execute("""
                CREATE TABLE IF NOT EXISTS stock_prices (
                    symbol VARCHAR,
                    date TIMESTAMP,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    volume BIGINT,
                    PRIMARY KEY (symbol, date)
                )
            """)
            self.duckdb_conn.execute("""
                CREATE TABLE IF NOT EXISTS user_queries (
                    id UUID DEFAULT gen_random_uuid(),
                    query_text VARCHAR,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    response_summary VARCHAR
                )
            """)
            logger.info(f"DuckDB initialized at {settings.duckdb_path}")
        except Exception as e:
            logger.error(f"Failed to initialize DuckDB: {e}")

    def _initialize_neo4j(self):
        try:
            self.neo4j_driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password)
            )
            # Verify connection
            self.neo4j_driver.verify_connectivity()
            logger.info(f"Neo4j connected at {settings.neo4j_uri}")
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j: {e}")

    def close(self):
        if self.duckdb_conn:
            self.duckdb_conn.close()
        if self.neo4j_driver:
            self.neo4j_driver.close()

    # DuckDB Operations
    def store_stock_data(self, symbol: str, data: List[Dict[str, Any]]):
        """
        Store time-series stock data in DuckDB.
        Expected data format: List of dicts with keys: date, open, high, low, close, volume
        """
        if not self.duckdb_conn:
            logger.warning("DuckDB connection not available")
            return

        try:
            # Prepare data for insertion
            # This is a naive insert, can be optimized with appender for large datasets
            for entry in data:
                self.duckdb_conn.execute(
                    """
                    INSERT OR REPLACE INTO stock_prices (symbol, date, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        symbol,
                        entry.get("date"),
                        entry.get("open"),
                        entry.get("high"),
                        entry.get("low"),
                        entry.get("close"),
                        entry.get("volume"),
                    ),
                )
            logger.info(f"Stored {len(data)} records for {symbol} in DuckDB")
        except Exception as e:
            logger.error(f"Error storing stock data in DuckDB: {e}")

    def log_query(self, query_text: str, response_summary: str = ""):
        if not self.duckdb_conn:
            return
        try:
            self.duckdb_conn.execute("""
                INSERT INTO user_queries (query_text, response_summary)
                VALUES (?, ?)
            """, (query_text, response_summary))
        except Exception as e:
            logger.error(f"Error logging query to DuckDB: {e}")

    def get_stock_history(self, symbol: str) -> List[Dict]:
        if not self.duckdb_conn:
            return []
        try:
            result = self.duckdb_conn.execute("""
                SELECT * FROM stock_prices WHERE symbol = ? ORDER BY date ASC
            """, (symbol,)).fetchall()

            columns = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
            serialized_rows = []
            for row in result:
                entry = dict(zip(columns, row))
                ts = entry.get('date')
                if ts is not None:
                    entry['date'] = ts.isoformat() if hasattr(ts, 'isoformat') else str(ts)
                serialized_rows.append(entry)
            return serialized_rows
        except Exception as e:
            logger.error(f"Error retrieving stock history: {e}")
            return []

    # Neo4j / Graphiti Operations
    def ingest_news_article(self, article: Dict[str, Any]):
        """
        Ingest a news article into the Knowledge Graph.
        Expected article keys: title, content, published_at, source
        
        Note: This is where Graphiti logic would go. 
        For now, we will simulate a basic node creation in Neo4j 
        representing the article and potential entities.
        """
        if not self.neo4j_driver:
            logger.warning("Neo4j driver not available")
            return

        title = article.get('title', '')
        content = article.get('content', '')
        source = article.get('source', 'unknown')
        published_at_value = article.get('published_at', datetime.now())
        if isinstance(published_at_value, datetime):
            published_at = published_at_value.isoformat()
        else:
            published_at = str(published_at_value)
        symbols = [
            str(symbol).upper()
            for symbol in article.get('symbols', [])
            if symbol
        ]

        query = """
        MERGE (a:Article {title: $title})
        SET a.content = $content,
            a.published_at = $published_at,
            a.source = $source,
            a.symbols = $symbols
        MERGE (s:Source {name: $source})
        MERGE (a)-[:PUBLISHED_BY]->(s)
        """
        
        # In a real Graphiti implementation, we would pass 'content' to Graphiti
        # which would extract entities (People, Companies, etc.) and edges with time validity.
        
        try:
            with self.neo4j_driver.session() as session:
                session.run(
                    query,
                    title=title,
                    content=content,
                    published_at=published_at,
                    source=source,
                    symbols=symbols,
                )
            logger.info(f"Ingested article '{title}' into Neo4j")
        except Exception as e:
            logger.error(f"Error ingesting article into Neo4j: {e}")

    def get_graph_data(self):
        """
        Retrieve graph data for visualization.
        Returns nodes and links in a format suitable for react-force-graph.
        """
        if not self.neo4j_driver:
            return {"nodes": [], "links": []}

        query = """
        MATCH (n)-[r]->(m)
        RETURN n, r, m
        LIMIT 100
        """
        
        nodes = {}
        links = []
        
        try:
            with self.neo4j_driver.session() as session:
                result = session.run(query)
                for record in result:
                    n = record['n']
                    m = record['m']
                    r = record['r']
                    
                    n_id = n.element_id if hasattr(n, 'element_id') else n.id
                    m_id = m.element_id if hasattr(m, 'element_id') else m.id
                    
                    if n_id not in nodes:
                        nodes[n_id] = {
                            "id": n_id,
                            "label": list(n.labels)[0] if n.labels else "Node",
                            "properties": dict(n)
                        }
                    if m_id not in nodes:
                        nodes[m_id] = {
                            "id": m_id,
                            "label": list(m.labels)[0] if m.labels else "Node",
                            "properties": dict(m)
                        }
                    
                    links.append({
                        "source": n_id,
                        "target": m_id,
                        "type": r.type,
                        "properties": dict(r)
                    })
            
            return {"nodes": list(nodes.values()), "links": links}
        except Exception as e:
            logger.error(f"Error retrieving graph data: {e}")
            return {"nodes": [], "links": []}

    def get_recent_articles(self, symbols: Optional[List[str]] = None, limit: int = 10):
        if not self.neo4j_driver:
            return []

        query = """
        MATCH (a:Article)
        WHERE $symbols IS NULL OR any(symbol IN $symbols WHERE symbol IN coalesce(a.symbols, []))
        RETURN a
        ORDER BY datetime(coalesce(a.published_at, datetime())) DESC
        LIMIT $limit
        """

        normalized_symbols = None
        if symbols:
            normalized_symbols = [str(symbol).upper() for symbol in symbols if symbol]

        try:
            with self.neo4j_driver.session() as session:
                result = session.run(
                    query,
                    symbols=normalized_symbols,
                    limit=limit,
                )
                articles = []
                for record in result:
                    node = record["a"]
                    article_data = dict(node)
                    article_data["id"] = getattr(node, "element_id", None)
                    articles.append(article_data)
                return articles
        except Exception as e:
            logger.error(f"Error retrieving recent articles: {e}")
            return []

    def get_recent_queries(self, limit: int = 10):
        if not self.duckdb_conn:
            return []

        try:
            result = self.duckdb_conn.execute(
                """
                SELECT query_text, response_summary, timestamp
                FROM user_queries
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,)
            ).fetchall()
            columns = ["query_text", "response_summary", "timestamp"]
            serialized_rows = []
            for row in result:
                entry = dict(zip(columns, row))
                ts = entry.get("timestamp")
                if ts is not None:
                    entry["timestamp"] = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
                serialized_rows.append(entry)
            return serialized_rows
        except Exception as e:
            logger.error(f"Error retrieving recent queries: {e}")
            return []

db_manager = DBManager()
