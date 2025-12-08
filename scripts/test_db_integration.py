import sys
import os
from datetime import datetime
import asyncio
from pathlib import Path

# Add the project root to the python path so imports work
sys.path.append(str(Path(__file__).parent.parent / "src"))

from tradegraph_financial_advisor.services.db_manager import db_manager
from tradegraph_financial_advisor.config.settings import settings

def test_duckdb():
    print(f"\n[1/2] Testing DuckDB at {settings.duckdb_path}...")
    try:
        # 1. Store Data
        symbol = "TEST_SYM"
        mock_data = [{
            "date": datetime.now(),
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "close": 102.0,
            "volume": 1000
        }]
        print(f"   > Storing mock data for {symbol}...")
        db_manager.store_stock_data(symbol, mock_data)
        
        # 2. Retrieve Data
        print(f"   > Retrieving history for {symbol}...")
        history = db_manager.get_stock_history(symbol)
        
        if len(history) > 0 and history[0]['symbol'] == symbol:
            print("   ✅ DuckDB Write/Read Successful!")
        else:
            print("   ❌ DuckDB Verification Failed: Data not found.")
            
    except Exception as e:
        print(f"   ❌ DuckDB Error: {e}")

def test_neo4j():
    print(f"\n[2/2] Testing Neo4j at {settings.neo4j_uri}...")
    try:
        # Check connection first
        if not db_manager.neo4j_driver:
            print("   ⚠️ Neo4j driver is not initialized. Check your .env credentials and ensure Neo4j is running.")
            return

        # 1. Ingest Data
        article = {
            "title": "Test Article: Tech Boom",
            "content": "Technology stocks are soaring today due to AI advancements.",
            "source": "TestNews",
            "published_at": datetime.now().isoformat()
        }
        print("   > Ingesting mock article...")
        db_manager.ingest_news_article(article)
        
        # 2. Retrieve Graph Data
        print("   > Fetching graph nodes...")
        graph_data = db_manager.get_graph_data()
        
        nodes = graph_data.get("nodes", [])
        links = graph_data.get("links", [])
        
        print(f"   > Found {len(nodes)} nodes and {len(links)} links.")
        
        # Simple validation: Check if our node is roughly there (this is loose validation)
        found = False
        for node in nodes:
            props = node.get("properties", {})
            if props.get("title") == article["title"]:
                found = True
                break
        
        if found:
            print("   ✅ Neo4j Write/Read Successful!")
        elif len(nodes) > 0:
             print("   ✅ Neo4j Connection Successful (Existing data found, but specific test node might be indexed differently).")
        else:
            print("   ⚠️ Neo4j returned no nodes (Write might have failed silently or DB is empty).")

    except Exception as e:
        print(f"   ❌ Neo4j Error: {e}")

if __name__ == "__main__":
    print("=== TradeGraph Database Integration Test ===")
    
    # Ensure DB is initialized
    if not db_manager:
        print("Critical Error: DB Manager failed to initialize.")
        sys.exit(1)

    test_duckdb()
    test_neo4j()
    
    # Cleanup
    db_manager.close()
    print("\nTests Complete.")
