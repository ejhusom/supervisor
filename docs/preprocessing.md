# Preprocessing

## Overview

Preprocessing transforms large log files into queryable formats, enabling efficient analysis without reading entire files into context.

## When to Use

**Use preprocessing when**:
- Log files > 1000 lines
- Need structured queries (SQL)
- Want semantic search
- Multiple queries on same data

**Skip preprocessing when**:
- Small files (< 500 lines)
- Single query
- File already structured

## Quick Start

```bash
# SQLite only
python main.py --preprocess workspace/data/logs.log

# With embeddings
python main.py --preprocess workspace/data/*.log --embeddings
```

## Components

### SQLite Log Ingestion

Parses logs into SQLite database for SQL queries.

**Features**:
- Auto-detects log format (OpenStack, BGL, generic)
- Creates indexed database
- Enables complex queries

**Schema**:
```sql
CREATE TABLE logs (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    level TEXT,
    component TEXT,
    message TEXT,
    raw_line TEXT,
    file_source TEXT
)
```

**Indexes**: level, timestamp, component

### Embedding RAG

Creates semantic embeddings for similarity search.

**Features**:
- Chunks logs (default: 100 lines)
- Generates embeddings (sentence-transformers)
- Enables natural language queries

**Models**:
- `all-MiniLM-L6-v2` - Fast, good quality (default)
- `all-mpnet-base-v2` - Slower, better quality
- `multi-qa-MiniLM-L6-cos-v1` - Optimized for Q&A

## Configuration

In `config.toml`:
```toml
embeddings_chunk_size = 100  # Lines per chunk
```

In code:
```python
preprocessor.add_step(SQLiteLogIngestion(parser_name="auto"))
preprocessor.add_step(EmbeddingRAG(
    chunk_size=100,
    model_name='all-MiniLM-L6-v2',
    overlap=10
))
```

## Available Tools

After preprocessing, these tools become available:

### SQL Queries

```python
query_logs_from_sqlite_database(
    sql="SELECT * FROM logs WHERE level = 'ERROR' LIMIT 10",
    params=[]
)
```

**Examples**:
```sql
-- Count errors
SELECT COUNT(*) FROM logs WHERE level = 'ERROR'

-- Top components
SELECT component, COUNT(*) as count 
FROM logs 
GROUP BY component 
ORDER BY count DESC 
LIMIT 10

-- Time-based
SELECT * FROM logs 
WHERE timestamp BETWEEN '2017-05-16 00:00' AND '2017-05-16 01:00'
```

### Error Logs

```python
get_error_logs_from_sqlite_database(limit=100)
```

Returns most recent ERROR-level entries.

### Keyword Search

```python
search_logs_from_sqlite_database(keyword="connection", limit=50)
```

Searches message field for keyword.

### Statistics

```python
get_log_stats_from_sqlite_database()
```

Returns:
```python
{
    "total_logs": 1000,
    "by_level": {"ERROR": 50, "INFO": 900, ...},
    "top_components": {"kernel": 300, "app": 200, ...}
}
```

### Semantic Search

```python
search_similar_from_embeddings(
    query="database connection failures",
    top_k=5,
    min_similarity=0.0
)
```

Returns chunks semantically similar to query.

**Example queries**:
- "connection timeout errors"
- "memory allocation failures"  
- "authentication problems"

### Chunk Operations

```python
# Get specific chunk
get_chunk_by_index_from_embeddings(chunk_index=42)

# Find similar chunks
find_similar_to_chunk_from_embeddings(chunk_index=42, top_k=5)

# Get info
get_embedding_info()
```

## Programmatic Usage

```python
from pathlib import Path
from core.preprocessor import Preprocessor, SQLiteLogIngestion, EmbeddingRAG

# Create preprocessor
preprocessor = Preprocessor(workspace=Path("workspace/data"))

# Add steps
preprocessor.add_step(SQLiteLogIngestion(parser_name="auto"))
preprocessor.add_step(EmbeddingRAG(chunk_size=100))

# Process
metadata = preprocessor.process(Path("logs/app.log"))

# Get tools
tools = preprocessor.get_all_tools()

# Use with supervisor
supervisor = Supervisor(
    llm_client=llm,
    tool_registry=tool_registry,
    agent_registry=agent_registry,
    instructions_dir="instructions",
    preprocessor=preprocessor  # Add preprocessor
)
```

## Log Format Support

### Built-in Parsers

**OpenStack**:
```
2017-05-16 00:00:00.008 25746 INFO nova.osapi_compute.wsgi.server ...
```

**BGL (Blue Gene/L)**:
```
- 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.363779 ...
```

**Generic**:
```
2024-01-15 10:30:45 ERROR Connection failed
```

### Custom Parser

```python
def my_parser(line: str) -> dict:
    """Parse custom format."""
    parts = line.split("|")
    return {
        "timestamp": parts[0],
        "level": parts[1],
        "component": parts[2],
        "message": parts[3]
    }

preprocessor.add_step(SQLiteLogIngestion(log_parser=my_parser))
```

## Performance

**SQLite Ingestion**:
- 1000 lines/sec typical
- ~1MB database per 10k lines
- Indexed for fast queries

**Embeddings**:
- ~50-100 lines/sec (GPU)
- ~10-20 lines/sec (CPU)
- ~500KB per 1000 chunks

**Query Speed**:
- SQL: < 100ms typical
- Semantic search: < 500ms typical

## Persistence

**Location**: `workspace/data/`

**Files created**:
- `{filename}_logs.db` - SQLite database
- `{filename}_chunks.json` - Chunk metadata
- `{filename}_embeddings.npy` - Embedding vectors

**Reuse**: Preprocessing runs once. Subsequent queries use cached data.

## Example Queries

### Using SQL

```python
# Error trend over time
query_logs_from_sqlite_database("""
    SELECT 
        substr(timestamp, 1, 13) as hour,
        COUNT(*) as error_count
    FROM logs
    WHERE level = 'ERROR'
    GROUP BY hour
    ORDER BY hour
""")
```

### Using Semantic Search

```python
# Find authentication issues
results = search_similar_from_embeddings(
    query="user authentication failed invalid credentials",
    top_k=10
)

for r in results:
    print(f"Similarity: {r['similarity']:.3f}")
    print(f"File: {r['file']}, Lines: {r['lines']}")
    print(f"Content: {r['chunk'][:200]}...")
```

## Best Practices

1. **Choose chunk size**: Smaller (50) for precise search, larger (200) for context
2. **Set similarity threshold**: 0.3-0.5 typical for semantic search
3. **Limit results**: Use `limit` parameter to avoid overwhelming context
4. **Use SQL for structure**: Counts, aggregations, time ranges
5. **Use embeddings for semantics**: Natural language queries, concept search

## Troubleshooting

**"Preprocessing not available"**:
```bash
pip install sentence-transformers
```

**Slow embedding creation**:
- Use GPU if available
- Reduce chunk_size
- Set max_chunks limit

**Large database files**:
- Expected for large logs
- SQLite is efficient for storage
- Query performance remains fast

**Parser errors**:
- Check log format matches parser
- Use custom parser for non-standard formats
- Try "generic" parser as fallback
