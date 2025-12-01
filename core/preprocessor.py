"""
Preprocessing system for large data that doesn't fit in context.

Handles:
- Loading data into queryable formats (SQLite, vector DBs)
- Creating embeddings for RAG
- Indexing and chunking large files
- Making preprocessed data available via tools
"""

import json
import re
import sqlite3
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from abc import ABC, abstractmethod

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    SentenceTransformer = None

from .config import config

class LogParsers:
    """Registry of built-in log parsers."""
    
    @staticmethod
    def openstack(line: str) -> Optional[Dict]:
        """OpenStack log format."""
        pattern = r'(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)\s+(?P<level>\w+)\s+(?P<component>[\w.]+).*\]\s+(?P<message>.*)'
        match = re.match(pattern, line)
        if match:
            return match.groupdict()
        return None
    
    @staticmethod
    def bgl(line: str) -> Optional[Dict]:
        """Blue Gene/L log format."""
        parts = line.strip().split(None, 9)
        if len(parts) < 10:
            return None
        return {
            "timestamp": parts[4],
            "level": parts[8],
            "component": parts[7],
            "message": parts[9]
        }
    
    @staticmethod
    def generic(line: str) -> Optional[Dict]:
        """Generic log format: TIMESTAMP LEVEL MESSAGE."""
        parts = line.split(maxsplit=2)
        if len(parts) >= 2:
            return {
                "timestamp": parts[0],
                "level": parts[1] if len(parts) > 1 else "INFO",
                "component": "",
                "message": parts[2] if len(parts) > 2 else ""
            }
        return None
    
    @staticmethod
    def auto(line: str) -> Optional[Dict]:
        """Auto-detect format and parse."""
        # Try each parser in order
        for parser in [LogParsers.openstack, LogParsers.bgl, LogParsers.generic]:
            result = parser(line)
            if result:
                return result
        return None


class PreprocessorStep(ABC):
    """Base class for preprocessing steps."""
    
    @abstractmethod
    def process(self, input_path: Path, workspace: Path) -> Dict[str, Any]:
        """
        Process data and return metadata about what was created.
        
        Args:
            input_path: Path to input data
            workspace: Workspace directory for outputs
            
        Returns:
            Dict with metadata (paths, stats, etc.)
        """
        pass
    
    @abstractmethod
    def get_tools(self) -> Dict[str, Dict[str, Any]]:
        """
        Get tools that agents can use to query preprocessed data.
        
        Returns:
            Dict of tool definitions (function + schema)
        """
        pass


class SQLiteLogIngestion(PreprocessorStep):
    """Ingest log files into SQLite for efficient querying."""
    
    def __init__(self, log_parser: Optional[Callable] = None, parser_name: str = "auto"):
        """
        Args:
            log_parser: Custom parser function(line: str) -> dict
            parser_name: Built-in parser name: 'auto', 'openstack', 'bgl', 'generic'
        """
        if log_parser:
            self.log_parser = log_parser
        else:
            # Use built-in parser from LogParsers registry
            self.log_parser = getattr(LogParsers, parser_name, LogParsers.auto)
        
        self.db_path = None
    
    def process(self, input_path: Path, workspace: Path) -> Dict[str, Any]:
        """Ingest log file(s) into SQLite."""
        self.db_path = workspace / Path(input_path.stem + "_logs.db")

        # If database exists, don't re-ingest
        if self.db_path.exists():
            print(f"SQLite database already exists at: {self.db_path}, skipping ingestion.")
            # Find metadata from existing DB
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM logs")
            total_lines = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM logs WHERE level = 'ERROR'")
            error_count = cursor.fetchone()[0]
            conn.close()

            return {
                "db_path": str(self.db_path),
                "total_lines": total_lines,
                "error_count": error_count,
                "files_processed": 0,
                "message": "Database already exists, ingestion skipped."
            }
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                level TEXT,
                component TEXT,
                message TEXT,
                raw_line TEXT,
                file_source TEXT
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_level ON logs(level)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON logs(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_component ON logs(component)")
        
        # Get files
        if input_path.is_file():
            files = [input_path]
        else:
            files = list(input_path.glob("*.log"))
        
        # Ingest files
        total_lines = 0
        error_count = 0
        
        for file in files:
            with open(file, 'r', encoding='utf-8', errors='ignore') as f:

                # If auto-detecting, determine parser from first line
                if self.log_parser == LogParsers.auto:
                    first_line = f.readline()
                    detected = LogParsers.auto(first_line)
                    if detected:
                        # Determine which parser succeeded
                        for parser_name in ['openstack', 'bgl', 'generic']:
                            parser_func = getattr(LogParsers, parser_name)
                            if parser_func(first_line):
                                self.log_parser = parser_func
                                print(f"Auto-detected log format: {parser_name}")
                                break
                    else:
                        print("Could not auto-detect log format; using generic parser.")
                        self.log_parser = LogParsers.generic
                    # Reset file pointer
                    f.seek(0)

                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    total_lines += 1
                    parsed = self.log_parser(line)
                    
                    if parsed:
                        cursor.execute("""
                            INSERT INTO logs (timestamp, level, component, message, raw_line, file_source)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (
                            parsed.get("timestamp", ""),
                            parsed.get("level", ""),
                            parsed.get("component", ""),
                            parsed.get("message", ""),
                            line,
                            file.name
                        ))
                        
                        if parsed.get("level") == "ERROR":
                            error_count += 1
        
        conn.commit()
        conn.close()
        
        return {
            "db_path": str(self.db_path),
            "total_lines": total_lines,
            "error_count": error_count,
            "files_processed": len(files)
        }
    
    def get_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get SQL query tools."""

        def query_logs_from_sqlite_database(sql: str, params: List = None) -> List[Dict]:
            """Execute SQL query on logs database."""
            if not self.db_path or not self.db_path.exists():
                return {"error": "Database not initialized"}
            
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            try:
                cursor.execute(sql, params or [])
                results = [dict(row) for row in cursor.fetchall()]
                conn.close()
                return results
            except Exception as e:
                conn.close()
                return {"error": str(e)}

        def get_error_logs_from_sqlite_database(limit: int = 100) -> List[Dict]:
            """Get error-level log entries."""
            return query_logs_from_sqlite_database(
                "SELECT * FROM logs WHERE level = 'ERROR' ORDER BY timestamp DESC LIMIT ?",
                [limit]
            )

        def search_logs_from_sqlite_database(keyword: str, limit: int = 100) -> List[Dict]:
            """Search logs by keyword."""
            return query_logs_from_sqlite_database(
                "SELECT * FROM logs WHERE message LIKE ? LIMIT ?",
                [f"%{keyword}%", limit]
            )
        
        def get_log_stats_from_sqlite_database() -> Dict:
            """Get statistics about logs."""
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            stats = {}
            
            # Total count
            cursor.execute("SELECT COUNT(*) FROM logs")
            stats["total_logs"] = cursor.fetchone()[0]
            
            # By level
            cursor.execute("SELECT level, COUNT(*) as count FROM logs GROUP BY level")
            stats["by_level"] = {row[0]: row[1] for row in cursor.fetchall()}
            
            # By component
            cursor.execute("SELECT component, COUNT(*) as count FROM logs GROUP BY component ORDER BY count DESC LIMIT 10")
            stats["top_components"] = {row[0]: row[1] for row in cursor.fetchall()}
            
            conn.close()
            return stats
        
        return {
            "query_logs_from_sqlite_database": {
                "function": query_logs_from_sqlite_database,
                "schema": {
                    "type": "function",
                    "function": {
                        "name": "query_logs_from_sqlite_database",
                        "description": "Execute SQL query on preprocessed logs database. Returns list of matching log entries.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "sql": {
                                    "type": "string",
                                    "description": "SQL query (SELECT only). Table: logs(id, timestamp, level, component, message, raw_line, file_source)"
                                },
                                "params": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Optional query parameters for ? placeholders"
                                }
                            },
                            "required": ["sql"]
                        }
                    }
                }
            },
            "get_error_logs_from_sqlite_database": {
                "function": get_error_logs_from_sqlite_database,
                "schema": {
                    "type": "function",
                    "function": {
                        "name": "get_error_logs_from_sqlite_database",
                        "description": "Get ERROR-level log entries, most recent first.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "limit": {
                                    "type": "integer",
                                    "description": "Maximum number of results (default: 100)",
                                    "default": 100
                                }
                            }
                        }
                    }
                }
            },
            "search_logs_from_sqlite_database": {
                "function": search_logs_from_sqlite_database,
                "schema": {
                    "type": "function",
                    "function": {
                        "name": "search_logs_from_sqlite_database",
                        "description": "Search logs by keyword in message field.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "keyword": {
                                    "type": "string",
                                    "description": "Keyword to search for"
                                },
                                "limit": {
                                    "type": "integer",
                                    "description": "Maximum results (default: 100)",
                                    "default": 100
                                }
                            },
                            "required": ["keyword"]
                        }
                    }
                }
            },
            "get_log_stats_from_sqlite_database": {
                "function": get_log_stats_from_sqlite_database,
                "schema": {
                    "type": "function",
                    "function": {
                        "name": "get_log_stats_from_sqlite_database",
                        "description": "Get statistics about preprocessed logs (counts by level, top components, etc.)",
                        "parameters": {
                            "type": "object",
                            "properties": {}
                        }
                    }
                }
            }
        }


class EmbeddingRAG(PreprocessorStep):
    """
    Create embeddings for semantic search over logs.
    
    Uses sentence-transformers to encode log chunks and enables
    similarity-based search for finding related log entries.
    """
    
    def __init__(
        self,
        chunk_size: int = 100,
        model_name: str = 'all-MiniLM-L6-v2',
        overlap: int = 10,
        max_chunks: Optional[int] = None
    ):
        """
        Args:
            chunk_size: Number of log lines per chunk
            model_name: Sentence-transformers model name
                        Options: 'all-MiniLM-L6-v2' (fast, good quality)
                                'all-mpnet-base-v2' (slower, better quality)
                                'multi-qa-MiniLM-L6-cos-v1' (optimized for Q&A)
            overlap: Number of lines to overlap between chunks
            max_chunks: Optional limit on number of chunks (for large files)
        """
        if not EMBEDDINGS_AVAILABLE:
            raise ImportError(
                "Embeddings require additional packages. Install with:\n"
                "pip install sentence-transformers numpy scikit-learn"
            )
        
        self.chunk_size = chunk_size
        self.model_name = model_name
        self.overlap = overlap
        self.max_chunks = max_chunks
        
        # Will be set during processing
        self.chunks_path = None
        self.embeddings_path = None
        self.model = None
        
        # Load model (cached after first load)
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def process(self, input_path: Path, workspace: Path) -> Dict[str, Any]:
        """Create chunks and embeddings from log files."""

        # If embeddings already exist, skip processing
        self.embeddings_path = workspace / Path(input_path.stem + "_embeddings.npy")
        self.chunks_path = workspace / Path(input_path.stem + "_chunks.json")

        if self.embeddings_path.exists():
            print(f"Embeddings already exist at: {self.embeddings_path}, skipping processing.")
            return {
                "chunks_path": str(self.chunks_path),
                "embeddings_path": str(self.embeddings_path),
                "message": "Embeddings already exist, processing skipped."
            }

        print("Creating log chunks...")
        
        chunks = []
        chunk_metadata = []
        
        # Get log files
        if input_path.is_file():
            files = [input_path]
        else:
            files = list(input_path.glob("*.log"))
        
        # Create chunks with overlap
        for file in files:
            with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # Chunk with overlap
            i = 0
            chunk_num = 0
            while i < len(lines):
                end = min(i + self.chunk_size, len(lines))
                chunk_lines = lines[i:end]
                chunk_text = ''.join(chunk_lines)
                
                # Skip empty chunks
                if chunk_text.strip():
                    chunks.append(chunk_text)
                    chunk_metadata.append({
                        "file": file.name,
                        "start_line": i + 1,  # 1-indexed for readability
                        "end_line": end,
                        "chunk_num": chunk_num,
                        "size": len(chunk_lines)
                    })
                    chunk_num += 1
                
                # Check max chunks limit
                if self.max_chunks and len(chunks) >= self.max_chunks:
                    print(f"Reached max chunks limit ({self.max_chunks})")
                    break
                
                # Move to next chunk (with overlap)
                i += self.chunk_size - self.overlap
            
            if self.max_chunks and len(chunks) >= self.max_chunks:
                break
        
        print(f"Created {len(chunks)} chunks from {len(files)} file(s)")
        
        # Save chunks
        with open(self.chunks_path, 'w') as f:
            json.dump({
                "chunks": chunks,
                "metadata": chunk_metadata,
                "config": {
                    "chunk_size": self.chunk_size,
                    "overlap": self.overlap,
                    "model_name": self.model_name
                }
            }, f)
        
        print(f"Saved chunks to: {self.chunks_path}")
        
        # Create embeddings
        print("Creating embeddings...")
        print("(This may take a few minutes for large datasets)")
        
        # Batch encode for efficiency
        batch_size = 32
        embeddings = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            batch_embeddings = self.model.encode(
                batch,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True  # For cosine similarity
            )
            embeddings.append(batch_embeddings)
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"  Processed {i + len(batch)}/{len(chunks)} chunks...")
        
        embeddings = np.vstack(embeddings)
        
        # Save embeddings
        np.save(self.embeddings_path, embeddings)
        
        print(f"Created embeddings with shape: {embeddings.shape}")
        print(f"Saved embeddings to: {self.embeddings_path}")
        
        return {
            "chunks_path": str(self.chunks_path),
            "embeddings_path": str(self.embeddings_path),
            "num_chunks": len(chunks),
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "embedding_dim": embeddings.shape[1],
            "model_name": self.model_name
        }
    
    def get_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get semantic search tools."""
        
        def search_similar_from_embeddings(query: str, top_k: int = 5, min_similarity: float = 0.0) -> List[Dict]:
            """
            Find log chunks semantically similar to query.
            
            Args:
                query: Search query in natural language
                top_k: Number of results to return
                min_similarity: Minimum similarity threshold (0-1)
            
            Returns:
                List of matching chunks with metadata and similarity scores
            """
            if not self.chunks_path or not self.chunks_path.exists():
                return {"error": "Chunks not initialized"}
            
            if not self.embeddings_path or not self.embeddings_path.exists():
                return {"error": "Embeddings not initialized"}
            
            # Load data
            with open(self.chunks_path, 'r') as f:
                data = json.load(f)
            
            embeddings = np.load(self.embeddings_path)
            
            # Encode query
            query_embedding = self.model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True
            )[0]
            
            # Compute similarities
            # Since embeddings are normalized, dot product = cosine similarity
            similarities = np.dot(embeddings, query_embedding)
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1]
            
            # Filter by minimum similarity and top_k
            results = []
            for idx in top_indices:
                similarity = float(similarities[idx])
                
                if similarity < min_similarity:
                    break
                
                # Get chunk and metadata
                chunk = data["chunks"][idx]
                metadata = data["metadata"][idx]
                
                results.append({
                    "chunk": chunk[:500] + "..." if len(chunk) > 500 else chunk,
                    "full_chunk": chunk,  # Include full chunk
                    "metadata": metadata,
                    "similarity": round(similarity, 4),
                    "file": metadata["file"],
                    "lines": f"{metadata['start_line']}-{metadata['end_line']}"
                })
                
                if len(results) >= top_k:
                    break
            
            return results
        
        def find_similar_to_chunk_from_embeddings(chunk_index: int, top_k: int = 5) -> List[Dict]:
            """
            Find chunks similar to a specific chunk index.
            
            Useful for finding related log entries to a known chunk.
            """
            if not self.chunks_path or not self.chunks_path.exists():
                return {"error": "Chunks not initialized"}
            
            if not self.embeddings_path or not self.embeddings_path.exists():
                return {"error": "Embeddings not initialized"}
            
            # Load data
            with open(self.chunks_path, 'r') as f:
                data = json.load(f)
            
            embeddings = np.load(self.embeddings_path)
            
            if chunk_index >= len(embeddings):
                return {"error": f"Chunk index {chunk_index} out of range (max: {len(embeddings)-1})"}
            
            # Get embedding for this chunk
            query_embedding = embeddings[chunk_index]
            
            # Compute similarities
            similarities = np.dot(embeddings, query_embedding)
            
            # Get top-k (excluding the query chunk itself)
            top_indices = np.argsort(similarities)[::-1]
            
            results = []
            for idx in top_indices:
                if idx == chunk_index:
                    continue  # Skip self
                
                similarity = float(similarities[idx])
                chunk = data["chunks"][idx]
                metadata = data["metadata"][idx]
                
                results.append({
                    "chunk": chunk[:500] + "..." if len(chunk) > 500 else chunk,
                    "metadata": metadata,
                    "similarity": round(similarity, 4),
                    "file": metadata["file"],
                    "lines": f"{metadata['start_line']}-{metadata['end_line']}"
                })
                
                if len(results) >= top_k:
                    break
            
            return results
        
        def get_chunk_by_index_from_embeddings(chunk_index: int) -> Dict:
            """Get full chunk content by index."""
            if not self.chunks_path or not self.chunks_path.exists():
                return {"error": "Chunks not initialized"}
            
            with open(self.chunks_path, 'r') as f:
                data = json.load(f)
            
            if chunk_index >= len(data["chunks"]):
                return {"error": f"Chunk index {chunk_index} out of range (max: {len(data['chunks'])-1})"}
            
            return {
                "chunk": data["chunks"][chunk_index],
                "metadata": data["metadata"][chunk_index]
            }
        
        def get_embedding_info() -> Dict:
            """Get information about the embedding setup."""
            if not self.chunks_path or not self.chunks_path.exists():
                return {"error": "Embeddings not initialized"}
            
            with open(self.chunks_path, 'r') as f:
                data = json.load(f)
            
            embeddings = np.load(self.embeddings_path)
            
            return {
                "model_name": data["config"]["model_name"],
                "num_chunks": len(data["chunks"]),
                "chunk_size": data["config"]["chunk_size"],
                "overlap": data["config"]["overlap"],
                "embedding_dim": embeddings.shape[1],
                "total_files": len(set(m["file"] for m in data["metadata"]))
            }
        
        return {
            "search_similar": {
                "function": search_similar_from_embeddings,
                "schema": {
                    "type": "function",
                    "function": {
                        "name": "search_similar_from_embeddings",
                        "description": "Find log chunks semantically similar to query using embeddings. Returns chunks with similarity scores.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Natural language query (e.g., 'connection timeout errors', 'database failures')"
                                },
                                "top_k": {
                                    "type": "integer",
                                    "description": "Number of results to return (default: 5)",
                                    "default": 5
                                },
                                "min_similarity": {
                                    "type": "number",
                                    "description": "Minimum similarity threshold 0-1 (default: 0.0)",
                                    "default": 0.0
                                }
                            },
                            "required": ["query"]
                        }
                    }
                }
            },
            "find_similar_to_chunk_from_embeddings": {
                "function": find_similar_to_chunk_from_embeddings,
                "schema": {
                    "type": "function",
                    "function": {
                        "name": "find_similar_to_chunk_from_embeddings",
                        "description": "Find chunks similar to a specific chunk by index. Useful for finding related log entries.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "chunk_index": {
                                    "type": "integer",
                                    "description": "Index of the chunk to find similar chunks for"
                                },
                                "top_k": {
                                    "type": "integer",
                                    "description": "Number of results (default: 5)",
                                    "default": 5
                                }
                            },
                            "required": ["chunk_index"]
                        }
                    }
                }
            },
            "get_chunk_by_index_from_embeddings": {
                "function": get_chunk_by_index_from_embeddings,
                "schema": {
                    "type": "function",
                    "function": {
                        "name": "get_chunk_by_index_from_embeddings",
                        "description": "Get full chunk content by index number.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "chunk_index": {
                                    "type": "integer",
                                    "description": "Index of chunk to retrieve"
                                }
                            },
                            "required": ["chunk_index"]
                        }
                    }
                }
            },
            "get_embedding_info": {
                "function": get_embedding_info,
                "schema": {
                    "type": "function",
                    "function": {
                        "name": "get_embedding_info",
                        "description": "Get information about the embedding setup (model, chunks, dimensions).",
                        "parameters": {
                            "type": "object",
                            "properties": {}
                        }
                    }
                }
            }
        }

class Preprocessor:
    """Main preprocessor that manages multiple preprocessing steps."""
    
    def __init__(self, workspace: Path = None):
        """
        Args:
            workspace: Workspace directory (defaults to config workspace_data)
        """
        self.workspace = workspace or Path(config["workspace_data"])
        self.steps: List[PreprocessorStep] = []
        self.metadata: Dict[str, Any] = {}
    
    def add_step(self, step: PreprocessorStep) -> None:
        """Add a preprocessing step."""
        self.steps.append(step)
    
    def process(self, input_path: Path) -> Dict[str, Any]:
        """Run all preprocessing steps."""
        print(f"Running preprocessing on: {input_path}")
        
        for i, step in enumerate(self.steps):
            step_name = step.__class__.__name__
            print(f"  Step {i+1}/{len(self.steps)}: {step_name}")
            
            metadata = step.process(input_path, self.workspace)
            self.metadata[step_name] = metadata
            
            print(f"    ✓ {step_name} complete")
            for key, value in metadata.items():
                print(f"      - {key}: {value}")
        
        return self.metadata
    
    def get_all_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get all tools from all preprocessing steps."""
        all_tools = {}
        
        for step in self.steps:
            tools = step.get_tools()
            all_tools.update(tools)
        
        return all_tools