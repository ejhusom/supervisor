"""
Session manager for maintaining supervisor state across requests.
"""

import time
import uuid
from typing import Dict, Optional, Any
from datetime import datetime
from threading import Lock
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import config as core_config
from core.llm_client import LLMClient
from core.supervisor import Supervisor
from core.preprocessor import Preprocessor, SQLiteLogIngestion, EmbeddingRAG
from registry.tool_registry import ToolRegistry
from registry.agent_registry import AgentRegistry


class Session:
    """
    Represents a user session with its own supervisor instance.
    """
    
    def __init__(
        self,
        session_id: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        preprocessing_enabled: bool = False
    ):
        self.session_id = session_id
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.preprocessing_enabled = preprocessing_enabled
        
        # Configuration
        self.config = {
            "provider": provider or core_config.get("provider"),
            "model": model or core_config.get("model"),
            "api_key": core_config.get("api_key"),
            "temperature": temperature if temperature is not None else core_config.get("temperature"),
            "max_tokens": max_tokens or core_config.get("max_tokens")
        }
        
        # Initialize components
        self.llm_client = LLMClient(
            provider=self.config["provider"],
            model=self.config["model"],
            api_key=self.config["api_key"]
        )
        
        self.tool_registry = ToolRegistry()
        self.agent_registry = AgentRegistry()
        self.preprocessor = None
        
        # Create supervisor
        self.supervisor = Supervisor(
            llm_client=self.llm_client,
            tool_registry=self.tool_registry,
            agent_registry=self.agent_registry,
            instructions_dir="instructions",
            preprocessor=self.preprocessor
        )
    
    def update_config(self, **kwargs):
        """Update session configuration."""
        for key, value in kwargs.items():
            if value is not None and key in self.config:
                self.config[key] = value
        
        # Recreate LLM client if provider/model changed
        if "provider" in kwargs or "model" in kwargs or "api_key" in kwargs:
            self.llm_client = LLMClient(
                provider=self.config["provider"],
                model=self.config["model"],
                api_key=self.config["api_key"]
            )
            
            # Recreate supervisor with new client
            self.supervisor = Supervisor(
                llm_client=self.llm_client,
                tool_registry=self.tool_registry,
                agent_registry=self.agent_registry,
                instructions_dir="instructions",
                preprocessor=self.preprocessor
            )
        
        self.last_accessed = datetime.now()
    
    def setup_preprocessing(
        self,
        log_path: Path,
        enable_embeddings: bool = False,
        chunk_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """Setup preprocessing for this session."""
        self.preprocessor = Preprocessor()
        self.preprocessor.add_step(SQLiteLogIngestion())
        
        if enable_embeddings:
            if chunk_size is None:
                chunk_size = core_config.get("embeddings_chunk_size", 100)
            try:
                self.preprocessor.add_step(EmbeddingRAG(chunk_size=chunk_size))
            except ImportError:
                pass  # Embeddings not available
        
        # Process logs
        metadata = self.preprocessor.process(log_path)
        
        # Recreate supervisor with preprocessor
        self.supervisor = Supervisor(
            llm_client=self.llm_client,
            tool_registry=self.tool_registry,
            agent_registry=self.agent_registry,
            instructions_dir="instructions",
            preprocessor=self.preprocessor
        )
        
        self.preprocessing_enabled = True
        self.last_accessed = datetime.now()
        
        return metadata
    
    def touch(self):
        """Update last accessed time."""
        self.last_accessed = datetime.now()
    
    def age_seconds(self) -> float:
        """Get age in seconds since last access."""
        return (datetime.now() - self.last_accessed).total_seconds()


class SessionManager:
    """
    Manages session lifecycle with TTL-based cleanup.
    """
    
    def __init__(self, ttl_seconds: int = 3600, max_sessions: int = 100):
        self.sessions: Dict[str, Session] = {}
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self.lock = Lock()
        self.last_cleanup = time.time()
    
    def create_session(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        preprocessing_enabled: bool = False
    ) -> Session:
        """Create a new session."""
        with self.lock:
            # Check if we need to cleanup
            self._maybe_cleanup()
            
            # Check max sessions
            if len(self.sessions) >= self.max_sessions:
                # Remove oldest session
                oldest_id = min(self.sessions.keys(), key=lambda k: self.sessions[k].last_accessed)
                del self.sessions[oldest_id]
            
            # Generate unique ID
            session_id = f"sess_{uuid.uuid4().hex[:12]}"
            
            # Create session
            session = Session(
                session_id=session_id,
                provider=provider,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                preprocessing_enabled=preprocessing_enabled
            )
            
            self.sessions[session_id] = session
            return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID."""
        with self.lock:
            session = self.sessions.get(session_id)
            if session:
                session.touch()
            return session
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                return True
            return False
    
    def list_sessions(self) -> list:
        """List all active session IDs."""
        with self.lock:
            return list(self.sessions.keys())
    
    def count_sessions(self) -> int:
        """Count active sessions."""
        with self.lock:
            return len(self.sessions)
    
    def _maybe_cleanup(self):
        """Cleanup expired sessions if needed."""
        now = time.time()
        
        # Only cleanup every minute
        if now - self.last_cleanup < 60:
            return
        
        self.last_cleanup = now
        
        # Find expired sessions
        expired = []
        for session_id, session in self.sessions.items():
            if session.age_seconds() > self.ttl_seconds:
                expired.append(session_id)
        
        # Remove expired
        for session_id in expired:
            del self.sessions[session_id]
    
    def cleanup_all(self):
        """Force cleanup of all expired sessions."""
        with self.lock:
            expired = []
            for session_id, session in self.sessions.items():
                if session.age_seconds() > self.ttl_seconds:
                    expired.append(session_id)
            
            for session_id in expired:
                del self.sessions[session_id]
            
            return len(expired)
