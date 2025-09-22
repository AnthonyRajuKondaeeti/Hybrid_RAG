"""
Conversation Memory Manager
Handles conversation history, context, and caching for enhanced RAG responses.
"""

import hashlib
import json
import logging
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

from config import Config

logger = logging.getLogger(__name__)

@dataclass
class ConversationTurn:
    """Represents a single conversation turn (question + answer)."""
    question: str
    answer: str
    timestamp: datetime
    confidence_score: float
    question_type: str
    processing_time: float
    sources_used: List[str]
    session_id: str
    file_id: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with serializable timestamp."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

@dataclass
class QueryCache:
    """Cache entry for storing query results."""
    query_hash: str
    result: Dict[str, Any]
    timestamp: datetime
    hit_count: int = 0
    session_id: str = ""
    file_id: str = ""

    def is_expired(self, ttl_hours: int = None) -> bool:
        """Check if cache entry has expired."""
        ttl_hours = ttl_hours or Config.CONVERSATION_CACHE_TTL_HOURS
        expiry_time = self.timestamp + timedelta(hours=ttl_hours)
        return datetime.now() > expiry_time


class ConversationMemoryManager:
    """Manages conversation history and context for enhanced RAG responses."""

    def __init__(self, max_turns: int = None, cache_size: int = None):
        self.max_turns = max_turns or Config.MAX_CONVERSATION_TURNS
        self.cache_size = cache_size or Config.CONVERSATION_CACHE_SIZE
        
        # Session-based conversation history
        self.conversations: Dict[str, deque] = {}  # session_id -> conversation_history
        
        # Global query cache
        self.query_cache: Dict[str, QueryCache] = {}
        
        # Session metadata
        self.session_metadata: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"ConversationMemoryManager initialized: max_turns={self.max_turns}, cache_size={self.cache_size}")

    def get_session_history(self, session_id: str) -> deque:
        """Get conversation history for a session."""
        if session_id not in self.conversations:
            self.conversations[session_id] = deque(maxlen=self.max_turns)
            self.session_metadata[session_id] = {
                'created_at': datetime.now(),
                'last_activity': datetime.now(),
                'total_turns': 0
            }
        return self.conversations[session_id]

    def add_turn(self, turn: ConversationTurn):
        """Add a conversation turn to session history."""
        session_history = self.get_session_history(turn.session_id)
        session_history.append(turn)
        
        # Update session metadata
        if turn.session_id in self.session_metadata:
            self.session_metadata[turn.session_id]['last_activity'] = datetime.now()
            self.session_metadata[turn.session_id]['total_turns'] += 1

    def get_context_string(self, session_id: str, file_id: str = None) -> str:
        """Generate context string from conversation history."""
        session_history = self.get_session_history(session_id)
        
        if not session_history:
            return ""

        # Filter by file_id if specified
        relevant_turns = []
        for turn in session_history:
            if file_id is None or turn.file_id == file_id:
                relevant_turns.append(turn)

        if not relevant_turns:
            return ""

        context_parts = ["CONVERSATION HISTORY (for context):"]
        
        for i, turn in enumerate(relevant_turns[-3:], 1):  # Last 3 relevant turns
            context_parts.append(f"\n{i}. Q: {turn.question}")
            # Truncate long answers for context
            answer_preview = turn.answer[:200] + "..." if len(turn.answer) > 200 else turn.answer
            context_parts.append(f"   A: {answer_preview}")

        context_parts.append("\nCURRENT QUESTION:")
        return "\n".join(context_parts)

    def get_related_questions(self, session_id: str, current_question: str, file_id: str = None) -> List[str]:
        """Find related questions from session history."""
        session_history = self.get_session_history(session_id)
        
        if not session_history:
            return []

        current_words = set(current_question.lower().split())
        related = []

        for turn in session_history:
            # Skip if file_id doesn't match
            if file_id and turn.file_id != file_id:
                continue
                
            turn_words = set(turn.question.lower().split())
            overlap = len(current_words.intersection(turn_words))
            
            if overlap > 1:  # At least 2 word overlap
                related.append({
                    'question': turn.question,
                    'overlap_score': overlap / len(current_words),
                    'timestamp': turn.timestamp
                })

        # Sort by overlap score and recency
        related.sort(key=lambda x: (x['overlap_score'], x['timestamp']), reverse=True)
        return [item['question'] for item in related[:2]]

    def cache_query(self, query: str, result: Dict[str, Any], session_id: str, file_id: str = ""):
        """Cache query result with session context."""
        query_hash = self._generate_query_hash(query, file_id)

        # Clean cache if it's full
        if len(self.query_cache) >= self.cache_size:
            self._cleanup_expired_cache()
            
            # If still full, remove oldest entries
            if len(self.query_cache) >= self.cache_size:
                oldest_keys = sorted(
                    self.query_cache.keys(),
                    key=lambda k: self.query_cache[k].timestamp
                )[:self.cache_size // 4]  # Remove 25% of oldest entries
                
                for key in oldest_keys:
                    del self.query_cache[key]

        self.query_cache[query_hash] = QueryCache(
            query_hash=query_hash,
            result=result,
            timestamp=datetime.now(),
            session_id=session_id,
            file_id=file_id
        )

    def get_cached_result(self, query: str, file_id: str = "") -> Optional[Dict[str, Any]]:
        """Get cached result for query."""
        query_hash = self._generate_query_hash(query, file_id)

        if query_hash in self.query_cache:
            cached = self.query_cache[query_hash]
            
            if not cached.is_expired():
                cached.hit_count += 1
                logger.info(f"Cache hit for query hash: {query_hash[:8]}...")
                return cached.result
            else:
                # Remove expired entry
                del self.query_cache[query_hash]
        
        return None

    def _generate_query_hash(self, query: str, file_id: str = "") -> str:
        """Generate hash for query caching."""
        cache_key = f"{query.lower().strip()}_{file_id}"
        return hashlib.md5(cache_key.encode()).hexdigest()

    def _cleanup_expired_cache(self):
        """Remove expired cache entries."""
        expired_keys = []
        
        for key, cached in self.query_cache.items():
            if cached.is_expired():
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.query_cache[key]
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a specific session."""
        if session_id not in self.session_metadata:
            return {}

        session_history = self.get_session_history(session_id)
        metadata = self.session_metadata[session_id]
        
        # Calculate statistics
        if session_history:
            avg_confidence = sum(turn.confidence_score for turn in session_history) / len(session_history)
            avg_processing_time = sum(turn.processing_time for turn in session_history) / len(session_history)
            question_types = [turn.question_type for turn in session_history]
            most_common_type = max(set(question_types), key=question_types.count) if question_types else "unknown"
        else:
            avg_confidence = 0.0
            avg_processing_time = 0.0
            most_common_type = "unknown"

        session_duration = datetime.now() - metadata['created_at']

        return {
            "session_id": session_id,
            "conversation_turns": len(session_history),
            "total_turns": metadata['total_turns'],
            "created_at": metadata['created_at'].isoformat(),
            "last_activity": metadata['last_activity'].isoformat(),
            "session_duration_minutes": session_duration.total_seconds() / 60,
            "avg_confidence": avg_confidence,
            "avg_processing_time": avg_processing_time,
            "most_common_question_type": most_common_type
        }

    def get_global_stats(self) -> Dict[str, Any]:
        """Get global memory and cache statistics."""
        total_turns = sum(len(history) for history in self.conversations.values())
        total_sessions = len(self.conversations)
        
        cache_hits = sum(cache.hit_count for cache in self.query_cache.values())
        cache_hit_rate = cache_hits / max(len(self.query_cache), 1)

        return {
            "total_sessions": total_sessions,
            "total_conversation_turns": total_turns,
            "cache_entries": len(self.query_cache),
            "cache_hit_rate": cache_hit_rate,
            "cache_size_limit": self.cache_size,
            "max_turns_per_session": self.max_turns,
            "memory_usage_estimate_kb": self._estimate_memory_usage()
        }

    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in KB."""
        # Rough estimation based on average string lengths
        conversation_size = sum(
            len(str(turn)) for history in self.conversations.values() 
            for turn in history
        )
        
        cache_size = sum(len(str(cache.result)) for cache in self.query_cache.values())
        
        total_bytes = conversation_size + cache_size
        return total_bytes / 1024  # Convert to KB

    def cleanup_session(self, session_id: str):
        """Clean up data for a specific session."""
        if session_id in self.conversations:
            del self.conversations[session_id]
        
        if session_id in self.session_metadata:
            del self.session_metadata[session_id]
        
        # Remove session-specific cache entries
        session_cache_keys = [
            key for key, cache in self.query_cache.items() 
            if cache.session_id == session_id
        ]
        
        for key in session_cache_keys:
            del self.query_cache[key]
        
        logger.info(f"Cleaned up session {session_id}")

    def cleanup_expired_sessions(self, max_age_hours: int = 24):
        """Clean up old sessions."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        expired_sessions = []
        
        for session_id, metadata in self.session_metadata.items():
            if metadata['last_activity'] < cutoff_time:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.cleanup_session(session_id)
        
        # Also cleanup expired cache
        self._cleanup_expired_cache()
        
        logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

    def export_session_history(self, session_id: str) -> Dict[str, Any]:
        """Export session history for analysis or backup."""
        session_history = self.get_session_history(session_id)
        metadata = self.session_metadata.get(session_id, {})
        
        return {
            "session_id": session_id,
            "metadata": {
                "created_at": metadata.get('created_at', datetime.now()).isoformat(),
                "last_activity": metadata.get('last_activity', datetime.now()).isoformat(),
                "total_turns": metadata.get('total_turns', 0)
            },
            "conversation_history": [turn.to_dict() for turn in session_history],
            "exported_at": datetime.now().isoformat()
        }


# Global instance for the application
conversation_memory = ConversationMemoryManager()