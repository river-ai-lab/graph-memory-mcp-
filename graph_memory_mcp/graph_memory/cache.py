"""Caching utilities for MCP Graph Memory."""

import hashlib
import json
from typing import Any, Optional

from cachetools import LRUCache, TTLCache


def _observe(cache: str, hit: bool) -> None:
    from graph_memory_mcp.metrics import observe_cache

    observe_cache(cache, hit)


class CacheManager:
    """Manages LRU and TTL caches for embeddings and search results."""

    def __init__(self, config):
        """Initialize cache manager with configuration."""
        self.config = config
        # Bumped on every write; in-process lexical indexes key on it.
        self.generation = 0

        # LRU cache for embeddings (text -> vector)
        self.embeddings = (
            LRUCache(maxsize=config.cache_embeddings_maxsize)
            if config.cache_embeddings_enabled
            else None
        )

        # TTL cache for search results (query hash -> results)
        self.search = (
            TTLCache(maxsize=config.cache_search_maxsize, ttl=config.cache_search_ttl)
            if config.cache_search_enabled
            else None
        )

    def get_embedding(self, text: str) -> Optional[list]:
        """Get cached embedding for text."""
        if self.embeddings is None:
            return None
        cached = self.embeddings.get(text)
        _observe("embeddings", cached is not None)
        return cached

    def set_embedding(self, text: str, embedding: list):
        """Cache embedding for text."""
        if self.embeddings is not None:
            self.embeddings[text] = embedding

    def get_search(self, query_hash: str) -> Optional[Any]:
        """Get cached search results."""
        if self.search is None:
            return None
        cached = self.search.get(query_hash)
        _observe("search", cached is not None)
        return cached

    def set_search(self, query_hash: str, results: Any):
        """Cache search results."""
        if self.search is not None:
            self.search[query_hash] = results

    def invalidate_search(self):
        """Invalidate all search caches (called on mutations)."""
        self.generation += 1
        if self.search is not None:
            self.search.clear()

    def stats(self) -> dict:
        """Get cache statistics for monitoring."""
        return {
            "embeddings": {
                "enabled": self.embeddings is not None,
                "size": len(self.embeddings) if self.embeddings is not None else 0,
                "maxsize": (
                    getattr(self.config, "cache_embeddings_maxsize", 0)
                    if self.embeddings is not None
                    else 0
                ),
            },
            "search": {
                "enabled": self.search is not None,
                "size": len(self.search) if self.search is not None else 0,
                "maxsize": (
                    getattr(self.config, "cache_search_maxsize", 0)
                    if self.search is not None
                    else 0
                ),
                "ttl": (
                    getattr(self.config, "cache_search_ttl", 0)
                    if self.search is not None
                    else 0
                ),
            },
        }


def hash_query(query: str, **kwargs) -> str:
    """Create deterministic hash for search query and parameters."""
    data = {"query": query, **kwargs}
    json_str = json.dumps(data, sort_keys=True)
    return hashlib.md5(json_str.encode()).hexdigest()
