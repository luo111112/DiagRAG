"""Vector store (Milvus) operations."""

from src.vectorstore.semantic_cache import (
    CacheHit,
    CacheStats,
    SemanticCache,
)
from src.vectorstore.context_fingerprint import (
    build_cache_key,
    check_context_match,
    compute_summary_fingerprint,
    extract_keywords,
    jaccard_similarity,
    normalize_text,
)
from src.vectorstore.query_cache import (
    clear_all_cache,
    count_entries,
    delete_cache_entry,
    drop_query_cache_collection,
    ensure_query_cache_collection,
    evict_oldest,
    insert_cache_record,
    search_similar,
    update_hit_stats,
)

__all__ = [
    # semantic_cache
    "CacheHit",
    "CacheStats",
    "SemanticCache",
    # context_fingerprint
    "normalize_text",
    "extract_keywords",
    "compute_summary_fingerprint",
    "jaccard_similarity",
    "check_context_match",
    "build_cache_key",
    # query_cache
    "ensure_query_cache_collection",
    "drop_query_cache_collection",
    "insert_cache_record",
    "update_hit_stats",
    "delete_cache_entry",
    "search_similar",
    "count_entries",
    "evict_oldest",
    "clear_all_cache",
]
