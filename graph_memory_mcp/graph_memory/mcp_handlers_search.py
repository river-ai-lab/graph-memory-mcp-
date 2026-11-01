"""Search handlers for MCP Graph Memory."""

import logging
import time
from typing import Dict, List, Optional

from graph_memory_mcp.config import MCPServerConfig
from graph_memory_mcp.graph_memory.cache import hash_query
from graph_memory_mcp.graph_memory.compact_recall import decorate_search_response
from graph_memory_mcp.graph_memory.database import FalkorDBClient
from graph_memory_mcp.graph_memory.lexical_search import (
    LEXICAL_MODES,
    cached_index,
    normalize_hybrid_modes,
    normalize_search_mode,
    rrf_fuse,
)
from graph_memory_mcp.graph_memory.owner_scoped_search import (
    SearchType,
    build_metadata_filter_clauses,
    normalize_search_type,
)
from graph_memory_mcp.graph_memory.search_rank import rerank_results
from graph_memory_mcp.graph_memory.utils import (
    ensure_text,
    error_response,
    load_json,
    mcp_handler,
    normalize_owner_id,
    parse_embedding_value,
    require_node_id,
    success_response,
    touch_nodes,
)

logger = logging.getLogger(__name__)


def _vector_ann_k(
    limit: int,
    graph_total: int | None,
    config: MCPServerConfig,
) -> int:
    """Size the ANN candidate pool for post_filter search."""
    baseline = max(limit * 2, config.post_filter_ann_k_min)
    if not graph_total or graph_total <= baseline:
        return baseline
    scaled = max(baseline, graph_total // 5)
    return min(scaled, config.post_filter_ann_k_max)


_LABEL_COUNT_TTL_SECONDS = 60


def _count_labeled_nodes(
    db: FalkorDBClient, node_type: str, owner_id: str
) -> int | None:
    """Approximate label count for ANN sizing, cached per owner graph for 60s."""
    import time as _time

    cache: dict = getattr(db, "_label_count_cache", None) or {}
    cache_key = f"{owner_id}:{node_type}"
    cached = cache.get(cache_key)
    now = _time.time()
    if cached and now - cached[1] < _LABEL_COUNT_TTL_SECONDS:
        return cached[0]
    try:
        count = int(db.count_labeled(node_type, owner_id))
        cache[cache_key] = (count, now)
        setattr(db, "_label_count_cache", cache)
        return count
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not count %s nodes for ANN sizing: %s", node_type, exc)
    return None


def _rows_to_search_results(rows) -> List[Dict]:
    results: List[Dict] = []
    for row in rows or []:
        results.append(
            {
                "node_id": str(row[0]),
                "node_type": row[1],
                "text": ensure_text(row[2]),
                "status": ensure_text(row[3]),
                "created_at": row[4],
                "metadata": load_json(row[5], {}),
                "similarity": 1.0 - float(row[6]),
            }
        )
    return results


def _search_nodes_post_filter(
    db: FalkorDBClient,
    config: MCPServerConfig,
    node_type: str,
    embedding: List[float],
    limit: int,
    max_distance: float,
    owner_id: str,
    include_outdated: bool = False,
    status: Optional[str] = None,
    metadata_filter: Optional[Dict] = None,
    include_ttl: bool = False,
) -> List[Dict]:
    """post_filter: global ANN (queryNodes), then owner/status filters."""
    ann_k = _vector_ann_k(limit, _count_labeled_nodes(db, node_type, owner_id), config)
    return _rows_to_search_results(
        db.ann_rows(
            node_type=node_type,
            embedding=embedding,
            owner_id=owner_id,
            ann_k=ann_k,
            limit=limit,
            max_distance=max_distance,
            include_outdated=include_outdated,
            status=status,
            metadata_filter=metadata_filter,
            include_ttl=include_ttl,
        )
    )


def _search_nodes_pre_filter(
    db: FalkorDBClient,
    node_type: str,
    embedding: List[float],
    limit: int,
    max_distance: float,
    owner_id: str,
    include_outdated: bool = False,
    status: Optional[str] = None,
    metadata_filter: Optional[Dict] = None,
    include_ttl: bool = False,
) -> List[Dict]:
    """pre_filter: owner/status filters first, then exact cosine distance."""
    return _rows_to_search_results(
        db.similarity_rows(
            node_type=node_type,
            embedding=embedding,
            owner_id=owner_id,
            limit=limit,
            max_distance=max_distance,
            include_outdated=include_outdated,
            status=status,
            metadata_filter=metadata_filter,
            include_ttl=include_ttl,
        )
    )


def _contradiction_pairs(
    db: FalkorDBClient, owner_id: str, results: List[Dict]
) -> List[Dict]:
    """Pairs in `results` that already have a CONTRADICTS edge."""
    ids = [str(item["node_id"]) for item in results if item.get("node_id")]
    if len(ids) < 2:
        return []
    pairs = []
    for row in db.contradicts_among(owner_id, ids):
        pairs.append(
            {
                "from_id": str(row[0]),
                "to_id": str(row[1]),
                "relation_type": "CONTRADICTS",
            }
        )
    return pairs


def _search_nodes_by_type(
    db: FalkorDBClient,
    config: MCPServerConfig,
    node_type: str,
    embedding: List[float],
    limit: int,
    max_distance: float,
    owner_id: str,
    *,
    search_type: SearchType,
    include_outdated: bool = False,
    status: Optional[str] = None,
    metadata_filter: Optional[Dict] = None,
    include_ttl: bool = False,
) -> List[Dict]:
    if search_type == "pre_filter":
        return _search_nodes_pre_filter(
            db,
            node_type,
            embedding,
            limit,
            max_distance,
            owner_id,
            include_outdated,
            status,
            metadata_filter,
            include_ttl,
        )
    return _search_nodes_post_filter(
        db,
        config,
        node_type,
        embedding,
        limit,
        max_distance,
        owner_id,
        include_outdated,
        status,
        metadata_filter,
        include_ttl,
    )


def _embedding_results(
    db: FalkorDBClient,
    config: MCPServerConfig,
    *,
    query: str,
    owner_id: str,
    node_types: List[str],
    candidate_limit: int,
    similarity_threshold: float,
    search_type: SearchType,
    include_outdated: bool,
    status: Optional[str],
    metadata_filter: Optional[Dict],
    include_ttl: bool,
) -> Optional[List[Dict]]:
    """Vector search plus the identifier/recency blend. None when no vector."""
    db.ensure_search_indexes_if_missing(owner_id=owner_id)
    embedding = db.get_embedding(query, kind="query")
    if not embedding:
        return None
    max_distance = 1.0 - similarity_threshold
    results: List[Dict] = []
    for node_type in node_types:
        results.extend(
            _search_nodes_by_type(
                db,
                config,
                node_type,
                embedding,
                candidate_limit,
                max_distance,
                owner_id,
                search_type=search_type,
                include_outdated=include_outdated,
                status=status,
                metadata_filter=metadata_filter,
                include_ttl=include_ttl,
            )
        )
    return rerank_results(
        results,
        query=query,
        now_ms=int(time.time() * 1000),
        lexical_weight=config.search_lexical_weight,
        recency_weight=config.search_recency_weight,
        half_life_days=config.search_recency_half_life_days,
    )


def _lexical_results(
    db: FalkorDBClient,
    *,
    query: str,
    mode: str,
    owner_id: str,
    node_types: List[str],
    candidate_limit: int,
    include_outdated: bool,
    status: Optional[str],
    metadata_filter: Optional[Dict],
    include_ttl: bool,
) -> List[Dict]:
    """exact / fuzzy / bm25 over the owner's filtered rows (see lexical_search)."""
    assert mode in LEXICAL_MODES
    results: List[Dict] = []
    for node_type in node_types:
        key = (
            owner_id,
            node_type,
            include_outdated,
            status,
            include_ttl,
            repr(sorted((metadata_filter or {}).items())),
        )

        def _load(node_type: str = node_type) -> list:
            return db.text_rows(
                node_type=node_type,
                owner_id=owner_id,
                include_outdated=include_outdated,
                status=status,
                metadata_filter=metadata_filter,
                include_ttl=include_ttl,
            )

        index = cached_index(db, key, _load)
        for row, score in index.search(query, mode, candidate_limit):
            item = _rows_to_search_results([row])[0]
            item.pop("similarity", None)
            item["score"] = round(float(score), 6)
            item["match"] = mode
            results.append(item)
    results.sort(key=lambda r: (-r["score"], r["node_id"]))
    return results[:candidate_limit]


def _fuse(per_mode: Dict[str, List[Dict]], limit: int) -> List[Dict]:
    """RRF over the per-mode rankings. Keeps the first-seen row per node."""
    scores = rrf_fuse([[r["node_id"] for r in rows] for rows in per_mode.values()])
    by_id: Dict[str, Dict] = {}
    hits: Dict[str, List[str]] = {}
    for one, rows in per_mode.items():
        for r in rows:
            by_id.setdefault(r["node_id"], dict(r))
            hits.setdefault(r["node_id"], []).append(one)
    fused = []
    for node_id, score in scores.items():
        item = by_id[node_id]
        item["score"] = round(score, 6)
        item["match"] = "+".join(hits[node_id])
        fused.append(item)
    fused.sort(key=lambda r: (-r["score"], r["node_id"]))
    return fused[:limit]


@mcp_handler
def search(
    db: FalkorDBClient,
    config: MCPServerConfig,
    *,
    query: str,
    owner_id: str = "default",
    limit: Optional[int] = None,
    node_types: Optional[List[str]] = None,
    status: Optional[str] = None,
    similarity_threshold: Optional[float] = None,
    include_outdated: bool = False,
    search_type: Optional[str] = None,
    metadata_filter: Optional[Dict] = None,
    compact: bool = False,
    include_ttl: bool = False,
    search_mode: Optional[str] = None,
    hybrid_modes: Optional[List[str]] = None,
    rerank: Optional[bool] = None,
) -> Dict:
    """Search nodes. search_mode: embedding (default) | exact | fuzzy | bm25 | hybrid."""
    owner_id = normalize_owner_id(owner_id)
    try:
        resolved_search_type = normalize_search_type(
            search_type if search_type is not None else config.default_search_type
        )
        mode = normalize_search_mode(
            search_mode if search_mode is not None else config.default_search_mode
        )
        fused_modes = normalize_hybrid_modes(hybrid_modes) if mode == "hybrid" else ()
        use_reranker = (
            config.search_reranker_enabled if rerank is None else bool(rerank)
        )
        # Validate filter keys early (raises ValueError → validation error).
        build_metadata_filter_clauses(metadata_filter)
    except ValueError as exc:
        return error_response(str(exc), code="memory_validation_error")

    cache_key = hash_query(
        query,
        owner_id=owner_id,
        limit=limit,
        node_types=node_types,
        status=status,
        similarity_threshold=similarity_threshold,
        include_outdated=include_outdated,
        search_type=resolved_search_type,
        metadata_filter=metadata_filter,
        include_ttl=include_ttl,
        search_mode=mode,
        hybrid_modes=list(fused_modes),
        rerank=use_reranker,
    )

    def _decorate(payload: Dict) -> Dict:
        return decorate_search_response(
            payload,
            compact=compact,
            snippet_chars=config.compact_snippet_chars,
            token_budget=config.compact_recall_token_budget,
            query=query,
            owner_id=owner_id,
            metadata_filter=metadata_filter,
        )

    if cached := db.cache.get_search(cache_key):
        return _decorate(cached)

    limit = max(1, min(limit or config.default_search_limit, config.max_search_limit))
    similarity_threshold = (
        similarity_threshold
        if similarity_threshold is not None
        else config.semantic_similarity_threshold
    )

    if node_types is None:
        node_types = ["Fact", "Entity"]
    else:
        node_types = [
            nt.capitalize()
            for nt in node_types
            if nt.capitalize() in ["Fact", "Entity"]
        ]
        if not node_types:
            node_types = ["Fact", "Entity"]

    filters = dict(
        include_outdated=include_outdated,
        status=status,
        metadata_filter=metadata_filter,
        include_ttl=include_ttl,
    )
    candidate_limit = min(config.max_search_limit, max(limit * 5, limit))
    if use_reranker:
        candidate_limit = max(candidate_limit, min(config.search_rerank_top_k, 1000))
    modes = fused_modes if mode == "hybrid" else (mode,)
    per_mode: Dict[str, List[Dict]] = {}
    for one in modes:
        if one == "embedding":
            ranked = _embedding_results(
                db,
                config,
                query=query,
                owner_id=owner_id,
                node_types=node_types,
                candidate_limit=candidate_limit,
                similarity_threshold=similarity_threshold,
                search_type=resolved_search_type,
                **filters,
            )
            if ranked is None:
                return error_response(
                    "Embedding service returned no vector; search was not run",
                    code="memory_embedding_error",
                )
        else:
            ranked = _lexical_results(
                db,
                query=query,
                mode=one,
                owner_id=owner_id,
                node_types=node_types,
                candidate_limit=candidate_limit,
                **filters,
            )
        per_mode[one] = ranked

    if mode == "hybrid":
        results = _fuse(per_mode, candidate_limit)
    else:
        results = per_mode[mode]

    if use_reranker and results:
        from graph_memory_mcp.graph_memory.reranker import rerank as ce_rerank

        results = ce_rerank(
            query,
            results,
            model_name=config.search_reranker_model,
            top_k=max(limit, config.search_rerank_top_k),
        )
    results = results[:limit]

    facts = [n for n in results if n.get("node_type") == "Fact"]
    entities = [n for n in results if n.get("node_type") == "Entity"]
    contradiction_pairs = _contradiction_pairs(db, owner_id, results)

    final_response = success_response(
        results=results,
        facts=facts,
        entities=entities,
        contradiction_pairs=contradiction_pairs,
    )

    db.cache.set_search(cache_key, final_response)
    touch_nodes(db, [r["node_id"] for r in results], owner_id)

    return _decorate(final_response)


@mcp_handler
def find_similar(
    db: FalkorDBClient,
    config: MCPServerConfig,
    *,
    fact_id: str,
    owner_id: str = "default",
    limit: int = 5,
    similarity_threshold: Optional[float] = None,
) -> Dict:
    """Find similar facts to a given fact."""
    owner_id = normalize_owner_id(owner_id)
    fact_id = require_node_id(fact_id, "fact_id")
    limit = max(1, min(int(limit), config.max_search_limit))
    db.ensure_vector_indexes_if_missing(owner_id=owner_id)
    similarity_threshold = (
        similarity_threshold
        if similarity_threshold is not None
        else config.semantic_similarity_threshold
    )

    found, raw_embedding = db.fact_embedding(fact_id, owner_id)
    if not found:
        return error_response(f"Fact {fact_id} not found", code="memory_not_found")

    embedding = parse_embedding_value(raw_embedding)
    if not embedding:
        return success_response(similar_facts=[])

    ann_k = _vector_ann_k(limit + 1, _count_labeled_nodes(db, "Fact", owner_id), config)
    rows = db.ann_rows(
        node_type="Fact",
        embedding=embedding,
        owner_id=owner_id,
        ann_k=ann_k,
        limit=limit,
        max_distance=1.0 - similarity_threshold,
        include_outdated=True,
        exclude_node_id=fact_id,
    )

    similar_facts = []
    for row in _rows_to_search_results(rows):
        similar_facts.append(
            {
                "node_id": row["node_id"],
                "text": row["text"],
                "status": row["status"],
                "created_at": row["created_at"],
                "metadata": row["metadata"],
                "similarity": row["similarity"],
            }
        )

    return success_response(similar_facts=similar_facts)
