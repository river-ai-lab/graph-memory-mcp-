"""Falkor Cypher for owner-scoped semantic search (exact cosine and ANN)."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

from graph_memory_mcp.graph_memory.backends.metadata_filter import (
    validate_metadata_filter,
)

SearchType = Literal["pre_filter", "post_filter"]


def build_metadata_filter_clauses(
    metadata_filter: Optional[Dict[str, Any]],
    *,
    node_alias: str = "node",
    param_prefix: str = "mf_",
) -> Tuple[str, Dict[str, Any]]:
    """Cypher clauses over promoted metadata properties.

    Supported: `project` / `created_by` / `type` (equality), `tags`
    (membership, all must be present), `confidence_min` (>=).
    Raises ValueError on unsupported keys. Runs natively in FalkorDB — no
    client-side filtering.
    """
    if not metadata_filter:
        return "", {}
    validate_metadata_filter(metadata_filter)
    clauses: list[str] = []
    params: Dict[str, Any] = {}
    for key, prop in (
        ("project", "project"),
        ("created_by", "created_by"),
        ("type", "meta_type"),
    ):
        if (value := metadata_filter.get(key)) is not None:
            clauses.append(f" AND {node_alias}.{prop} = ${param_prefix}{key}")
            params[f"{param_prefix}{key}"] = str(value)
    tags = metadata_filter.get("tags")
    if tags is not None:
        tags = [tags] if isinstance(tags, str) else list(tags)
        for i, tag in enumerate(tags):
            clauses.append(f" AND ${param_prefix}tag{i} IN {node_alias}.tags")
            params[f"{param_prefix}tag{i}"] = str(tag)
    if (conf_min := metadata_filter.get("confidence_min")) is not None:
        clauses.append(f" AND {node_alias}.confidence >= ${param_prefix}conf")
        params[f"{param_prefix}conf"] = float(conf_min)
    return "".join(clauses), params


def normalize_search_type(
    value: str | None, *, default: SearchType = "pre_filter"
) -> SearchType:
    """Normalize MCP search_type to pre_filter or post_filter."""
    if value is None or not str(value).strip():
        return default
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"pre_filter", "prefilter", "pre"}:
        return "pre_filter"
    if normalized in {"post_filter", "postfilter", "post"}:
        return "post_filter"
    raise ValueError(
        f"Invalid search_type {value!r}; expected 'pre_filter' or 'post_filter'"
    )


def _property_filter_clauses(
    node_type: str,
    *,
    include_outdated: bool,
    status: Optional[str],
    node_alias: str = "node",
    include_ttl: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    clauses: list[str] = []
    params: Dict[str, Any] = {}
    if status:
        clauses.append(f" AND {node_alias}.status = $status")
        params["status"] = status
    elif not include_outdated:
        clauses.append(
            f" AND ({node_alias}.status IS NULL OR {node_alias}.status = 'active')"
        )

    if node_type == "Fact" and (status is None or status == "active"):
        if include_outdated:
            pass
        elif include_ttl:
            clauses.append(
                f" AND ({node_alias}.expires_at IS NULL OR {node_alias}.expires_at > timestamp())"
            )
        else:
            clauses.append(f" AND {node_alias}.expires_at IS NULL")
    return "".join(clauses), params


def build_owner_scoped_similarity_query(
    *,
    node_type: str,
    embedding: List[float],
    owner_id: str,
    limit: int,
    max_distance: float,
    include_outdated: bool = False,
    status: Optional[str] = None,
    exclude_node_id: Optional[str] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
    include_ttl: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """Cypher + params: owner-scoped nodes, exact vec.cosineDistance, ORDER BY score."""
    property_filters, params = _property_filter_clauses(
        node_type,
        include_outdated=include_outdated,
        status=status,
        include_ttl=include_ttl,
    )
    meta_clauses, meta_params = build_metadata_filter_clauses(metadata_filter)
    property_filters += meta_clauses
    params.update(meta_params)
    exclude_clause = ""
    if exclude_node_id is not None:
        exclude_clause = " AND node.uid <> $exclude_uid"
        params["exclude_uid"] = str(exclude_node_id)

    params.update(
        {
            "owner_id": owner_id,
            "embedding": [float(v) for v in embedding],
            "max_distance": float(max_distance),
        }
    )

    query = f"""
    MATCH (node:{node_type})
    WHERE node.owner_id = $owner_id
      AND node.embedding IS NOT NULL
    {property_filters}
    WITH node, vec.cosineDistance(node.embedding, vecf32($embedding)) AS score
    WHERE score <= $max_distance{exclude_clause}
    RETURN
        node.uid as node_id,
        '{node_type}' as node_type,
        node.text as text,
        node.status as status,
        node.created_at as created_at,
        node.metadata_str as metadata_str,
        score
    ORDER BY score ASC
    LIMIT {int(limit)}
    """
    return query, params


def count_labeled(store: Any, node_type: str, owner_id: str) -> int:
    rows = store.rows(
        f"MATCH (n:{node_type}) WHERE n.owner_id = $owner_id RETURN count(n)",
        {"owner_id": owner_id},
    )
    if not rows:
        return 0
    return int(rows[0][0])


def similarity_rows(
    store: Any,
    *,
    node_type: str,
    embedding: List[float],
    owner_id: str,
    limit: int,
    max_distance: float,
    include_outdated: bool = False,
    status: Optional[str] = None,
    exclude_node_id: Optional[str] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
    include_ttl: bool = False,
) -> List[List[Any]]:
    query, params = build_owner_scoped_similarity_query(
        node_type=node_type,
        embedding=embedding,
        owner_id=owner_id,
        limit=limit,
        max_distance=max_distance,
        include_outdated=include_outdated,
        status=status,
        exclude_node_id=exclude_node_id,
        metadata_filter=metadata_filter,
        include_ttl=include_ttl,
    )
    return store.rows(query, params)


def ann_rows(
    store: Any,
    *,
    node_type: str,
    embedding: List[float],
    owner_id: str,
    ann_k: int,
    limit: int,
    max_distance: float,
    include_outdated: bool = False,
    status: Optional[str] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
    exclude_node_id: Optional[str] = None,
    include_ttl: bool = False,
) -> List[List[Any]]:
    """Falkor HNSW lookup, then owner/status filters."""
    params: Dict[str, Any] = {
        "embedding": embedding,
        "max_distance": float(max_distance),
        "owner_id": owner_id,
    }
    filters, filter_params = _property_filter_clauses(
        node_type,
        include_outdated=include_outdated,
        status=status,
        include_ttl=include_ttl,
    )
    params.update(filter_params)
    meta_clauses, meta_params = build_metadata_filter_clauses(metadata_filter)
    filters += meta_clauses
    params.update(meta_params)
    if exclude_node_id is not None:
        filters += " AND node.uid <> $exclude_uid"
        params["exclude_uid"] = str(exclude_node_id)

    query = f"""
    CALL db.idx.vector.queryNodes('{node_type}', 'embedding', {int(ann_k)}, vecf32($embedding))
    YIELD node, score
    WHERE score <= $max_distance
      AND node.owner_id = $owner_id
    {filters}
    RETURN
        node.uid as node_id,
        '{node_type}' as node_type,
        node.text as text,
        node.status as status,
        node.created_at as created_at,
        node.metadata_str as metadata_str,
        score
    LIMIT {int(limit)}
    """
    return store.rows(query, params)


def text_rows(
    store: Any,
    *,
    node_type: str,
    owner_id: str,
    include_outdated: bool = False,
    status: Optional[str] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
    include_ttl: bool = False,
) -> List[List[Any]]:
    """Owner-scoped candidate rows for lexical search (same filters, no vector).

    Rows have the vector row shape with distance 0.0 in the last column.
    """
    property_filters, params = _property_filter_clauses(
        node_type,
        include_outdated=include_outdated,
        status=status,
        include_ttl=include_ttl,
    )
    meta_clauses, meta_params = build_metadata_filter_clauses(metadata_filter)
    params.update(meta_params)
    params["owner_id"] = owner_id
    query = f"""
    MATCH (node:{node_type})
    WHERE node.owner_id = $owner_id
      AND node.text IS NOT NULL
    {property_filters}{meta_clauses}
    RETURN
        node.uid as node_id,
        '{node_type}' as node_type,
        node.text as text,
        node.status as status,
        node.created_at as created_at,
        node.metadata_str as metadata_str,
        0.0 as score
    """
    return store.rows(query, params)


def fact_embedding(store: Any, fact_id: str, owner_id: str) -> tuple[bool, Any]:
    """Return (found, embedding). Embedding may be empty when the fact exists."""
    rows = store.rows(
        """
        MATCH (n:Fact)
        WHERE n.uid = $fact_id AND n.owner_id = $owner_id
        RETURN n.embedding as embedding
        """,
        {"fact_id": fact_id, "owner_id": owner_id},
    )
    if not rows:
        return False, None
    return True, rows[0][0]
