"""ArcadeDB vector search.

Each owner has a separate database, so the HNSW index in that database is
only that owner's. ``vector.neighbors`` runs inside it.

Status, expiry, and metadata are not a post-filter. Those predicates become a
RID list passed as ``vector.neighbors`` ``filter``, so the HNSW walk only
visits matching records. ``db.index.vector.queryNodes`` is not used.

``efSearch`` is raised when that RID list is very selective. An unfiltered
ANN call uses the HNSW index directly (no filter list).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from graph_memory_mcp.graph_memory.backends.metadata_filter import (
    validate_metadata_filter,
)

# A filter is very selective when it keeps under 5% of the owner's labeled
# rows, or when fewer than this many RIDs remain (the beam would starve).
_SELECTIVITY = 0.05
_MIN_RIDS = 32


def _needs_attribute_filter(
    node_type: str,
    *,
    include_outdated: bool,
    status: Optional[str],
    metadata_filter: Optional[Dict[str, Any]],
    exclude_node_id: Optional[str],
) -> bool:
    if status or metadata_filter or exclude_node_id:
        return True
    if not include_outdated:
        return True
    return False


def _attribute_where(
    node_type: str,
    *,
    include_outdated: bool,
    status: Optional[str],
    metadata_filter: Optional[Dict[str, Any]],
    exclude_node_id: Optional[str],
    now_ms: int,
    include_ttl: bool = False,
    require_embedding: bool = True,
) -> tuple[str, Dict[str, Any]]:
    """SQL predicates for status, expiry, metadata, and an excluded uid.

    Default Fact search hides every node that has ``expires_at``.
    ``include_ttl`` also keeps facts that have not expired yet.
    ``include_outdated`` drops the expiry predicate.
    """
    validate_metadata_filter(metadata_filter)
    parts: list[str] = [
        "embedding IS NOT NULL" if require_embedding else "text IS NOT NULL"
    ]
    params: Dict[str, Any] = {}
    if status:
        parts.append("status = :status")
        params["status"] = status
    elif not include_outdated:
        parts.append("(status IS NULL OR status = 'active')")
    if (
        node_type == "Fact"
        and not include_outdated
        and (status is None or status == "active")
    ):
        if include_ttl:
            parts.append(f"(expires_at IS NULL OR expires_at > {int(now_ms)})")
        else:
            parts.append("expires_at IS NULL")
    if metadata_filter:
        for key, prop in (
            ("project", "project"),
            ("created_by", "created_by"),
            ("type", "meta_type"),
        ):
            if (value := metadata_filter.get(key)) is not None:
                parts.append(f"{prop} = :mf_{key}")
                params[f"mf_{key}"] = str(value)
        tags = metadata_filter.get("tags")
        if tags is not None:
            tags = [tags] if isinstance(tags, str) else list(tags)
            for i, tag in enumerate(tags):
                parts.append(f":mf_tag{i} IN tags")
                params[f"mf_tag{i}"] = str(tag)
        if (conf_min := metadata_filter.get("confidence_min")) is not None:
            parts.append("confidence >= :mf_conf")
            params["mf_conf"] = float(conf_min)
    if exclude_node_id is not None:
        parts.append("uid <> :exclude_uid")
        params["exclude_uid"] = str(exclude_node_id)
    return " AND ".join(parts), params


def _ef_search(rid_count: int, labeled: int, k: int, ann_k: int | None) -> int | None:
    if labeled <= 0:
        return max(500, k * 50)
    selective = rid_count < _MIN_RIDS or (rid_count / labeled) < _SELECTIVITY
    if not selective:
        return None
    return max(500, k * 50, int(ann_k or 0))


def _neighbor_view(item: Dict[str, Any]) -> Dict[str, Any]:
    record = item.get("record") if isinstance(item.get("record"), dict) else {}
    view = dict(record)
    for key, value in item.items():
        if key != "record":
            view[key] = value
    return view


def _row_from_neighbor(view: Dict[str, Any], node_type: str) -> List[Any]:
    return [
        view.get("uid"),
        node_type,
        view.get("text"),
        view.get("status"),
        view.get("created_at"),
        view.get("metadata_str"),
        float(view.get("distance") or 0.0),
    ]


def neighbor_rows(
    store: Any,
    *,
    node_type: str,
    embedding: List[float],
    owner_id: str,
    limit: int,
    max_distance: float,
    include_outdated: bool = False,
    status: Optional[str] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
    exclude_node_id: Optional[str] = None,
    ann_k: int | None = None,
    include_ttl: bool = False,
) -> List[List[Any]]:
    """Top-k cosine neighbors from this owner's own HNSW index."""
    if node_type not in ("Fact", "Entity"):
        raise ValueError(f"Vector search is not indexed for {node_type}")
    with store.owner_scope(owner_id):
        return _neighbor_rows(
            store,
            node_type=node_type,
            embedding=embedding,
            owner_id=owner_id,
            limit=limit,
            max_distance=max_distance,
            include_outdated=include_outdated,
            status=status,
            metadata_filter=metadata_filter,
            exclude_node_id=exclude_node_id,
            ann_k=ann_k,
            include_ttl=include_ttl,
        )


def _neighbor_rows(
    store: Any,
    *,
    node_type: str,
    embedding: List[float],
    owner_id: str,
    limit: int,
    max_distance: float,
    include_outdated: bool,
    status: Optional[str],
    metadata_filter: Optional[Dict[str, Any]],
    exclude_node_id: Optional[str],
    ann_k: int | None,
    include_ttl: bool = False,
) -> List[List[Any]]:
    owner = store.owner_literal(owner_id)
    k = max(int(limit), 1)
    dim = len(embedding)
    store.ensure_vector_index(node_type, dim)
    vector = [float(v) for v in embedding]
    now_ms = int(time.time() * 1000)
    filtered = _needs_attribute_filter(
        node_type,
        include_outdated=include_outdated,
        status=status,
        metadata_filter=metadata_filter,
        exclude_node_id=exclude_node_id,
    )
    opts: Dict[str, Any] = {"maxDistance": float(max_distance)}
    plan: Dict[str, Any] = {
        "filtered": filtered,
        "ef_search": None,
        "rid_count": None,
        "labeled": None,
        "k": k,
    }
    if filtered:
        where, params = _attribute_where(
            node_type,
            include_outdated=include_outdated,
            status=status,
            metadata_filter=metadata_filter,
            exclude_node_id=exclude_node_id,
            now_ms=now_ms,
            include_ttl=include_ttl,
        )
        rid_rows = store.query(
            f"SELECT @rid AS rid FROM {node_type} "
            f"WHERE owner_id = '{owner}' AND {where}",
            params,
        )
        rids = [str(row.get("rid")) for row in rid_rows if row.get("rid")]
        labeled = int(store.count_labeled(node_type, owner_id))
        plan["rid_count"] = len(rids)
        plan["labeled"] = labeled
        if not rids:
            store.last_vector_search = plan
            return []
        opts["filter"] = rids
        ef = _ef_search(len(rids), labeled, k, ann_k)
        plan["ef_search"] = ef
        if ef is not None:
            opts["efSearch"] = int(ef)
    store.last_vector_search = plan
    index_name = f"{node_type}[embedding]"
    rows = store.query(
        f"SELECT vector.neighbors('{index_name}', :q, :k, :opts) AS neighbors "
        f"FROM {node_type} WHERE owner_id = '{owner}' LIMIT 1",
        {"q": vector, "k": k, "opts": opts},
    )
    if not rows:
        return []
    neighbors = rows[0].get("neighbors") or []
    out: List[List[Any]] = []
    for item in neighbors:
        if not isinstance(item, dict):
            continue
        view = _neighbor_view(item)
        if view.get("owner_id") != owner_id:
            continue
        if exclude_node_id is not None and str(view.get("uid")) == str(exclude_node_id):
            continue
        out.append(_row_from_neighbor(view, node_type))
        if len(out) >= k:
            break
    return out


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
    """Owner candidate rows for lexical search, in the vector row shape (distance 0)."""
    if node_type not in ("Fact", "Entity"):
        raise ValueError(f"Text search is not supported for {node_type}")
    with store.owner_scope(owner_id):
        owner = store.owner_literal(owner_id)
        where, params = _attribute_where(
            node_type,
            include_outdated=include_outdated,
            status=status,
            metadata_filter=metadata_filter,
            exclude_node_id=None,
            now_ms=int(time.time() * 1000),
            include_ttl=include_ttl,
            require_embedding=False,
        )
        rows = store.query(
            f"SELECT uid, text, status, created_at, metadata_str FROM {node_type} "
            f"WHERE owner_id = '{owner}' AND {where}",
            params,
        )
    return [
        [
            r.get("uid"),
            node_type,
            r.get("text"),
            r.get("status"),
            r.get("created_at"),
            r.get("metadata_str"),
            0.0,
        ]
        for r in rows
    ]
