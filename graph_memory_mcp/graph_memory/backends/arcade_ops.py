"""ArcadeDB SQL for the handler and job method surface.

Cypher fragments that handlers build for Falkor (SET clauses, edge property
assignments, relation patterns, page clauses) are interpreted here. They are
not sent to ArcadeDB as Cypher.

Each public method that carries an owner runs inside that owner's database.
"""

from __future__ import annotations

import functools
import inspect
import logging
import re
import time
from typing import Any

logger = logging.getLogger(__name__)

VERTEX_TYPES = ("Fact", "Entity", "Collection", "User", "Project", "FactVersion")
_SYSTEM = {"@rid", "@type", "@cat", "@in", "@out"}
_ASSIGN = re.compile(
    r"^n\.([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
    r"(timestamp\(\)|NULL|vecf32\(\$([A-Za-z_][A-Za-z0-9_]*)\)|\$([A-Za-z_][A-Za-z0-9_]*))\s*$"
)
_EDGE_PROP = re.compile(r"r\.([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\$([A-Za-z_][A-Za-z0-9_]*)")
_REL = re.compile(r"^\[r(?::([A-Z][A-Z0-9_]*))?\]$")
_PAGE = re.compile(r"^\s*SKIP\s+(\d+)\s+LIMIT\s+(\d+)\s*$", re.IGNORECASE)
_TOUCHED = re.compile(
    r"AND coalesce\(n\.updated_at, n\.created_at\)\s*(>=|<)\s*(\d+)\s*$"
)
_RID = re.compile(r"^#\d+:\d+$")


def _now_ms() -> int:
    return int(time.time() * 1000)


def _rid(value: Any) -> str:
    text = str(value or "")
    if not _RID.match(text):
        raise ValueError(f"Invalid record id: {value!r}")
    return text


def _plain(record: dict) -> dict:
    return {
        key: value
        for key, value in record.items()
        if key not in _SYSTEM and not str(key).startswith("@")
    }


def _node_row(record: dict, node_type: str) -> list:
    return [
        record.get("uid"),
        node_type,
        record.get("text"),
        record.get("description"),
        record.get("status"),
        record.get("created_at"),
        record.get("updated_at"),
        record.get("metadata_str"),
        record.get("shared_with_ids") or [],
        record.get("ttl_days"),
        record.get("expires_at"),
        record.get("type"),
        record.get("source_str"),
        record.get("created_by"),
        record.get("updated_by"),
    ]


def _slice_page(rows: list, page_clause: str | None) -> list:
    match = _PAGE.match(page_clause or "")
    if not match:
        return rows
    start = int(match.group(1))
    limit = int(match.group(2))
    return rows[start : start + limit]


_NO_OWNER_SCOPE = frozenset(
    {
        "embedding_literal",
        "updated_at_assignment",
        "similarity_rows",
        "ann_rows",
        "list_owners",
        "delete_owner_graph",
        "prune_empty_owner_graphs",
        "fact_neighbor_rows",
    }
)


def _scoped_kw(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        bound = inspect.signature(fn).bind(self, *args, **kwargs)
        bound.apply_defaults()
        with self.owner_scope(str(bound.arguments["owner_id"])):
            return fn(self, *args, **kwargs)

    return wrapper


def _scoped_params(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        bound = inspect.signature(fn).bind(self, *args, **kwargs)
        bound.apply_defaults()
        owner_id = bound.arguments["params"]["owner_id"]
        with self.owner_scope(str(owner_id)):
            return fn(self, *args, **kwargs)

    return wrapper


def _route_owners(cls):
    """Send each owner-scoped method to that owner's database."""
    for name, fn in list(vars(cls).items()):
        if name.startswith("_") or name in _NO_OWNER_SCOPE or not callable(fn):
            continue
        params = inspect.signature(fn).parameters
        if "owner_id" in params:
            setattr(cls, name, _scoped_kw(fn))
        elif "params" in params:
            setattr(cls, name, _scoped_params(fn))
    return cls


@_route_owners
class ArcadeOps:
    """Methods mixed into the ArcadeDB client."""

    def embedding_literal(self, param: str, present: bool) -> str:
        return f"vecf32(${param})" if present else "NULL"

    def updated_at_assignment(self) -> str:
        return "n.updated_at = timestamp()"

    def ann_rows(self, **kwargs: Any) -> list:
        from graph_memory_mcp.graph_memory.backends.arcade_search import neighbor_rows

        return neighbor_rows(self, **kwargs)

    def similarity_rows(self, **kwargs: Any) -> list:
        return self.ann_rows(**kwargs)

    def text_rows(self, **kwargs: Any) -> list:
        from graph_memory_mcp.graph_memory.backends.arcade_search import text_rows

        return text_rows(self, **kwargs)

    def fact_embedding(self, fact_id: str, owner_id: str) -> tuple[bool, Any]:
        _label, record = self._find(fact_id, owner_id, types=("Fact",))
        if record is None:
            return False, None
        return True, record.get("embedding")

    def list_owners(self) -> list[str]:
        prefix = self.database_prefix + "_"
        owners = []
        for name in self.list_database_names():
            if name.startswith(prefix):
                owners.append(name[len(prefix) :])
        return sorted(set(owners))

    def delete_owner_graph(self, owner_id: str) -> bool:
        """Drop the owner's database, including its vertices and HNSW indexes."""
        name = self.database_name(owner_id)
        self._ready.discard(name)
        self._edge_types.pop(name, None)
        try:
            self._server(f"drop database {name}")
        except Exception as exc:  # noqa: BLE001
            message = str(exc).lower()
            if (
                "not found" in message
                or "does not exist" in message
                or "not available" in message
            ):
                with self._vector_index_lock:
                    self._vector_indexes_ready.discard(name)
                return True
            logger.error("Failed to delete owner database %s: %s", name, exc)
            return False
        with self._vector_index_lock:
            self._vector_indexes_ready.discard(name)
        logger.info("Deleted owner database %s", name)
        return True

    def prune_empty_owner_graphs(self) -> list[str]:
        pruned: list[str] = []
        for owner_id in self.list_owners():
            with self.owner_scope(owner_id):
                empty = True
                for label in VERTEX_TYPES:
                    rows = self.query(f"SELECT count(*) AS c FROM {label}")
                    if rows and int(rows[0].get("c") or 0) > 0:
                        empty = False
                        break
            if empty and self.delete_owner_graph(owner_id):
                pruned.append(owner_id)
        return pruned

    def backfill_all_owners(self) -> None:
        for owner_id in self.list_owners():
            try:
                self.backfill_uids(owner_id=owner_id)
                self.backfill_entity_name_norm(owner_id=owner_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Backfill failed for owner %s: %s", owner_id, exc)

    def backfill_entity_name_norm(self, owner_id: str = "default") -> None:
        owner = self.owner_literal(owner_id)
        rows = self.query(
            "SELECT uid, text FROM Entity "
            f"WHERE owner_id = '{owner}' AND name_norm IS NULL AND text IS NOT NULL"
        )
        for row in rows:
            text = str(row.get("text") or "").strip().lower()
            self.command(
                "UPDATE Entity SET name_norm = :name WHERE uid = :uid "
                f"AND owner_id = '{owner}'",
                {"name": text, "uid": row.get("uid")},
            )

    def backfill_uids(self, batch_size: int = 1000, owner_id: str = "default") -> int:
        del batch_size
        from graph_memory_mcp.graph_memory.utils import new_uid

        owner = self.owner_literal(owner_id)
        total = 0
        for label in VERTEX_TYPES:
            rows = self.query(
                f"SELECT @rid AS rid FROM {label} "
                f"WHERE owner_id = '{owner}' AND uid IS NULL"
            )
            for row in rows:
                self.command(
                    f"UPDATE {label} SET uid = :uid WHERE @rid = {_rid(row.get('rid'))}",
                    {"uid": new_uid()},
                )
                total += 1
        return total

    def touch_nodes(self, node_ids: list, owner_id: str) -> None:
        if not node_ids:
            return
        now = _now_ms()
        for node_id in node_ids:
            label, record = self._find(str(node_id), owner_id)
            if record is None or label is None:
                continue
            count = int(record.get("access_count") or 0) + 1
            self._update(
                label,
                str(node_id),
                owner_id,
                {"access_count": count, "last_accessed_at": now},
            )

    def create_node_row(
        self,
        *,
        node_type: str,
        params: dict[str, Any],
        has_embedding: bool,
        entity_type: str | None,
    ) -> list[list[Any]]:
        self.ensure_vector_indexes_if_missing(owner_id=str(params["owner_id"]))
        self._check_label(node_type)
        owner = self.owner_literal(str(params["owner_id"]))
        self.ensure_ready()
        embedding = params.get("embedding") if has_embedding else None
        if embedding:
            self.ensure_vector_index(node_type, len(embedding))
        now = _now_ms()
        fields = {
            "uid": params.get("uid"),
            "owner_id": owner,
            "text": params.get("text"),
            "description": params.get("description"),
            "status": params.get("status"),
            "created_at": now,
            "updated_at": now,
            "metadata_str": params.get("metadata_str"),
            "tags": params.get("tags"),
            "meta_type": params.get("meta_type"),
            "confidence": params.get("confidence"),
            "project": params.get("project"),
            "created_by": params.get("created_by"),
            "updated_by": params.get("updated_by"),
            "shared_with_ids": params.get("shared_with_ids") or [],
            "ttl_days": params.get("ttl_days"),
            "expires_at": params.get("expires_at"),
            "last_dedup_at": None,
            "source_str": params.get("source_str"),
            "source_ref": params.get("source_ref"),
            "source_type": params.get("source_type"),
            "source_uri": params.get("source_uri"),
            "content_hash": params.get("content_hash"),
            "source_updated_at": params.get("source_updated_at"),
            "embedding": embedding,
        }
        if node_type == "Entity":
            fields["name_norm"] = params.get("name_norm")
            if entity_type:
                fields["type"] = entity_type
        record = self._insert(node_type, fields)
        return [_node_row(record, node_type)]

    def create_nodes_bulk(self, *, node_type: str, rows: list, owner_id: str) -> list:
        self.ensure_vector_indexes_if_missing(owner_id=owner_id)
        self._check_label(node_type)
        owner = self.owner_literal(owner_id)
        self.ensure_ready()
        if rows and rows[0].get("emb"):
            self.ensure_vector_index(node_type, len(rows[0]["emb"]))
        created: list[list[Any]] = []
        with self.transaction() as session:
            now = _now_ms()
            for row in rows:
                fields = {
                    "uid": row.get("uid"),
                    "owner_id": owner,
                    "text": row.get("text"),
                    "description": row.get("description"),
                    "embedding": row.get("emb"),
                    "status": row.get("status"),
                    "created_at": now,
                    "updated_at": now,
                    "metadata_str": row.get("metadata_str"),
                    "tags": row.get("tags"),
                    "meta_type": row.get("meta_type"),
                    "confidence": row.get("confidence"),
                    "project": row.get("project"),
                    "created_by": row.get("created_by"),
                    "updated_by": row.get("updated_by"),
                    "ttl_days": row.get("ttl_days"),
                    "expires_at": row.get("expires_at"),
                    "name_norm": row.get("name_norm"),
                    "last_dedup_at": None,
                }
                self._insert(node_type, fields, session=session)
                created.append([row.get("uid")])
        return created

    def fetch_node_row(self, node_id: str, owner_id: str) -> list[list[Any]]:
        label, record = self._find(node_id, owner_id)
        if record is None or label is None:
            return []
        return [_node_row(record, label)]

    def fetch_version_after(self, node_id: str, owner_id: str, as_of: int) -> list:
        owner = self.owner_literal(owner_id)
        rows = self.query(
            "SELECT text, description, metadata_str, source_str, status, "
            "ttl_days, expires_at, version_timestamp FROM FactVersion "
            f"WHERE fact_id = :fact_id AND owner_id = '{owner}' "
            "AND version_timestamp > :as_of "
            "ORDER BY version_timestamp ASC LIMIT 1",
            {"fact_id": node_id, "as_of": int(as_of)},
        )
        if not rows:
            return []
        row = rows[0]
        return [
            [
                row.get("text"),
                row.get("description"),
                row.get("metadata_str"),
                row.get("source_str"),
                row.get("status"),
                row.get("ttl_days"),
                row.get("expires_at"),
                row.get("version_timestamp"),
            ]
        ]

    def snapshot_fact(self, *, node_id: str, owner_id: str, version_uid: str) -> Any:
        label, record = self._find(node_id, owner_id, types=("Fact",))
        if record is None:
            return None
        now = _now_ms()
        self._insert(
            "FactVersion",
            {
                "uid": version_uid,
                "fact_id": record.get("uid"),
                "owner_id": record.get("owner_id"),
                "text": record.get("text"),
                "description": record.get("description"),
                "metadata_str": record.get("metadata_str"),
                "source_str": record.get("source_str"),
                "shared_with_ids": record.get("shared_with_ids"),
                "status": record.get("status"),
                "ttl_days": record.get("ttl_days"),
                "expires_at": record.get("expires_at"),
                "version_timestamp": now,
                "original_created_at": record.get("created_at"),
            },
        )
        return version_uid

    def apply_node_update(self, *, set_clauses: list[str], params: dict) -> list:
        self.ensure_vector_indexes_if_missing(owner_id=str(params["owner_id"]))
        owner_id = str(params["owner_id"])
        node_id = str(params["node_id"])
        label, record = self._find(node_id, owner_id)
        if record is None or label is None:
            return []
        fields: dict[str, Any] = {}
        for clause in set_clauses:
            match = _ASSIGN.match(clause.strip())
            if not match:
                raise ValueError(f"Unsupported update clause: {clause}")
            prop, kind, vec_param, value_param = match.groups()
            if kind == "timestamp()":
                fields[prop] = _now_ms()
            elif kind == "NULL":
                fields[prop] = None
            elif vec_param:
                fields[prop] = params.get(vec_param)
            else:
                fields[prop] = params.get(value_param)
        if "embedding" in fields and fields["embedding"]:
            self.ensure_vector_index(label, len(fields["embedding"]))
        self._update(label, node_id, owner_id, fields)
        _label, updated = self._find(node_id, owner_id, types=(label,))
        if updated is None:
            return []
        return [_node_row(updated, label)]

    def delete_fact_versions(self, node_id: str, owner_id: str) -> Any:
        owner = self.owner_literal(owner_id)
        self.ensure_ready()
        return self.command(
            f"DELETE FROM FactVersion WHERE fact_id = :fact_id AND owner_id = '{owner}'",
            {"fact_id": node_id},
        )

    def delete_node_row(self, node_id: str, owner_id: str) -> list:
        label, record = self._find(node_id, owner_id)
        if record is None or label is None:
            return [[0]]
        owner = self.owner_literal(owner_id)
        self.command(
            f"DELETE FROM {label} WHERE uid = :uid AND owner_id = '{owner}'",
            {"uid": node_id},
        )
        return [[1]]

    def list_fact_versions(self, node_id: str, owner_id: str) -> list:
        owner = self.owner_literal(owner_id)
        rows = self.query(
            "SELECT @rid AS rid, text, metadata_str, source_str, status, ttl_days, "
            "version_timestamp, original_created_at FROM FactVersion "
            f"WHERE fact_id = :fact_id AND owner_id = '{owner}' "
            "ORDER BY version_timestamp DESC",
            {"fact_id": node_id},
        )
        return [
            [
                row.get("rid"),
                row.get("text"),
                row.get("metadata_str"),
                row.get("source_str"),
                row.get("status"),
                row.get("ttl_days"),
                row.get("version_timestamp"),
                row.get("original_created_at"),
            ]
            for row in rows
        ]

    def fetch_by_source_ref(
        self, *, node_type: str, owner_id: str, source_ref: str
    ) -> list:
        self._check_label(node_type)
        owner = self.owner_literal(owner_id)
        rows = self.query(
            f"SELECT FROM {node_type} WHERE owner_id = '{owner}' "
            "AND source_ref = :source_ref LIMIT 1",
            {"source_ref": source_ref},
        )
        if not rows:
            return []
        return [_node_row(rows[0], node_type)]

    def auto_link_mentions(
        self,
        *,
        relation: str,
        owner_id: str,
        embedding: list,
        max_distance: float,
        node_id: str,
    ) -> Any:
        rows = self.similarity_rows(
            node_type="Entity",
            embedding=embedding,
            owner_id=owner_id,
            limit=10,
            max_distance=max_distance,
            include_outdated=False,
        )
        linked = 0
        for row in rows:
            target = str(row[0])
            if target == str(node_id):
                continue
            created = self.merge_relation(
                rel_type=relation,
                params={
                    "from_id": node_id,
                    "to_id": target,
                    "owner_id": owner_id,
                    "auto_linked": True,
                },
                prop_assignments=", r.auto_linked = $auto_linked",
            )
            if created:
                linked += 1
        return linked

    def add_to_collection(
        self, node_id: str, collection_id: str, owner_id: str
    ) -> list:
        return self.merge_relation(
            rel_type="CONTAINS",
            params={
                "from_id": collection_id,
                "to_id": node_id,
                "owner_id": owner_id,
            },
            prop_assignments="",
        )

    def merge_relation(
        self,
        *,
        rel_type: str,
        params: dict,
        prop_assignments: str,
    ) -> list:
        self.ensure_vector_indexes_if_missing(owner_id=str(params["owner_id"]))
        owner_id = str(params["owner_id"])
        from_label, from_rec = self._find(str(params["from_id"]), owner_id)
        to_label, to_rec = self._find(str(params["to_id"]), owner_id)
        if from_rec is None or to_rec is None:
            return []
        self.ensure_edge_type(rel_type)
        now = _now_ms()
        props = {
            "created_at": now,
            "updated_at": now,
            "created_by": params.get("created_by"),
            "updated_by": params.get("updated_by"),
        }
        for prop, param in _EDGE_PROP.findall(prop_assignments or ""):
            if prop in {"created_by", "created_at", "updated_by", "updated_at"}:
                continue
            props[prop] = params.get(param)
        from_rid = _rid(from_rec.get("@rid"))
        to_rid = _rid(to_rec.get("@rid"))
        existing = self.query(
            f"SELECT @rid AS rid FROM {rel_type} "
            f"WHERE @out = {from_rid} AND @in = {to_rid} LIMIT 1"
        )
        if existing:
            return [[existing[0].get("rid")]]
        created = self._insert_edge(rel_type, from_rid, to_rid, props)
        return [[created.get("@rid")]]

    def merge_triplet(
        self,
        *,
        rel_type: str,
        params: dict,
        subj_emb_expr: str,
        obj_emb_expr: str,
    ) -> list:
        del subj_emb_expr, obj_emb_expr
        self.ensure_vector_indexes_if_missing(owner_id=str(params["owner_id"]))
        owner_id = str(params["owner_id"])
        owner = self.owner_literal(owner_id)
        self.ensure_ready()
        self.ensure_edge_type(rel_type)
        for key in ("subj_emb", "obj_emb"):
            if params.get(key):
                self.ensure_vector_index("Entity", len(params[key]))
        with self.transaction() as session:
            subject = self._merge_entity(
                owner,
                name_norm=str(params["subject_norm"]),
                text=str(params["subject"]),
                uid=str(params["subj_uid"]),
                embedding=params.get("subj_emb"),
                params=params,
                session=session,
            )
            obj = self._merge_entity(
                owner,
                name_norm=str(params["object_norm"]),
                text=str(params["object"]),
                uid=str(params["obj_uid"]),
                embedding=params.get("obj_emb"),
                params=params,
                session=session,
            )
            from_rid = _rid(subject.get("@rid"))
            to_rid = _rid(obj.get("@rid"))
            existing = self.query(
                f"SELECT @rid AS rid FROM {rel_type} "
                f"WHERE @out = {from_rid} AND @in = {to_rid} LIMIT 1",
                session=session,
            )
            if existing:
                rel_id = existing[0].get("rid")
            else:
                now = _now_ms()
                edge = self._insert_edge(
                    rel_type,
                    from_rid,
                    to_rid,
                    {
                        "created_at": now,
                        "updated_at": now,
                        "created_by": params.get("created_by"),
                        "updated_by": params.get("updated_by"),
                    },
                    session=session,
                )
                rel_id = edge.get("@rid")
        return [[subject.get("uid"), obj.get("uid"), rel_id]]

    def link_extracted_from(
        self, *, fact_id: str, subject_id: str, owner_id: str
    ) -> Any:
        return self.merge_relation(
            rel_type="EXTRACTED_FROM",
            params={"from_id": fact_id, "to_id": subject_id, "owner_id": owner_id},
            prop_assignments="",
        )

    def search_triplet_rows(
        self, *, rel_pattern: str, where_sql: str, params: dict, limit: int
    ) -> list:
        del where_sql
        match = _REL.match(rel_pattern.strip())
        if not match:
            raise ValueError(f"Unsupported relation pattern: {rel_pattern}")
        rel_type = match.group(1)
        owner_id = str(params["owner_id"])
        owner = self.owner_literal(owner_id)
        entities = self.query(f"SELECT FROM Entity WHERE owner_id = '{owner}'")
        by_rid = {str(row.get("@rid")): row for row in entities}
        if rel_type:
            edge_types = [rel_type]
        else:
            edge_types = self._known_edge_types()
        found = []
        for edge_type in edge_types:
            edges = self.query(f"SELECT FROM {edge_type}")
            for edge in edges:
                subject = by_rid.get(str(edge.get("@out")))
                obj = by_rid.get(str(edge.get("@in")))
                if subject is None or obj is None:
                    continue
                if (
                    params.get("subject_norm")
                    and subject.get("name_norm") != params["subject_norm"]
                ):
                    continue
                if (
                    params.get("object_norm")
                    and obj.get("name_norm") != params["object_norm"]
                ):
                    continue
                found.append(
                    [
                        subject.get("uid"),
                        subject.get("text"),
                        edge_type,
                        obj.get("uid"),
                        obj.get("text"),
                        edge.get("@rid"),
                    ]
                )
                if len(found) >= int(limit):
                    return found
        return found

    def unlink_relation(self, *, rel_pattern: str, params: dict) -> list:
        match = _REL.match(rel_pattern.strip())
        if not match:
            raise ValueError(f"Unsupported relation pattern: {rel_pattern}")
        rel_type = match.group(1)
        owner_id = str(params["owner_id"])
        _from_label, from_rec = self._find(str(params["from_id"]), owner_id)
        _to_label, to_rec = self._find(str(params["to_id"]), owner_id)
        if from_rec is None or to_rec is None:
            return [[0]]
        from_rid = _rid(from_rec.get("@rid"))
        to_rid = _rid(to_rec.get("@rid"))
        types = [rel_type] if rel_type else self._known_edge_types()
        deleted = 0
        for edge_type in types:
            rows = self.query(
                f"SELECT @rid AS rid FROM {edge_type} "
                f"WHERE @out = {from_rid} AND @in = {to_rid}"
            )
            for row in rows:
                self.command(
                    f"DELETE FROM {edge_type} WHERE @rid = {_rid(row.get('rid'))}"
                )
                deleted += 1
        return [[deleted]]

    def edges_between(self, node_ids: list[str], owner_id: str) -> list:
        wanted = {str(node_id) for node_id in node_ids}
        records = []
        for node_id in wanted:
            label, record = self._find(node_id, owner_id)
            if record is not None:
                records.append((label, record))
        by_rid = {str(record.get("@rid")): record for _label, record in records}
        edges = []
        seen: set[str] = set()
        for label, record in records:
            rid = _rid(record.get("@rid"))
            for edge in self.query(
                f"SELECT expand(outE()) FROM {label} WHERE @rid = {rid}"
            ):
                edge_rid = str(edge.get("@rid"))
                if edge_rid in seen:
                    continue
                target = by_rid.get(str(edge.get("@in")))
                if target is None:
                    continue
                seen.add(edge_rid)
                edges.append(
                    [
                        record.get("uid"),
                        edge.get("@type"),
                        target.get("uid"),
                        _plain(edge),
                    ]
                )
        return edges

    def bfs_init_rows(self, ids: list, owner_id: str) -> list:
        rows = []
        for node_id in ids:
            label, record = self._find(str(node_id), owner_id)
            if record is None or label is None:
                continue
            rows.append(
                [
                    record.get("uid"),
                    label,
                    record.get("text"),
                    (
                        record.get("updated_at")
                        if record.get("updated_at") is not None
                        else record.get("created_at")
                    ),
                    record.get("access_count") or 0,
                    record.get("status"),
                    record.get("created_at"),
                    record.get("metadata_str"),
                ]
            )
        return rows

    def bfs_step_rows(
        self,
        *,
        frontier: list,
        seen: list,
        owner_id: str,
        remaining: int,
        include_outdated: bool,
        include_ttl: bool = False,
        rank_expr: str = "",
        exclude_types: list | None = None,
    ) -> list:
        seen_ids = {str(node_id) for node_id in seen}
        dropped = {str(name) for name in (exclude_types or [])}
        now = _now_ms()
        collected: dict[str, dict] = {}
        for node_id in frontier:
            label, record = self._find(str(node_id), owner_id)
            if record is None or label is None:
                continue
            rid = _rid(record.get("@rid"))
            edges = self.query(
                f"SELECT expand(bothE()) FROM {label} WHERE @rid = {rid}"
            )
            for edge in edges:
                edge_type = str(edge.get("@type") or "")
                if edge_type in dropped:
                    continue
                other_rid = (
                    edge.get("@in")
                    if str(edge.get("@out")) == rid
                    else edge.get("@out")
                )
                other = self._find_rid(str(other_rid), owner_id)
                if other is None:
                    continue
                other_label, other_rec = other
                uid = str(other_rec.get("uid"))
                if uid in seen_ids or not _visible(
                    other_rec, include_outdated, now, include_ttl
                ):
                    continue
                rank = _edge_rank(rank_expr, edge_type)
                slot = collected.get(uid)
                if slot is None:
                    collected[uid] = {
                        "rank": rank,
                        "row": [
                            other_rec.get("uid"),
                            other_label,
                            other_rec.get("text"),
                            (
                                other_rec.get("updated_at")
                                if other_rec.get("updated_at") is not None
                                else other_rec.get("created_at")
                            ),
                            other_rec.get("access_count") or 0,
                            [],
                            other_rec.get("status"),
                            other_rec.get("created_at"),
                            other_rec.get("metadata_str"),
                        ],
                    }
                    slot = collected[uid]
                elif rank < slot["rank"]:
                    slot["rank"] = rank
                parents: list = slot["row"][5]
                parent = str(node_id)
                if parent not in parents:
                    parents.append(parent)
        ordered = sorted(
            collected.values(),
            key=lambda slot: (slot["rank"], str(slot["row"][0])),
        )
        return [slot["row"] for slot in ordered[: int(remaining)]]

    def context_page_rows(
        self,
        *,
        depth: int,
        offset: int,
        limit: int,
        include_outdated: bool,
        node_id: str,
        owner_id: str,
        include_ttl: bool = False,
        exclude_types: list | None = None,
    ) -> list:
        label, center = self._find(node_id, owner_id)
        if center is None or label is None:
            return []
        now = _now_ms()
        dropped = {str(name) for name in (exclude_types or [])}
        visited = {str(center.get("@rid")): (label, center)}
        frontier = [str(center.get("@rid"))]
        for _hop in range(max(0, int(depth))):
            nxt = []
            for rid in frontier:
                current_label, _current = visited[rid]
                edges = self.query(
                    f"SELECT expand(bothE()) FROM {current_label} WHERE @rid = {rid}"
                )
                for edge in edges:
                    if str(edge.get("@type") or "") in dropped:
                        continue
                    other_rid = (
                        edge.get("@in")
                        if str(edge.get("@out")) == rid
                        else edge.get("@out")
                    )
                    other_rid = str(other_rid)
                    if other_rid in visited:
                        continue
                    other = self._find_rid(other_rid, owner_id)
                    if other is None:
                        continue
                    other_label, other_rec = other
                    if str(other_rec.get("uid")) != str(
                        center.get("uid")
                    ) and not _visible(other_rec, include_outdated, now, include_ttl):
                        continue
                    visited[other_rid] = (other_label, other_rec)
                    nxt.append(other_rid)
            frontier = nxt
        ordered = sorted(visited.items(), key=lambda item: item[0])
        page = ordered[int(offset) : int(offset) + int(limit)]
        return [
            [
                rec.get("uid"),
                rec_label,
                rec.get("text"),
                rec.get("status"),
                rec.get("created_at"),
                rec.get("metadata_str"),
            ]
            for _rid, (rec_label, rec) in page
        ]

    def shortest_path_row(
        self,
        *,
        directed: bool,
        max_depth: int,
        from_id: str,
        to_id: str,
        owner_id: str,
    ) -> list:
        from_label, from_rec = self._find(from_id, owner_id)
        to_label, to_rec = self._find(to_id, owner_id)
        if from_rec is None or to_rec is None:
            return []
        from_rid = _rid(from_rec.get("@rid"))
        to_rid = _rid(to_rec.get("@rid"))
        direction = "OUT" if directed else "BOTH"
        rows = self.query(
            f"SELECT shortestPath({from_rid}, {to_rid}, '{direction}') AS path"
        )
        path = []
        if rows:
            raw = rows[0].get("path")
            if isinstance(raw, list):
                path = [str(rid) for rid in raw]
        if not path or len(path) - 1 > int(max_depth):
            return []
        nodes = []
        labels = []
        for rid in path:
            found = self._find_rid(rid, owner_id)
            if found is None:
                return []
            label, record = found
            labels.append(label)
            nodes.append(
                {
                    "node_id": record.get("uid"),
                    "node_type": label,
                    "text": record.get("text"),
                    "status": record.get("status"),
                    "created_at": record.get("created_at"),
                    "updated_at": record.get("updated_at"),
                    "metadata": record.get("metadata_str"),
                }
            )
        relations = []
        for index, rid in enumerate(path[:-1]):
            nxt = path[index + 1]
            edge_type = self._edge_type_between(labels[index], rid, nxt, directed)
            relations.append({"relation_type": edge_type or ""})
        return [[nodes, relations]]

    def label_counts(self, owner_id: str) -> list:
        owner = self.owner_literal(owner_id)
        rows = []
        for label in ("Fact", "Entity"):
            count = self.count_labeled(label, owner)
            if count:
                rows.append([label, count])
        return rows

    def fact_status_counts(self, owner_id: str) -> list:
        owner = self.owner_literal(owner_id)
        rows = self.query(
            "SELECT status, count(*) AS c FROM Fact "
            f"WHERE owner_id = '{owner}' GROUP BY status"
        )
        return [[row.get("status"), int(row.get("c") or 0)] for row in rows]

    def relation_count(self, owner_id: str) -> list:
        total = 0
        for label in ("Fact", "Entity", "Collection"):
            owner = self.owner_literal(owner_id)
            vertices = self.query(
                f"SELECT @rid AS rid FROM {label} WHERE owner_id = '{owner}'"
            )
            for vertex in vertices:
                edges = self.query(
                    f"SELECT expand(outE()) FROM {label} WHERE @rid = {_rid(vertex.get('rid'))}"
                )
                total += len(edges)
        return [[total]]

    def brief_about(self, owner_id: str, source_ref: str) -> list:
        """Latest active no-TTL fact with this source.ref, or an empty list."""
        owner = self.owner_literal(owner_id)
        facts = self.query(
            f"SELECT FROM Fact WHERE owner_id = '{owner}' AND source_ref = :source_ref",
            {"source_ref": source_ref},
        )
        durable = [fact for fact in facts if _durable_active(fact)]
        durable.sort(
            key=lambda fact: -(fact.get("updated_at") or fact.get("created_at") or 0)
        )
        if not durable:
            return []
        fact = durable[0]
        return [[fact.get("uid"), fact.get("text"), fact.get("source_ref")]]

    def brief_top_facts(self, owner_id: str, limit: int, about_ref: str = "") -> list:
        """Recent active facts with no TTL. Not a degree ranking."""
        owner = self.owner_literal(owner_id)
        facts = self.query(
            "SELECT FROM Fact WHERE owner_id = "
            f"'{owner}' AND (status IS NULL OR status = 'active')"
        )
        recent = []
        for fact in facts:
            if not _durable_active(fact):
                continue
            if about_ref and str(fact.get("source_ref") or "") == about_ref:
                continue
            recent.append(fact)
        recent.sort(key=lambda fact: -(fact.get("created_at") or 0))
        return [
            [fact.get("uid"), fact.get("text"), fact.get("created_at")]
            for fact in recent[: int(limit)]
        ]

    def _vertices_by_rid(self, owner: str) -> dict[str, dict]:
        by_rid: dict[str, dict] = {}
        for label in VERTEX_TYPES:
            for row in self.query(f"SELECT FROM {label} WHERE owner_id = '{owner}'"):
                by_rid[str(row.get("@rid"))] = row
        return by_rid

    def brief_contradictions(self, owner_id: str) -> list:
        owner = self.owner_literal(owner_id)
        if "CONTRADICTS" not in self._known_edge_types():
            return []
        by_rid = self._vertices_by_rid(owner)
        rows = []
        for edge in self.query("SELECT FROM CONTRADICTS"):
            left = by_rid.get(str(edge.get("@out")))
            right = by_rid.get(str(edge.get("@in")))
            if left is None or right is None:
                continue
            if not _durable_active(left) or not _durable_active(right):
                continue
            if str(left.get("uid")) >= str(right.get("uid")):
                continue
            rows.append(
                [left.get("uid"), left.get("text"), right.get("uid"), right.get("text")]
            )
            if len(rows) >= 10:
                break
        return rows

    def contradicts_among(self, owner_id: str, node_ids: list) -> list:
        """Existing CONTRADICTS edges whose both ends are in node_ids."""
        wanted = {str(node_id) for node_id in node_ids}
        if len(wanted) < 2 or "CONTRADICTS" not in self._known_edge_types():
            return []
        owner = self.owner_literal(owner_id)
        by_rid = self._vertices_by_rid(owner)
        pairs = []
        for edge in self.query("SELECT FROM CONTRADICTS"):
            left = by_rid.get(str(edge.get("@out")))
            right = by_rid.get(str(edge.get("@in")))
            if left is None or right is None:
                continue
            left_id = str(left.get("uid"))
            right_id = str(right.get("uid"))
            if left_id not in wanted or right_id not in wanted:
                continue
            if left_id > right_id:
                left_id, right_id = right_id, left_id
            pair = [left_id, right_id]
            if pair not in pairs:
                pairs.append(pair)
        return pairs

    def _stale_fact_rows(self, owner_id: str, cutoff_ms: int) -> list:
        owner = self.owner_literal(owner_id)
        facts = self.query(
            "SELECT FROM Fact WHERE owner_id = "
            f"'{owner}' AND (status IS NULL OR status = 'active')"
        )
        stale = []
        for fact in facts:
            if fact.get("expires_at") is not None:
                continue
            touched = fact.get("last_accessed_at")
            if touched is None:
                touched = fact.get("created_at")
            if touched is not None and int(touched) < int(cutoff_ms):
                stale.append(fact)
        stale.sort(
            key=lambda fact: int(
                fact.get("last_accessed_at") or fact.get("created_at") or 0
            )
        )
        return [
            [
                fact.get("uid"),
                fact.get("text"),
                fact.get("last_accessed_at"),
                fact.get("access_count"),
            ]
            for fact in stale
        ]

    def brief_stale_count(self, owner_id: str, cutoff_ms: int) -> int:
        return len(self._stale_fact_rows(owner_id, cutoff_ms))

    def brief_stale_facts(
        self, owner_id: str, cutoff_ms: int, *, limit: int | None = 10
    ) -> list:
        rows = self._stale_fact_rows(owner_id, cutoff_ms)
        if limit is None:
            return rows
        return rows[: int(limit)]

    def export_node_rows(
        self, owner_id: str, page_clause: str, include_versions: bool = True
    ) -> list:
        owner = self.owner_literal(owner_id)
        rows = []
        # Filter versions before slicing so pages neither repeat nor skip rows.
        labels = (
            ("Fact", "Entity", "FactVersion")
            if include_versions
            else ("Fact", "Entity")
        )
        for label in labels:
            records = self.query(f"SELECT FROM {label} WHERE owner_id = '{owner}'")
            for record in records:
                rows.append([label, _plain(record)])
        rows.sort(key=lambda row: str((row[1] or {}).get("uid") or ""))
        return _slice_page(rows, page_clause)

    def export_relation_rows(self, owner_id: str, page_clause: str) -> list:
        owner = self.owner_literal(owner_id)
        by_rid = self._vertices_by_rid(owner)
        edges = []
        for label in VERTEX_TYPES:
            vertices = [rec for rec in by_rid.values() if rec.get("@type") == label]
            for record in vertices:
                for edge in self.query(
                    f"SELECT expand(outE()) FROM {label} WHERE @rid = {_rid(record.get('@rid'))}"
                ):
                    target = by_rid.get(str(edge.get("@in")))
                    if target is None:
                        continue
                    edges.append(
                        [
                            record.get("uid"),
                            edge.get("@type"),
                            target.get("uid"),
                            _plain(edge),
                        ]
                    )
        edges.sort(key=lambda row: (str(row[0]), str(row[2]), str(row[1])))
        return _slice_page(edges, page_clause)

    def import_node_batch(
        self, *, label: str, has_emb: bool, rows: list, owner_id: str
    ) -> Any:
        self.ensure_vector_indexes_if_missing(owner_id=owner_id)
        self._check_label(label)
        owner = self.owner_literal(owner_id)
        self.ensure_ready()
        if has_emb and rows and rows[0].get("emb"):
            self.ensure_vector_index(label, len(rows[0]["emb"]))
        with self.transaction() as session:
            for row in rows:
                props = dict(row.get("props") or {})
                props.pop("embedding", None)
                props["uid"] = row.get("uid")
                props["owner_id"] = owner
                if has_emb:
                    props["embedding"] = row.get("emb")
                now = _now_ms()
                if props.get("created_at") is None:
                    props["created_at"] = now
                if props.get("updated_at") is None:
                    props["updated_at"] = now
                existing = self.query(
                    f"SELECT @rid AS rid FROM {label} WHERE uid = :uid AND owner_id = '{owner}' LIMIT 1",
                    {"uid": row.get("uid")},
                    session=session,
                )
                if existing:
                    self._update(
                        label, str(row.get("uid")), owner, props, session=session
                    )
                else:
                    self._insert(label, props, session=session)
        return True

    def import_relation_batch(self, *, rel_type: str, rows: list, owner_id: str) -> Any:
        self.ensure_vector_indexes_if_missing(owner_id=owner_id)
        self.ensure_edge_type(rel_type)
        for row in rows:
            from_label, from_rec = self._find(str(row.get("from_id")), owner_id)
            to_label, to_rec = self._find(str(row.get("to_id")), owner_id)
            if from_rec is None or to_rec is None:
                continue
            del from_label, to_label
            from_rid = _rid(from_rec.get("@rid"))
            to_rid = _rid(to_rec.get("@rid"))
            existing = self.query(
                f"SELECT @rid AS rid FROM {rel_type} "
                f"WHERE @out = {from_rid} AND @in = {to_rid} LIMIT 1"
            )
            props = dict(row.get("props") or {})
            now = _now_ms()
            if props.get("created_at") is None:
                props["created_at"] = now
            if props.get("updated_at") is None:
                props["updated_at"] = now
            if existing:
                self._set_edge_props(rel_type, _rid(existing[0].get("rid")), props)
            else:
                self._insert_edge(rel_type, from_rid, to_rid, props)
        return True

    def graph_size_rows(self, owner_id: str) -> list:
        return self.label_counts(owner_id)

    def dedup_candidate_rows(
        self, *, label: str, owner_id: str, limit: int, touched_filter: str
    ) -> list:
        self._check_label(label)
        owner = self.owner_literal(owner_id)
        compare = None
        threshold = None
        if touched_filter and touched_filter.strip():
            match = _TOUCHED.search(touched_filter.strip())
            if not match:
                raise ValueError("Unsupported touched_filter")
            compare, threshold = match.group(1), int(match.group(2))
        rows = self.query(
            f"SELECT uid, text, created_at, updated_at, last_dedup_at, embedding FROM {label} "
            f"WHERE owner_id = '{owner}' AND embedding IS NOT NULL "
            "AND (status IS NULL OR status = 'active')"
        )
        picked = []
        for row in rows:
            touched = row.get("updated_at")
            if touched is None:
                touched = row.get("created_at")
            touched = int(touched or 0)
            last_dedup = row.get("last_dedup_at")
            if last_dedup is not None and int(last_dedup) >= touched:
                continue
            if compare == ">=" and touched < int(threshold):
                continue
            if compare == "<" and touched >= int(threshold):
                continue
            picked.append(
                [
                    row.get("uid"),
                    row.get("text"),
                    row.get("created_at"),
                    touched,
                    row.get("embedding"),
                ]
            )
        picked.sort(key=lambda row: (row[3], row[2] or 0, str(row[0])))
        return picked[: int(limit)]

    def dedup_similar_rows(
        self,
        *,
        label: str,
        embedding: list,
        max_distance: float,
        owner_id: str,
        node_id: str,
        ann_k: int,
        top_k: int,
    ) -> list:
        rows = self.ann_rows(
            node_type=label,
            embedding=embedding,
            owner_id=owner_id,
            ann_k=ann_k,
            limit=top_k,
            max_distance=max_distance,
            include_outdated=False,
            exclude_node_id=node_id,
        )
        # ann rows are uid, type, text, status, created_at, metadata, distance
        similar = []
        for row in rows:
            created = None
            _label, record = self._find(str(row[0]), owner_id, types=(label,))
            if record is not None:
                created = record.get("created_at")
            similar.append([row[0], created, float(row[6])])
        similar.sort(key=lambda row: (row[2], row[1] or 0, str(row[0])))
        return similar[: int(top_k)]

    def mark_deduped(self, *, label: str, node_ids: list, owner_id: str) -> Any:
        now = _now_ms()
        for node_id in node_ids:
            self._update(label, str(node_id), owner_id, {"last_dedup_at": now})
        return True

    def rels_out(self, *, label: str, dup_id: str, owner_id: str) -> list:
        return self._incident_edges(label, dup_id, owner_id, outgoing=True)

    def rels_in(self, *, label: str, dup_id: str, owner_id: str) -> list:
        return self._incident_edges(label, dup_id, owner_id, outgoing=False)

    def copy_rel_out(self, *, label: str, rel_type: str, params: dict) -> Any:
        del label
        return self._copy_edge(
            rel_type,
            str(params["primary_id"]),
            str(params["target_id"]),
            str(params.get("owner_id") or ""),
            params.get("props") or {},
            outgoing=True,
        )

    def copy_rel_in(self, *, label: str, rel_type: str, params: dict) -> Any:
        del label
        owner_id = str(
            params.get("owner_id") or self._owner_of(str(params["primary_id"])) or ""
        )
        return self._copy_edge(
            rel_type,
            str(params["source_id"]),
            str(params["primary_id"]),
            owner_id,
            params.get("props") or {},
            outgoing=True,
        )

    def delete_rels_out(self, *, label: str, dup_id: str, owner_id: str) -> Any:
        return self._delete_incident(label, dup_id, owner_id, outgoing=True)

    def delete_rels_in(self, *, label: str, dup_id: str, owner_id: str) -> Any:
        return self._delete_incident(label, dup_id, owner_id, outgoing=False)

    def mark_merged(
        self, *, label: str, dup_id: str, primary_id: str, owner_id: str
    ) -> Any:
        return self._update(
            label,
            dup_id,
            owner_id,
            {"status": "outdated", "merged_into": primary_id, "updated_at": _now_ms()},
        )

    def touch_dedup_primary(self, *, label: str, primary_id: str, owner_id: str) -> Any:
        return self._update(label, primary_id, owner_id, {"last_dedup_at": _now_ms()})

    def expired_fact_ids(self, owner_id: str, now_ms: int) -> list:
        owner = self.owner_literal(owner_id)
        rows = self.query(
            "SELECT uid FROM Fact WHERE owner_id = "
            f"'{owner}' AND (status IS NULL OR status = 'active') "
            "AND expires_at IS NOT NULL AND expires_at <= :now_ms",
            {"now_ms": int(now_ms)},
        )
        return [[row.get("uid")] for row in rows]

    def stale_fact_ids(self, owner_id: str, cutoff_ms: int) -> list:
        rows = self.brief_stale_facts(owner_id, cutoff_ms)
        return [[row[0]] for row in rows]

    def fact_neighbor_rows(self, fact_id: str, owner_id: str) -> list:
        """Neighbors of one fact in that owner's database."""
        with self.owner_scope(owner_id):
            rows = self.query(
                "SELECT @rid AS rid FROM Fact WHERE uid = :uid LIMIT 1",
                {"uid": fact_id},
            )
            if not rows:
                return []
            neighbors = self.query(
                "SELECT expand(both()) FROM Fact WHERE @rid = "
                f"{_rid(rows[0].get('rid'))}"
            )
        return [
            [[str(record.get("@type") or "")], record.get("status")]
            for record in neighbors
        ]

    def _merge_entity(
        self,
        owner: str,
        *,
        name_norm: str,
        text: str,
        uid: str,
        embedding: Any,
        params: dict,
        session: str,
    ) -> dict:
        found = self.query(
            "SELECT FROM Entity WHERE owner_id = "
            f"'{owner}' AND name_norm = :name_norm LIMIT 1",
            {"name_norm": name_norm},
            session=session,
        )
        promoted = {
            "project": params.get("project"),
            "created_by": params.get("created_by"),
            "updated_by": params.get("updated_by"),
            "tags": params.get("tags"),
            "meta_type": params.get("meta_type"),
            "confidence": params.get("confidence"),
        }
        if found:
            record = found[0]
            changes = {
                key: value
                for key, value in promoted.items()
                if value is not None and key != "created_by"
            }
            changes["updated_at"] = _now_ms()
            if params.get("updated_by") is not None:
                changes["updated_by"] = params.get("updated_by")
            if changes:
                self._update(
                    "Entity", str(record.get("uid")), owner, changes, session=session
                )
                refreshed = self.query(
                    "SELECT FROM Entity WHERE uid = :uid AND owner_id = "
                    f"'{owner}' LIMIT 1",
                    {"uid": record.get("uid")},
                    session=session,
                )
                return refreshed[0] if refreshed else record
            return record
        now = _now_ms()
        return self._insert(
            "Entity",
            {
                "uid": uid,
                "owner_id": owner,
                "name_norm": name_norm,
                "text": text,
                "created_at": now,
                "updated_at": now,
                "embedding": embedding,
                "status": "active",
                "metadata_str": params.get("metadata_str"),
                **promoted,
            },
            session=session,
        )

    def _find(
        self,
        uid: str,
        owner_id: str,
        *,
        types: tuple[str, ...] = VERTEX_TYPES,
    ) -> tuple[str | None, dict | None]:
        owner = self.owner_literal(owner_id)
        self.ensure_ready()
        for label in types:
            self._check_label(label)
            rows = self.query(
                f"SELECT FROM {label} WHERE uid = :uid AND owner_id = '{owner}' LIMIT 1",
                {"uid": uid},
            )
            if rows:
                return label, rows[0]
        return None, None

    def _find_rid(self, rid: str, owner_id: str) -> tuple[str, dict] | None:
        owner = self.owner_literal(owner_id)
        safe = _rid(rid)
        for label in VERTEX_TYPES:
            rows = self.query(
                f"SELECT FROM {label} WHERE @rid = {safe} AND owner_id = '{owner}' LIMIT 1"
            )
            if rows:
                return label, rows[0]
        return None

    def _insert(self, label: str, fields: dict, *, session: str | None = None) -> dict:
        columns, params = _bind_fields(fields)
        rows = self.command(
            f"INSERT INTO {label} SET {columns}",
            params,
            session=session,
        )
        if not rows:
            raise RuntimeError(f"ArcadeDB insert into {label} returned no record")
        return rows[0]

    def _update(
        self,
        label: str,
        uid: str,
        owner_id: str,
        fields: dict,
        *,
        session: str | None = None,
    ) -> Any:
        if not fields:
            return None
        owner = self.owner_literal(owner_id)
        columns, params = _bind_fields(fields)
        params["uid"] = uid
        return self.command(
            f"UPDATE {label} SET {columns} WHERE uid = :uid AND owner_id = '{owner}'",
            params,
            session=session,
        )

    def _insert_edge(
        self,
        rel_type: str,
        from_rid: str,
        to_rid: str,
        props: dict,
        *,
        session: str | None = None,
    ) -> dict:
        columns, params = _bind_fields(props)
        assignment = f" SET {columns}" if columns else ""
        rows = self.command(
            f"CREATE EDGE {rel_type} FROM {from_rid} TO {to_rid}{assignment}",
            params,
            session=session,
        )
        if not rows:
            raise RuntimeError(f"ArcadeDB did not create {rel_type}")
        return rows[0]

    def _set_edge_props(self, rel_type: str, rid: str, props: dict) -> None:
        columns, params = _bind_fields(props)
        if not columns:
            return
        self.command(
            f"UPDATE {rel_type} SET {columns} WHERE @rid = {rid}",
            params,
        )

    def _edge_type_between(
        self, label: str, from_rid: str, to_rid: str, directed: bool
    ) -> str | None:
        expander = "outE()" if directed else "bothE()"
        edges = self.query(
            f"SELECT expand({expander}) FROM {label} WHERE @rid = {from_rid}"
        )
        for edge in edges:
            other = (
                edge.get("@in")
                if directed or str(edge.get("@out")) == from_rid
                else edge.get("@out")
            )
            if not directed and str(edge.get("@out")) == from_rid:
                other = edge.get("@in")
            elif not directed:
                other = edge.get("@out")
            if str(other) == to_rid:
                return str(edge.get("@type") or "")
        return None

    def _incident_edges(
        self, label: str, node_id: str, owner_id: str, *, outgoing: bool
    ) -> list:
        self._check_label(label)
        _found, record = self._find(node_id, owner_id, types=(label,))
        if record is None:
            return []
        rid = _rid(record.get("@rid"))
        expander = "outE()" if outgoing else "inE()"
        rows = []
        for edge in self.query(
            f"SELECT expand({expander}) FROM {label} WHERE @rid = {rid}"
        ):
            other_rid = edge.get("@in") if outgoing else edge.get("@out")
            other = self._find_rid(str(other_rid), owner_id)
            if other is None:
                continue
            _other_label, other_rec = other
            endpoint = other_rec.get("uid")
            rows.append([edge.get("@type"), _plain(edge), endpoint])
        return rows

    def _delete_incident(
        self, label: str, node_id: str, owner_id: str, *, outgoing: bool
    ) -> Any:
        self._check_label(label)
        _found, record = self._find(node_id, owner_id, types=(label,))
        if record is None:
            return 0
        rid = _rid(record.get("@rid"))
        expander = "outE()" if outgoing else "inE()"
        edges = self.query(f"SELECT expand({expander}) FROM {label} WHERE @rid = {rid}")
        for edge in edges:
            edge_type = str(edge.get("@type") or "")
            if not re.fullmatch(r"[A-Z][A-Z0-9_]*", edge_type):
                continue
            self.command(
                f"DELETE FROM {edge_type} WHERE @rid = {_rid(edge.get('@rid'))}"
            )
        return len(edges)

    def _copy_edge(
        self,
        rel_type: str,
        from_id: str,
        to_id: str,
        owner_id: str,
        props: dict,
        *,
        outgoing: bool,
    ) -> Any:
        del outgoing
        if not owner_id:
            return None
        from_label, from_rec = self._find(from_id, owner_id)
        to_label, to_rec = self._find(to_id, owner_id)
        if from_rec is None or to_rec is None:
            return None
        del from_label, to_label
        self.ensure_edge_type(rel_type)
        from_rid = _rid(from_rec.get("@rid"))
        to_rid = _rid(to_rec.get("@rid"))
        existing = self.query(
            f"SELECT @rid AS rid FROM {rel_type} "
            f"WHERE @out = {from_rid} AND @in = {to_rid} LIMIT 1"
        )
        copied = dict(props or {})
        stamp = copied.get("updated_at")
        if stamp is None:
            stamp = _now_ms()
        if existing:
            # Keep the edge that is already on the survivor, including created_*.
            self._set_edge_props(
                rel_type, _rid(existing[0].get("rid")), {"updated_at": stamp}
            )
            return [[existing[0].get("rid")]]
        if copied.get("created_at") is None:
            copied["created_at"] = stamp
        copied["updated_at"] = stamp
        created = self._insert_edge(rel_type, from_rid, to_rid, copied)
        return [[created.get("@rid")]]

    def _owner_of(self, uid: str) -> str | None:
        self.ensure_ready()
        for label in VERTEX_TYPES:
            rows = self.query(
                f"SELECT owner_id FROM {label} WHERE uid = :uid LIMIT 1",
                {"uid": uid},
            )
            if rows and rows[0].get("owner_id"):
                return str(rows[0]["owner_id"])
        return None

    def _known_edge_types(self) -> list[str]:
        rows = self.query("SELECT name FROM schema:types")
        names = []
        for row in rows:
            name = str(row.get("name") or "")
            if name in VERTEX_TYPES or name in {"GmMeta", "V", "E"}:
                continue
            if re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
                names.append(name)
        # _edge_types is keyed by database name. The values are edge types.
        current = self._edge_types.get(self._require_db(), ())
        return sorted(set(names) | set(current))

    def _check_label(self, label: str) -> None:
        if label not in VERTEX_TYPES:
            raise ValueError(f"Unsupported node type: {label}")


def _bind_fields(fields: dict) -> tuple[str, dict]:
    parts = []
    params: dict[str, Any] = {}
    for key, value in fields.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(key)):
            raise ValueError(f"Invalid property name: {key!r}")
        param = f"p_{key}"
        parts.append(f"{key} = :{param}")
        params[param] = value
    return ", ".join(parts), params


_WHEN_RANK = re.compile(r"WHEN '([A-Z][A-Z0-9_]*)' THEN (\d+)")
_ELSE_RANK = re.compile(r"ELSE (\d+)")


def _edge_rank(rank_expr: str, edge_type: str) -> int:
    """Lower rank is kept first. Mirrors ``neighbor_rank_expr`` CASE text."""
    if not rank_expr:
        return 0
    for name, rank in _WHEN_RANK.findall(rank_expr):
        if name == edge_type:
            return int(rank)
    else_match = _ELSE_RANK.search(rank_expr)
    return int(else_match.group(1)) if else_match else 0


def _durable_active(record: dict) -> bool:
    """Active fact with no TTL. Matches Falkor ``expires_at IS NULL``."""
    status = record.get("status") or "active"
    return status == "active" and record.get("expires_at") is None


def _visible(
    record: dict,
    include_outdated: bool,
    now_ms: int,
    include_ttl: bool = False,
) -> bool:
    """Active durable nodes by default. ``include_ttl`` keeps unexpired TTL facts."""
    if include_outdated:
        return True
    status = record.get("status") or "active"
    if status != "active":
        return False
    expires = record.get("expires_at")
    if expires is None:
        return True
    return bool(include_ttl and int(expires) > now_ms)
