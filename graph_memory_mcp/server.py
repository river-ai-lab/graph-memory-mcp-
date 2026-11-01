"""
MCP Graph Memory server (FalkorDB Graph Database).

Layers:
- MCP layer (this file): MCP tools registration only (thin wrapper);
- handlers layer: high-level business logic in `graph_memory_mcp.graph_memory.mcp_handlers`;
- DB layer: `graph_memory_mcp.graph_memory.database.FalkorDBClient`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Literal

from graph_memory_mcp.base_server import BaseGraphMemoryMCP
from graph_memory_mcp.config import MCPServerConfig
from graph_memory_mcp.graph_memory import (
    mcp_handlers_admin,
    mcp_handlers_graph,
    mcp_handlers_ingest,
    mcp_handlers_nodes,
    mcp_handlers_relations,
    mcp_handlers_search,
)
from graph_memory_mcp.tool_annotations import (
    ADDITIVE,
    ADDITIVE_IDEMPOTENT,
    DESTRUCTIVE,
    READ_ONLY,
)

logger = logging.getLogger(__name__)


class GraphMemoryMCP(BaseGraphMemoryMCP):
    """Graph Memory MCP Server: builds the MCP server app and registers tools."""

    def __init__(self, server_config: MCPServerConfig):
        super().__init__(server_config)

        # Auto-create vector indices if enabled (opt-in)
        if self.server_config.auto_create_indexes and self._db_connected:
            logger.info(
                "AUTO-CREATE: Creating vector indexes (config.auto_create_indexes=true)"
            )
            try:
                self._ensure_indexes_if_needed()
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to auto-create vector indexes: %s", exc)

    def _ensure_indexes_if_needed(self) -> None:
        """Create vector indexes in every existing owner graph.

        New owner graphs get their indexes on the first write.
        Search and auto_link still create them when missing.
        """
        dim = int(getattr(self.embedding_service, "dimension", 0) or 0)
        owners = self.db_client.list_owners() or ["default"]
        for owner_id in owners:
            self.db_client.ensure_vector_indexes_if_missing(
                owner_id=owner_id, dimension=dim
            )

    def _register_tools(self) -> None:
        exposed: Dict[str, Any] = {}
        exposed.update(self._register_information_tools())
        exposed.update(self._register_fact_tools())
        exposed.update(self._register_bulk_and_transfer_tools())
        exposed.update(self._register_triplet_tools())
        exposed.update(self._register_graph_tools())

        for attr, func in exposed.items():
            setattr(self, attr, func)

    def _register_information_tools(self) -> Dict[str, Any]:
        exposed: Dict[str, Any] = {}
        db = self.db_client
        mcp = self.mcp
        assert mcp is not None
        default_owner = self._tool_owner_default()

        if self.config.mcp_expose_admin_tools:

            @mcp.tool(
                title="Test connection",
                annotations=READ_ONLY,
                description="Simple ping to verify MCP server availability.",
            )
            def test_connection() -> dict:
                return mcp_handlers_admin.test_connection(db)

            exposed["test_connection"] = test_connection

        @mcp.tool(
            title="Get stats",
            annotations=READ_ONLY,
            description="Get graph statistics (node counts, etc.).",
        )
        def get_stats(owner_id: str = default_owner) -> dict:
            return mcp_handlers_admin.get_stats(db, owner_id=owner_id)

        @mcp.tool(
            title="Health check",
            annotations=READ_ONLY,
            description=(
                "Check server health. healthy follows the graph backend and embeddings. "
                "vector_index_missing lists sampled owners that have no vector index yet; "
                "that does not clear healthy. The first write creates both indexes. "
                "Search creates them if they are still missing."
            ),
        )
        def health_check() -> dict:
            return mcp_handlers_admin.health_check(db, self.embedding_service)

        @mcp.tool(
            title="Get brief",
            annotations=READ_ONLY,
            description=(
                "Session warm-up: optional project anchor (source.ref=about), "
                "recent active facts without TTL (by time, not connectivity), "
                "open CONTRADICTS pairs, stale_count, and stats. "
                "stale_count is active facts not accessed for STALE_FACTS_DAYS. "
                "Do not pass include_stale at task start: it returns those texts "
                "and is a maintenance view, not project state. "
                "limit applies only to recent facts. "
                "This is where the project stands, not a search for the task. "
                "After it, search(..., compact=true) for the task topic. "
                "Keep the about fact current via upsert_node(source.ref='about')."
            ),
        )
        def get_brief(
            owner_id: str = default_owner,
            limit: int = 10,
            include_stale: bool = False,
        ) -> dict:
            return mcp_handlers_admin.get_brief(
                db,
                owner_id=owner_id,
                limit=limit,
                include_stale=include_stale,
            )

        if self.config.mcp_expose_admin_tools:

            @mcp.tool(
                title="Ensure vector indexes",
                annotations=ADDITIVE_IDEMPOTENT,
                description=(
                    "Create or verify Fact and Entity vector indexes for semantic search. "
                    "Idempotent — safe to call multiple times. "
                    "The first write for an owner creates them too, and search or "
                    "auto_link creates them when missing. Set "
                    "AUTO_CREATE_INDEXES=true to create them at server startup."
                ),
            )
            def ensure_vector_indexes() -> dict:
                return self.ensure_indexes_status()

            exposed["ensure_vector_indexes"] = ensure_vector_indexes

        exposed["get_stats"] = get_stats
        exposed["health_check"] = health_check
        exposed["get_brief"] = get_brief
        return exposed

    def ensure_indexes_status(self) -> dict:
        """Create missing indexes and report per-owner status (MCP and /admin)."""
        try:
            self._ensure_indexes_if_needed()
            owners = self.db_client.list_owners() or ["default"]
            return {
                "success": True,
                "indexes": {
                    owner: self.db_client.get_vector_index_status(owner_id=owner)
                    for owner in owners
                },
                "dimension": getattr(self.embedding_service, "dimension", 0),
            }
        except Exception as e:  # noqa: BLE001
            return {"success": False, "error": str(e)}

    def _register_fact_tools(self) -> Dict[str, Any]:
        """Node tools: provenance-specific writes + shared read/lifecycle tools."""
        exposed: Dict[str, Any] = self._register_node_write_tools()
        exposed.update(self._register_node_shared_tools())
        return exposed

    def _register_node_write_tools(self) -> Dict[str, Any]:
        """create/upsert/update with nested `source` (overridden by simple profile)."""
        exposed: Dict[str, Any] = {}
        db = self.db_client
        config = self.config
        mcp = self.mcp
        assert mcp is not None
        default_owner = self._tool_owner_default()

        @mcp.tool(
            title="Create node",
            annotations=ADDITIVE,
            description=(
                "Create a node (Fact or Entity). "
                "Required: `text`, "
                "Optional: `node_type` ('Fact' default, or 'Entity'), `owner_id`, "
                "`metadata`, `source` (provenance dict with keys like ref/type/uri/content_hash/source_updated_at/version; "
                "`source_updated_at` is the source document time, not the server's `updated_at`), "
                "`auto_link` (Facts only), `ttl_days` (Facts only), "
                "`links` (create relations immediately after creation). "
                "Note: auto_link defaults to false. Pass true on a Fact to add MENTIONS edges to similar Entity nodes "
                "(vector search on Entity index; threshold AUTO_LINKING_SEMANTIC_THRESHOLD). "
                "Use create_relation for links between arbitrary node pairs. "
                "Set ttl_days for a temporary fact (the server never assigns it). "
                "After expires_at the archive job sets status=archived. "
                "`check_duplicates` (default true) blocks a near-paraphrase before write. "
                "Pass false when the nearby fact is similar by distance but is a different fact. "
                "A very long text is warned in `warnings` and still stored. "
                f"{self._owner_tool_help()} {self._attribution_tool_help()}"
            ),
        )
        def create_node(
            text: str,
            node_type: Literal["Fact", "Entity"] = "Fact",
            owner_id: str = default_owner,
            metadata: dict | None = None,
            source: dict | None = None,
            status: Literal["active", "outdated", "archived"] | None = None,
            ttl_days: float | None = None,
            entity_type: str | None = None,
            auto_link: bool = False,
            semantic_threshold: float | None = None,
            links: list[dict] | None = None,
            check_duplicates: bool = True,
        ) -> dict:
            return mcp_handlers_nodes.create_node(
                db,
                config,
                text=text,
                node_type=node_type,
                owner_id=owner_id,
                metadata=metadata,
                source=source,
                status=status,
                ttl_days=ttl_days,
                entity_type=entity_type,
                auto_link=auto_link,
                semantic_threshold=semantic_threshold,
                links=links,
                check_duplicates=check_duplicates,
            )

        @mcp.tool(
            title="Upsert node",
            annotations=DESTRUCTIVE,
            description=(
                "Create or update a node using `source.ref` as a stable sync key. "
                "Required: `text`, `source.ref`. "
                "Optional: same fields as create_node, plus `versioning=true` to store "
                "a version snapshot and auto-increment `source.version` when it is omitted. "
                f"{self._owner_tool_help()} {self._attribution_tool_help()}"
            ),
        )
        def upsert_node(
            text: str,
            source: dict,
            node_type: Literal["Fact", "Entity"] = "Fact",
            owner_id: str = default_owner,
            metadata: dict | None = None,
            status: Literal["active", "outdated", "archived"] | None = None,
            ttl_days: float | None = None,
            versioning: bool | None = None,
            entity_type: str | None = None,
            auto_link: bool = False,
            semantic_threshold: float | None = None,
            links: list[dict] | None = None,
            description: str | None = None,
        ) -> dict:
            return mcp_handlers_nodes.upsert_node(
                db,
                config,
                text=text,
                description=description,
                node_type=node_type,
                owner_id=owner_id,
                metadata=metadata,
                source=source,
                status=status,
                ttl_days=ttl_days,
                versioning=versioning,
                entity_type=entity_type,
                auto_link=auto_link,
                semantic_threshold=semantic_threshold,
                links=links,
            )

        @mcp.tool(
            title="Update node",
            annotations=DESTRUCTIVE,
            description=(
                "Update a node (Fact or Entity). "
                "You can update text, metadata, source, status, or ttl_days. "
                "Set versioning=true to store a previous snapshot and auto-increment source.version when it is omitted. "
                f"{self._owner_tool_help()} {self._attribution_tool_help()}"
            ),
        )
        def update_node(
            node_id: str,
            owner_id: str = default_owner,
            text: str | None = None,
            metadata: dict | None = None,
            source: dict | None = None,
            status: Literal["active", "outdated", "archived"] | None = None,
            ttl_days: float | None = None,
            entity_type: str | None = None,
            versioning: bool | None = None,
        ) -> dict:
            return mcp_handlers_nodes.update_node(
                db,
                node_id=node_id,
                owner_id=owner_id,
                text=text,
                metadata=metadata,
                source=source,
                status=status,
                ttl_days=ttl_days,
                entity_type=entity_type,
                versioning=versioning,
            )

        exposed["create_node"] = create_node
        exposed["upsert_node"] = upsert_node
        exposed["update_node"] = update_node
        return exposed

    def _register_bulk_and_transfer_tools(self) -> Dict[str, Any]:
        """Bulk ingest and owner export/import (shared by both profiles)."""
        exposed: Dict[str, Any] = {}
        db = self.db_client
        config = self.config
        mcp = self.mcp
        assert mcp is not None
        default_owner = self._tool_owner_default()

        @mcp.tool(
            title="Ingest knowledge",
            annotations=ADDITIVE,
            description=(
                "Persist knowledge you extracted from one document/conversation in a "
                "single call: creates a source node (document.ref required), facts "
                "with provenance (source.ref = 'doc#{fact.ref|hash(text)}'), "
                "EXTRACTED_FROM links to the source, and optional triplets. "
                "Prefer facts[].ref for stable ids across edits/reorders; omit ref "
                "to key by text hash. Idempotent on fact ref. Extract first, then "
                "call once. Check possible_duplicates in the response. "
                f"{self._owner_tool_help()} {self._attribution_tool_help()}"
            ),
        )
        def ingest_knowledge(
            document: dict,
            facts: list[dict] | None = None,
            triplets: list[dict] | None = None,
            owner_id: str = default_owner,
            auto_link: bool = False,
        ) -> dict:
            return mcp_handlers_ingest.ingest_knowledge(
                db,
                config,
                document=document,
                facts=facts,
                triplets=triplets,
                owner_id=owner_id,
                auto_link=auto_link,
            )

        exposed["ingest_knowledge"] = ingest_knowledge

        @mcp.tool(
            title="Create nodes (bulk)",
            annotations=ADDITIVE,
            description=(
                "Bulk create up to 200 nodes of one type in a single call with "
                "batched embeddings. Each item: {text, description?, metadata?, "
                "status?, ttl_days?}. No auto_link/links/possible_duplicates — "
                "link explicitly afterwards. "
                f"{self._owner_tool_help()} {self._attribution_tool_help()}"
            ),
        )
        def create_nodes(
            items: list[dict],
            node_type: Literal["Fact", "Entity"] = "Fact",
            owner_id: str = default_owner,
        ) -> dict:
            return mcp_handlers_nodes.create_nodes(
                db,
                config,
                items=items,
                node_type=node_type,
                owner_id=owner_id,
            )

        if self.config.mcp_expose_admin_tools:

            @mcp.tool(
                title="Export owner",
                annotations=READ_ONLY,
                description=(
                    "Export all nodes and relations of an owner as JSON "
                    "(one object per node/relation; write items as lines for JSONL). "
                    "include_embeddings=true makes re-import lossless and cheap."
                ),
            )
            def export_owner(
                owner_id: str = default_owner,
                include_embeddings: bool = True,
                include_versions: bool = False,
                offset: int = 0,
                limit: int | None = None,
                section: str = "all",
            ) -> dict:
                return mcp_handlers_admin.export_owner(
                    db,
                    owner_id=owner_id,
                    include_embeddings=include_embeddings,
                    include_versions=include_versions,
                    offset=offset,
                    limit=limit,
                    section=section,
                )

            @mcp.tool(
                title="Import owner",
                annotations=DESTRUCTIVE,
                description=(
                    "Import an export_owner payload into an owner scope. "
                    "Nodes merge by uid (idempotent). Embeddings are taken from the "
                    "payload or recomputed when missing/regenerate_embeddings=true. "
                    "created_*/updated_* in the payload are ignored; "
                    "/admin/import keeps them."
                ),
            )
            def import_owner(
                owner_id: str,
                nodes: list[dict],
                relations: list[dict] | None = None,
                regenerate_embeddings: bool = False,
            ) -> dict:
                return mcp_handlers_admin.import_owner(
                    db,
                    owner_id=owner_id,
                    nodes=nodes,
                    relations=relations,
                    regenerate_embeddings=regenerate_embeddings,
                )

            exposed["export_owner"] = export_owner
            exposed["import_owner"] = import_owner

        exposed["create_nodes"] = create_nodes
        return exposed

    def _register_node_shared_tools(self) -> Dict[str, Any]:
        """Read/lifecycle node tools shared by both server profiles."""
        exposed: Dict[str, Any] = {}
        db = self.db_client
        config = self.config
        mcp = self.mcp
        assert mcp is not None
        default_owner = self._tool_owner_default()

        @mcp.tool(
            title="Search",
            annotations=READ_ONLY,
            description=(
                "Semantic search using embedding similarity (cosine distance). "
                "Returns active durable nodes by default. "
                "Facts with ttl_days are omitted until include_ttl=true "
                "(still-living temporary facts only). "
                "Use include_outdated=true to include outdated and archived nodes. "
                "Results ranked by a hybrid score: cosine, plus a boost when the "
                "query's identifiers (job/RUN id, symbol, S3 path, size) occur in the text, "
                "plus a smaller recency boost. Raw cosine stays in `similarity`. "
                "If two hits already have a CONTRADICTS edge, they are listed in "
                "`contradiction_pairs` and suggested_next points at get_trace / mark_outdated. "
                "Supports multi-tenant isolation via owner_id. "
                "search_type: pre_filter (filter owner first, best for large/multi-tenant graphs) "
                "or post_filter (global ANN then filter, best for small graphs); "
                "defaults to server SEARCH_TYPE when omitted. "
                "Always includes suggested_next / do_not guidance. "
                "Set compact=true for snippets under a token budget (expand via get_node / get_context). "
                "search_mode: embedding (default, as above), exact (phrase/substring; "
                "quote phrases to require each), fuzzy (typo-tolerant keywords), "
                "bm25 (keyword ranking), or hybrid (RRF of hybrid_modes, default "
                "embedding+bm25). rerank=true reorders top hits with a local cross-encoder."
            ),
        )
        def search(
            query: str,
            owner_id: str = default_owner,
            limit: int | None = None,
            node_types: list[str] | None = None,
            status: str | None = None,
            similarity_threshold: float | None = None,
            include_outdated: bool = False,
            include_ttl: bool = False,
            search_type: str | None = None,
            metadata_filter: dict | None = None,
            compact: bool = False,
            search_mode: str | None = None,
            hybrid_modes: list[str] | None = None,
            rerank: bool | None = None,
        ) -> dict:
            return mcp_handlers_search.search(
                db,
                config,
                query=query,
                owner_id=owner_id,
                limit=limit,
                node_types=node_types,
                status=status,
                similarity_threshold=similarity_threshold,
                include_outdated=include_outdated,
                include_ttl=include_ttl,
                search_type=search_type,
                metadata_filter=metadata_filter,
                compact=compact,
                search_mode=search_mode,
                hybrid_modes=hybrid_modes,
                rerank=rerank,
            )

        @mcp.tool(
            title="Get node",
            annotations=READ_ONLY,
            description=(
                "Retrieve a single node (Fact or Entity) by its ID. "
                "Optional as_of (unix ms): read the state at that time via version "
                "snapshots (requires updates made with versioning=true)."
            ),
        )
        def get_node(
            node_id: str, owner_id: str = default_owner, as_of: int | None = None
        ) -> dict:
            return mcp_handlers_nodes.get_node(
                db, node_id=node_id, owner_id=owner_id, as_of=as_of
            )

        @mcp.tool(
            title="Delete node",
            annotations=DESTRUCTIVE,
            description=(
                "PERMANENT: Irreversibly delete a node (Fact or Entity) and all its relations. "
                "This operation cannot be undone. "
                "For reversible removal of Facts, use mark_outdated instead."
            ),
        )
        def delete_node(node_id: str, owner_id: str = default_owner) -> dict:
            return mcp_handlers_nodes.delete_node(
                db, node_id=node_id, owner_id=owner_id
            )

        @mcp.tool(
            title="Mark fact as outdated",
            annotations=DESTRUCTIVE,
            description=(
                "Soft-delete a Fact by setting status='outdated'. "
                "Fact remains in graph but excluded from default searches. "
                "Optionally stores reason in metadata. "
                "Note: Only Facts support soft-delete. For Entities, use delete_node. "
                f"{self._owner_tool_help()} {self._attribution_tool_help(source=None)}"
            ),
        )
        def mark_outdated(
            fact_id: str,
            reason: str | None = None,
            owner_id: str = default_owner,
        ) -> dict:
            return mcp_handlers_nodes.mark_outdated(
                db,
                fact_id=fact_id,
                reason=reason,
                owner_id=owner_id,
            )

        @mcp.tool(
            title="Get node change history",
            annotations=READ_ONLY,
            description=(
                "Retrieve version history for a node. "
                "Returns a list of previous versions with timestamps. "
                "Currently only supported for Fact nodes."
            ),
        )
        def get_node_change_history(
            node_id: str, owner_id: str = default_owner
        ) -> dict:
            return mcp_handlers_nodes.get_node_change_history(
                db, node_id=node_id, owner_id=owner_id
            )

        exposed["search"] = search
        exposed["get_node"] = get_node
        exposed["delete_node"] = delete_node
        exposed["mark_outdated"] = mark_outdated
        exposed["get_node_change_history"] = get_node_change_history

        return exposed

    def _register_triplet_tools(self) -> Dict[str, Any]:
        exposed: Dict[str, Any] = {}
        db = self.db_client
        config = self.config
        mcp = self.mcp
        assert mcp is not None
        default_owner = self._tool_owner_default()

        @mcp.tool(
            title="Create triplet",
            annotations=ADDITIVE_IDEMPOTENT,
            description=(
                "Create a subject-predicate-object triplet. "
                "Entities are created if they don't exist. "
                "Optional: link to a source fact_id. "
                f"{self._owner_tool_help()} {self._attribution_tool_help()}"
            ),
        )
        def create_triplet(
            subject: str,
            predicate: str,
            object_value: str,
            metadata: dict | None = None,
            fact_id: str | None = None,
            owner_id: str = default_owner,
        ) -> dict:
            return mcp_handlers_relations.create_triplet(
                db,
                subject=subject,
                predicate=predicate,
                object_value=object_value,
                metadata=metadata,
                fact_id=fact_id,
                owner_id=owner_id,
                config=config,
            )

        @mcp.tool(
            title="Search triplets",
            annotations=READ_ONLY,
            description=(
                "Search for triplets matching a pattern. "
                "All parameters are optional; omit to match any value."
            ),
        )
        def search_triplets(
            subject: str | None = None,
            predicate: str | None = None,
            object_value: str | None = None,
            owner_id: str = default_owner,
            limit: int = 10,
        ) -> dict:
            return mcp_handlers_relations.search_triplets(
                db,
                subject=subject,
                predicate=predicate,
                object_value=object_value,
                owner_id=owner_id,
                limit=limit,
            )

        exposed["create_triplet"] = create_triplet
        exposed["search_triplets"] = search_triplets
        return exposed

    def _register_graph_tools(self) -> Dict[str, Any]:
        exposed: Dict[str, Any] = {}
        db = self.db_client
        config = self.config
        mcp = self.mcp
        assert mcp is not None
        default_owner = self._tool_owner_default()

        @mcp.tool(
            title="Create relation",
            annotations=ADDITIVE_IDEMPOTENT,
            description=(
                "Create a direct relation between two nodes. "
                "Both nodes must exist. "
                "Uses MERGE — same from_id, to_id, and relation_type will not create duplicates. "
                "Default: RELATED_TO for general links, MENTIONS when one node refers to another. "
                "Use SUMMARIZES, FOLLOWS_FROM, CONTRADICTS only when semantics matter. "
                "Types must be in server RELATION_ALLOWED_TYPES (see memory policies). "
                f"{self._owner_tool_help()} {self._attribution_tool_help(source='properties')}"
            ),
        )
        def create_relation(
            from_id: str,
            to_id: str,
            relation_type: str,
            properties: dict | None = None,
            owner_id: str = default_owner,
        ) -> dict:
            return mcp_handlers_relations.create_relation(
                db,
                from_id=from_id,
                to_id=to_id,
                relation_type=relation_type,
                properties=properties,
                owner_id=owner_id,
                config=config,
            )

        @mcp.tool(
            title="Get trace",
            annotations=READ_ONLY,
            description=(
                "Find shortest path between two nodes. "
                "Returns the path as a sequence of nodes and relations."
            ),
        )
        def get_trace(
            from_id: str,
            to_id: str,
            owner_id: str = default_owner,
            max_depth: int = 5,
            directed: bool = True,
        ) -> dict:
            return mcp_handlers_graph.get_trace(
                db,
                from_id=from_id,
                to_id=to_id,
                owner_id=owner_id,
                max_depth=max_depth,
                directed=directed,
            )

        @mcp.tool(
            title="Delete relation",
            annotations=DESTRUCTIVE,
            description=(
                "Remove relations between two nodes. "
                "Optionally specify relation_type to remove only specific relations. "
                "Works for any node types (Fact, Entity)."
            ),
        )
        def delete_relation(
            from_id: str,
            to_id: str,
            relation_type: str | None = None,
            owner_id: str = default_owner,
        ) -> dict:
            return mcp_handlers_relations.unlink_facts(
                db,
                from_id=from_id,
                to_id=to_id,
                relation_type=relation_type,
                owner_id=owner_id,
            )

        @mcp.tool(
            title="Get context",
            annotations=READ_ONLY,
            description=(
                "Get subgraph context around a node. "
                "Returns nodes and edges within specified depth. "
                "Pass offset (with max_nodes as page size) for paginated neighbor loading. "
                "page=true reads stable id-ordered pages from that offset, including the first page. "
                "offset=0 without page is the priority walk. "
                "Useful for building agent context from related facts and entities. "
                "Depth defaults to config (SUBGRAPH_DEFAULT_DEPTH, default 1). "
                "Neighbors that have ttl_days are omitted unless include_ttl=true. "
                "The node budget is filled with CONTRADICTS, then FOLLOWS_FROM, then other allowlist edges; MENTIONS last. "
                "relation_priority overrides that order. exclude_relation_types drops those edges."
            ),
        )
        def get_context(
            node_id: str,
            owner_id: str = default_owner,
            depth: int | None = None,
            max_nodes: int | None = None,
            offset: int = 0,
            page: bool = False,
            include_outdated: bool = False,
            include_ttl: bool = False,
            relation_priority: list[str] | None = None,
            exclude_relation_types: list[str] | None = None,
        ) -> dict:
            return mcp_handlers_graph.get_context(
                db,
                config,
                node_id=node_id,
                owner_id=owner_id,
                depth=depth,
                max_nodes=max_nodes,
                offset=offset,
                page=page,
                include_outdated=include_outdated,
                include_ttl=include_ttl,
                relation_priority=relation_priority,
                exclude_relation_types=exclude_relation_types,
            )

        @mcp.tool(
            title="Recall context",
            annotations=READ_ONLY,
            description=(
                "Optional shortcut: semantic search + graph expansion in one call. "
                "Default recall workflow is search → get_context → get_trace (see memory policies). "
                "Use on small/sparse owner graphs; prefer depth=1. "
                "Set include_paths=true only when you need an approximate path hint "
                "between the top two seeds. "
                "Always includes suggested_next / do_not. "
                "Set compact=true for snippets under a token budget. "
                "Facts with ttl_days are omitted unless include_ttl=true. "
                "Expansion fills the node budget with CONTRADICTS and FOLLOWS_FROM before MENTIONS."
            ),
        )
        def recall_context(
            query: str,
            owner_id: str = default_owner,
            depth: int | None = None,
            limit: int | None = None,
            max_nodes: int | None = None,
            similarity_threshold: float | None = None,
            include_outdated: bool = False,
            include_ttl: bool = False,
            search_type: str | None = None,
            include_paths: bool = False,
            metadata_filter: dict | None = None,
            compact: bool = False,
            relation_priority: list[str] | None = None,
            exclude_relation_types: list[str] | None = None,
        ) -> dict:
            return mcp_handlers_graph.recall_context(
                db,
                config,
                query=query,
                owner_id=owner_id,
                depth=depth,
                limit=limit,
                max_nodes=max_nodes,
                similarity_threshold=similarity_threshold,
                include_outdated=include_outdated,
                include_ttl=include_ttl,
                search_type=search_type,
                include_paths=include_paths,
                metadata_filter=metadata_filter,
                compact=compact,
                relation_priority=relation_priority,
                exclude_relation_types=exclude_relation_types,
            )

        @mcp.tool(
            title="Find similar",
            annotations=READ_ONLY,
            description=(
                "Find facts similar to a given fact using embedding similarity. "
                "Returns facts ranked by semantic similarity (excludes the query fact itself). "
                "Useful for discovering related knowledge or identifying potential duplicates."
            ),
        )
        def find_similar(
            fact_id: str,
            owner_id: str = default_owner,
            limit: int = 5,
            similarity_threshold: float | None = None,
        ) -> dict:
            return mcp_handlers_search.find_similar(
                db,
                config,
                fact_id=fact_id,
                owner_id=owner_id,
                limit=limit,
                similarity_threshold=similarity_threshold,
            )

        @mcp.tool(
            title="Create summary fact",
            annotations=ADDITIVE,
            description=(
                "Create a summary fact from multiple source facts. "
                "Links the summary to all source facts."
            ),
        )
        def create_summary_fact(
            fact_ids: list[str],
            summary_text: str,
            owner_id: str = default_owner,
            metadata: dict | None = None,
        ) -> dict:
            return mcp_handlers_admin.create_summary_fact(
                db,
                config,
                fact_ids=fact_ids,
                summary_text=summary_text,
                owner_id=owner_id,
                metadata=metadata,
            )

        exposed["create_relation"] = create_relation
        exposed["delete_relation"] = delete_relation
        exposed["get_trace"] = get_trace
        exposed["get_context"] = get_context
        exposed["recall_context"] = recall_context
        exposed["find_similar"] = find_similar
        exposed["create_summary_fact"] = create_summary_fact

        return exposed
