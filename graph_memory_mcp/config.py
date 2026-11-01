from __future__ import annotations

from typing import Any, Dict, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MCPServerConfig(BaseSettings):
    """
    Memory MCP server configuration (single source of truth).
    """

    default_owner_id: str = Field(
        default="default", validation_alias="DEFAULT_OWNER_ID"
    )
    # client: tool argument owner_id (trusted local / company).
    # header: ignore the tool argument; owner is X-Owner-Id from a gateway
    # in front of 127.0.0.1. The gateway sends ADMIN_TOKEN when it is set.
    # token: Bearer API key. owner_id must be one of the key's projects;
    # omit it to use the first project. Author is the key's name.
    owner_scope: Literal["client", "header", "token"] = Field(
        default="client", validation_alias="OWNER_SCOPE"
    )
    # Hash file for OWNER_SCOPE=token. Relative paths use the process cwd.
    api_keys_file: str = Field(default="keys.json", validation_alias="API_KEYS_FILE")

    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_ignore_empty=False,
        # Allow constructing with field names (not only env aliases), e.g. in tests.
        populate_by_name=True,
        # If `.env` exists, it will be loaded; if not, defaults + process env vars are used.
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # MCP server (top-level)
    enabled: bool = Field(default=True, validation_alias="MCP_SERVER_ENABLED")
    # Admin/agent separation: admin operations live on HTTP /admin/* routes.
    # Set true to additionally expose them as MCP tools (legacy behavior).
    mcp_expose_admin_tools: bool = Field(
        default=False, validation_alias="MCP_EXPOSE_ADMIN_TOOLS"
    )
    admin_token: str = Field(default="", validation_alias="ADMIN_TOKEN")
    # Header mode only. True logs a warning and allows a non-loopback bind
    # when ADMIN_TOKEN is empty. client and token ignore it.
    allow_insecure_header: bool = Field(
        default=False, validation_alias="ALLOW_INSECURE_HEADER"
    )
    # Client mode only. True logs a warning and allows a non-loopback bind.
    # /mcp has no authentication in client mode.
    allow_insecure_client: bool = Field(
        default=False, validation_alias="ALLOW_INSECURE_CLIENT"
    )
    name: str = Field(default="memory", validation_alias="MCP_SERVER_NAME")
    description: str = Field(
        default="Memory MCP server for knowledge graph (FalkorDB)",
        validation_alias="MCP_SERVER_DESCRIPTION",
    )

    # Graph backend. arcadedb is the default. falkordb is the other implemented store.
    graph_backend: str = Field(default="arcadedb", validation_alias="GRAPH_BACKEND")

    # FalkorDB
    falkordb_host: str = "localhost"
    falkordb_port: int = 6379
    falkordb_graph: str = "memory"
    falkordb_password: str = ""

    # ArcadeDB server. Each owner is a database named {ARCADE_DATABASE}_{owner_id}.
    arcade_host: str = Field(default="localhost", validation_alias="ARCADE_HOST")
    arcade_port: int = Field(default=2480, validation_alias="ARCADE_PORT")
    arcade_user: str = Field(default="root", validation_alias="ARCADE_USER")
    arcade_password: str = Field(default="", validation_alias="ARCADE_PASSWORD")
    arcade_database: str = Field(default="memory", validation_alias="ARCADE_DATABASE")

    # Embeddings. onnx is ONNX Runtime with the repo's fp32 onnx/model.onnx.
    # torch is sentence-transformers and needs the optional torch extra.
    embedding_backend: Literal["onnx", "torch"] = Field(
        default="onnx",
        validation_alias="EMBEDDING_BACKEND",
        description="onnx (default) or torch (sentence-transformers extra)",
    )
    embedding_model: str = "intfloat/multilingual-e5-base"
    # Model prompt prefixes (e5-family: "query: " / "passage: "). Empty = none.
    # Changing prefixes (like changing the model) requires re-embedding the corpus.
    embedding_query_prefix: str = Field(
        default="", validation_alias="EMBEDDING_QUERY_PREFIX"
    )
    embedding_passage_prefix: str = Field(
        default="", validation_alias="EMBEDDING_PASSAGE_PREFIX"
    )

    # Vector indexes (auto-creation)
    auto_create_indexes: bool = False  # opt-in: create indexes on startup

    # Search/graph defaults
    default_search_limit: int = 10
    max_search_limit: int = 100  # hard cap for limit on search-like tools
    semantic_similarity_threshold: float = 0.55
    default_search_type: str = Field(
        default="pre_filter",
        validation_alias="SEARCH_TYPE",
        description="Default search_type when MCP search omits it: pre_filter | post_filter",
    )
    post_filter_ann_k_min: int = Field(
        default=100,
        validation_alias="POST_FILTER_ANN_K_MIN",
        description="Minimum global ANN candidates for post_filter search (queryNodes k)",
    )
    post_filter_ann_k_max: int = Field(
        default=2000,
        validation_alias="POST_FILTER_ANN_K_MAX",
        description="Maximum global ANN candidates for post_filter search (queryNodes k)",
    )

    # Graph/auto-linking
    auto_linking_semantic_threshold: float = 0.75
    neighbours_search_threshold: float = 0.8
    subgraph_default_depth: int = 1
    subgraph_max_depth: int = 3
    subgraph_default_max_nodes: int = 20
    subgraph_max_nodes_limit: int = 50
    recall_context_default_depth: int = Field(
        default=2,
        validation_alias="RECALL_CONTEXT_DEFAULT_DEPTH",
    )
    recall_context_default_seed_limit: int = Field(
        default=5,
        validation_alias="RECALL_CONTEXT_SEED_LIMIT",
    )
    recall_context_hop_decay: float = Field(
        default=0.7,
        validation_alias="RECALL_CONTEXT_HOP_DECAY",
    )
    # Compact recall: snippet + estimated-token budget for search/recall_context
    compact_recall_token_budget: int = Field(
        default=800,
        validation_alias="COMPACT_RECALL_TOKEN_BUDGET",
        description="Max estimated tokens for texts when compact=true",
    )
    compact_snippet_chars: int = Field(
        default=180,
        validation_alias="COMPACT_SNIPPET_CHARS",
        description="Max characters per snippet when compact=true",
    )
    about_source_ref: str = Field(
        default="about",
        validation_alias="ABOUT_SOURCE_REF",
        description="source.ref of the optional project-anchor fact returned by get_brief",
    )
    search_lexical_weight: float = Field(
        default=0.35,
        validation_alias="SEARCH_LEXICAL_WEIGHT",
        description="Hybrid rank weight for identifier matches; 0 disables",
    )
    search_recency_weight: float = Field(
        default=0.10,
        validation_alias="SEARCH_RECENCY_WEIGHT",
        description="Hybrid rank weight for created_at recency; 0 disables",
    )
    search_recency_half_life_days: float = Field(
        default=14.0,
        validation_alias="SEARCH_RECENCY_HALF_LIFE_DAYS",
    )
    default_search_mode: str = Field(
        default="embedding",
        validation_alias="SEARCH_MODE",
        description="Default search_mode: embedding | exact | fuzzy | bm25 | hybrid",
    )
    search_reranker_enabled: bool = Field(
        default=False,
        validation_alias="SEARCH_RERANKER",
        description="Rerank search results with a local ONNX cross-encoder by default",
    )
    search_reranker_model: str = Field(
        default="Xenova/ms-marco-MiniLM-L-6-v2",
        validation_alias="SEARCH_RERANKER_MODEL",
    )
    search_rerank_top_k: int = Field(
        default=30,
        validation_alias="SEARCH_RERANK_TOP_K",
        description="How many top results the cross-encoder reorders",
    )
    # Time-aware recall: score ×= (1-w) + w * 0.5^(age_days/half_life) and
    # ×= (1-w) + w * usage. Weights 0 disable the corresponding factor.
    recall_recency_weight: float = Field(
        default=0.2, validation_alias="RECALL_RECENCY_WEIGHT"
    )
    recall_recency_half_life_days: float = Field(
        default=30.0, validation_alias="RECALL_RECENCY_HALF_LIFE_DAYS"
    )
    recall_usage_weight: float = Field(
        default=0.1, validation_alias="RECALL_USAGE_WEIGHT"
    )
    duplicate_similarity_threshold: float = 0.85
    create_duplicate_block_threshold: float = Field(
        default=0.92,
        validation_alias="CREATE_DUPLICATE_BLOCK_THRESHOLD",
        description="Block create_node when a nearer active fact is at least this similar",
    )
    long_text_warn_chars: int = Field(
        default=600,
        validation_alias="LONG_TEXT_WARN_CHARS",
        description="create_node warns when text is longer; it still writes",
    )
    duplicate_max_group_size: int = 10
    duplicate_top_k: int = 100
    summary_similarity_threshold: float = 0.7

    # Relation policy (create_relation, create_node links, create_triplet predicates)
    relation_policy_enforce: str = Field(
        default="warn",
        validation_alias="RELATION_POLICY_ENFORCE",
        description="off | warn | enforce — allowlist for relation types",
    )
    relation_allowed_types: str = Field(
        default=(
            "RELATED_TO,MENTIONS,SUMMARIZES,FOLLOWS_FROM,CONTRADICTS,"
            "EXTRACTED_FROM,SIMILAR_TO,NEXT_STEP,EXPOSES,CROSS_REF,"
            "RUNS_ON,USES,SERVED_BY"
        ),
        validation_alias="RELATION_ALLOWED_TYPES",
    )

    # Housekeeping
    cleanup_days_threshold: int = 90
    stale_facts_days: int = 30  # get_brief: facts not recalled for N days
    log_ttl_days: int = 10
    log_cleanup_frequency: int = 10

    # Background jobs (APScheduler). JOBS_ENABLED is enough; both jobs then run.
    # JOBS_OWNER_IDS limits them to those tenants. Empty means every owner.
    jobs_enabled: bool = False
    jobs_owner_ids: str = ""
    jobs_process_all_owners: bool = (
        False  # force every owner, even if JOBS_OWNER_IDS is set
    )
    jobs_lock_ttl_seconds: int = 600

    # Job: deduplicate_facts (+ entities in the same job). On when jobs are on.
    job_deduplicate_enabled: bool = True
    job_deduplicate_cron: str = "0 * * * *"
    job_deduplicate_hours_threshold: int = 24
    job_deduplicate_similarity_threshold: float = 0.95

    # Job: archive_old_facts. On when jobs are on.
    job_archive_enabled: bool = True
    job_archive_cron: str = "0 3 * * 0"
    # Also archive active facts not recalled for `stale_facts_days` (opt-in)
    job_archive_stale_enabled: bool = Field(
        default=False, validation_alias="JOB_ARCHIVE_STALE_ENABLED"
    )

    # Job retry/backoff (shared)
    job_retry_max_attempts: int = 3
    job_retry_backoff_base: float = 2.0
    job_retry_backoff_max: float = 30.0

    # Versioning: snapshot before update when the caller omits `versioning`
    versioning_default: bool = Field(
        default=False, validation_alias="VERSIONING_DEFAULT"
    )

    # Validation limits (configurable via .env)
    max_text_length: int = 10_000  # Maximum text length in characters
    max_metadata_size: int = 100_000  # Maximum metadata size in bytes
    min_ttl_days: float = 0.0  # Minimum TTL in days
    max_ttl_days: float = 3650.0  # Maximum TTL in days (10 years)

    # Cache settings (configurable via .env)
    cache_embeddings_enabled: bool = True
    cache_embeddings_maxsize: int = 1000
    cache_search_enabled: bool = True
    cache_search_maxsize: int = 100
    cache_search_ttl: int = 60  # seconds

    @property
    def config(self) -> Dict[str, Any]:
        """Return the memory config dict (everything except MCP metadata)."""
        return self.model_dump(exclude={"enabled", "name", "description"})


def load_mcp_server_config() -> MCPServerConfig:
    """
    Load server config from environment variables and optional `.env`.

    If `.env` file doesn't exist, defaults + process env vars are used.
    """
    # NOTE: pydantic-settings handles missing `.env` gracefully.
    # This function exists mostly to keep call sites consistent.
    return MCPServerConfig()
