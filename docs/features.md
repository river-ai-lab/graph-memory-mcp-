# API contract

Normative tool signatures. Jobs: [background_jobs.md](./background_jobs.md). Agents: [memory_policies_for_LLM.md](./memory_policies_for_LLM.md). Admin: [admin.md](./admin.md).

**Storage:** ArcadeDB by default, one database per `owner_id` (`{ARCADE_DATABASE}_{owner_id}`), with that owner's own HNSW indexes. `GRAPH_BACKEND=falkordb` uses one FalkorDB graph per owner (`{FALKORDB_GRAPH}_{owner_id}`). **IDs:** tools use stable `node_id` (= `uid`, UUID hex). A node without `uid` receives one at startup.

## Data model

**Common:** `node_id`, `owner_id`, `metadata`, `created_by`, `created_at`, `updated_by`, `updated_at`, `status` (`active`|`outdated`|`archived`), optional `ttl_days`/`expires_at`, `source` `{ref,type,uri,content_hash,source_updated_at,version}`, `embedding` (not returned via MCP). `created_at` / `updated_at` are set by the server. `source_updated_at` is the source document time (Unix ms), stored separately from that clock. Older `source.updated_at` is still accepted and read back as `source_updated_at`. In `client` and `header`, `created_by` / `updated_by` come from `metadata.created_by` / `metadata.updated_by`. In `token` they are the API key name.

**Fact** — `text`. **Entity** — optional `type`. Practical labels, not ontology.

**Reserved metadata** (typed, promoted, filterable): `project`, `created_by`, `tags`, `type`, `confidence`. Other keys free-form.

**Edges:** default allowlist `RELATED_TO`, `MENTIONS`, `SUMMARIZES`, `FOLLOWS_FROM`, `CONTRADICTS`, `EXTRACTED_FROM`, `SIMILAR_TO`, `NEXT_STEP`, `EXPOSES`, `CROSS_REF`, `RUNS_ON`, `USES`, `SERVED_BY`. Policy: `RELATION_POLICY_ENFORCE` (`off`|`warn`|`enforce`), `RELATION_ALLOWED_TYPES`. Fact `auto_link` (default false) → `MENTIONS` to similar Entities when passed true.

**Indexes:** Fact + Entity vectors — created on the first write for an owner (`create_node`, `create_nodes`, `upsert_node`, `create_relation`, entity creation, `import_owner`), cached per owner. Also on `search`, `find_similar`, or Fact `auto_link` when still missing, `AUTO_CREATE_INDEXES=true`, or `POST /admin/ensure-indexes`.

## Validation

| Field | Rule | Default |
|-------|------|---------|
| `text` | max length | 10_000 |
| `metadata` | max bytes | 100_000 |
| `ttl_days` | range | (0, 3650] |
| `owner_id` | `[A-Za-z0-9_@-]+` | |
| `relation_type` | `[A-Za-z0-9_]+` | |

Errors: `{success:false, error, code}` — `memory_validation_error`, `memory_not_found`, `memory_duplicate`, `memory_relation_policy_error`, `memory_forbidden`, `memory_service_error`, `memory_conflict`, `memory_embedding_error`. `memory_duplicate` also returns `possible_duplicates` (each item has `diff_hint`). The node is not written. `memory_embedding_error` means search did not run: an empty result list is not used for a missing vector. `memory_conflict` is the Falkor Redis lock when two upserts hit the same `(owner_id, node_type, source.ref)`. Arcade has no Redis client, so that lock is skipped.

## Cache

LRU embeddings + TTL search (defaults in `env.example`). Search cache cleared on mutations. In-process only. Falkor locks upsert and jobs with Redis. Arcade is one process and does not. Stats in `health_check` / `/metrics`.

## Server profiles

Independent processes. Tool names match. Pick one client-side and stay on it. Handlers are the same; write tools expose more or less.

| Profile | Start | What the agent can pass on writes |
|---------|--------|-------------------------------------|
| default | `graph-memory-mcp` | Nested `source`. Fact and Entity. |
| `--simple` | `graph-memory-mcp --simple` | Same tools. Provenance is flat (`ref`, `provenance_type`, `source_updated_at`, …) instead of nested `source`. `upsert_node` requires `ref`. |
| `--extended` | `graph-memory-mcp --extended` | Same tools as default, plus `description`, `shared_with_ids`, `collection_id` (`CONTAINS`), and node types `Collection`, `User`, `Project`. |
| `--auth` | `graph-memory-mcp --auth` | Same tools as default, without `owner_id`. `OWNER_SCOPE=token`. The server uses the key's first project. |

## Agent tools

Defaults: `owner_id` from `DEFAULT_OWNER_ID` (env, default `"default"`). Pass `owner_id` explicitly in agent rules.

`OWNER_SCOPE=client` (default) uses that argument. `OWNER_SCOPE=header` ignores it and takes `X-Owner-Id`; a missing header is `memory_validation_error` and does not use `DEFAULT_OWNER_ID`. Run header mode on `127.0.0.1` behind a gateway that sends `ADMIN_TOKEN`. A missing or wrong bearer is HTTP 401 when that token is set. `OWNER_SCOPE=token` requires a Bearer API key. `owner_id` must be one of that key's projects; omit it to use the first project. A project outside the key is `memory_forbidden`. A missing or revoked key is HTTP 401. Tool schemas do not change. `graph-memory-mcp --auth` is the exception: schemas omit `owner_id`, and the server uses the key's first project. Keys: `graph-memory-mcp keys create --name <name> --projects <p1,p2>` (prints the raw key once), `keys list`, `keys revoke <id>`. The file stores the sha256 hash. Jobs are in-process and keep their own owner. See [configuration.md](./configuration.md) and [admin.md](./admin.md).

### Writes

**`create_node`** — `text`; opt: `node_type` (`Fact`|`Entity`), `owner_id`, `metadata`, `status`, `ttl_days`, `source`, `entity_type`, `auto_link` (default false), `semantic_threshold`, `links[{to_id,relation_type,properties?}]`, `check_duplicates` (default true). `--extended` also takes `description`, `shared_with_ids`, `collection_id`, and `node_type` `Collection`|`User`|`Project`. The server never sets `ttl_days`. A Fact near-paraphrase at or above `CREATE_DUPLICATE_BLOCK_THRESHOLD` (0.92) is rejected (`memory_duplicate`, with `diff_hint`) before write. A similar but different fact is written with `check_duplicates=false`. Text longer than `LONG_TEXT_WARN_CHARS` (600) adds `warnings` and is still stored. `auto_link=true` adds `MENTIONS` to similar Entities (threshold `AUTO_LINKING_SEMANTIC_THRESHOLD`).
→ success `{node, possible_duplicates?, warnings?, link_errors?, link_warnings?}`

**`upsert_node`** — `text`, `source.ref`; same write fields as `create_node` except `check_duplicates` (upsert updates the node for that ref, including a revised `about`). Opt `description`, `auto_link` (default false) + `versioning`. `--extended` also takes `shared_with_ids` and `collection_id`. On Falkor, Redis-locked per `(owner_id, node_type, ref)`. Arcade does not take that lock. → `{success, node, operation}`

**`create_nodes`** — `items[{text,…}]` ≤200; opt `node_type`, `owner_id`. No auto_link / duplicates.

**`ingest_knowledge`** — agent extracts; server writes. `document{ref,…}`; opt `facts[{text, ref?,…}]`, `triplets[{subject,predicate,object,metadata?}]`, `owner_id`, `auto_link=false`. Fact key: `facts[].ref` or `hash(text)` → `doc#…` (≤200 facts/triplets).

**`update_node`** — `node_id`; opt fields + `versioning` (default `VERSIONING_DEFAULT`).
**`delete_node`** — hard delete (+ FactVersion cascade).
**`mark_outdated`** — `fact_id`, opt `reason`.

**`create_relation`** / **`delete_relation`** — `from_id`, `to_id`, `relation_type`; opt `properties` (`created_by` / `updated_by` in `client` and `header`). Delete: `relation_type` optional.
**`create_triplet`** / **`search_triplets`** — entities merge by `name_norm`; opt `metadata` on create (reserved keys coalesce on match) and `fact_id` (source fact). Search opt: `subject`, `predicate`, `object_value`, `limit` (default 10, cap `MAX_SEARCH_LIMIT`).
**`create_summary_fact`** — `fact_ids`, `summary_text`; opt `metadata`.

### Reads

**`get_node`** — `node_id`; opt `as_of` (unix ms, needs versioning snapshots).
**`get_node_change_history`** — `node_id`.
**`search`** — `query`; opt `limit` (default `DEFAULT_SEARCH_LIMIT`, 10; cap `MAX_SEARCH_LIMIT`), `node_types`, `status`, `similarity_threshold` (default `SEMANTIC_SIMILARITY_THRESHOLD`, 0.55), `include_outdated`, `include_ttl` (default false: hide facts that have `ttl_days`; true: also still-living temporary facts), `search_type` (`pre_filter`|`post_filter`; omitted uses `SEARCH_TYPE`, default `pre_filter`), `metadata_filter` (`project`/`created_by`/`type`/`tags`/`confidence_min`), `compact` (snippets + token budget). Rank is hybrid (`score`): cosine + identifier match + recency. `search_mode` (omitted uses `SEARCH_MODE`, default `embedding`, unchanged behaviour): `embedding` (vector path above), `bm25` (Okapi BM25 over fact text), `fuzzy` (BM25 with typo-tolerant term expansion: trigram + edit distance), `exact` (case-insensitive substring; quoted phrases must all match), `hybrid` (Reciprocal Rank Fusion of `hybrid_modes`, default `embedding`+`bm25`). Lexical hits return `score` and `match` instead of `similarity`; `similarity_threshold` and `search_type` apply to `embedding` only. `rerank=true` (or `SEARCH_RERANKER=true`) reorders the top `SEARCH_RERANK_TOP_K` (30) candidates with an ONNX cross-encoder (`SEARCH_RERANKER_MODEL`, default `Xenova/ms-marco-MiniLM-L-6-v2`, English-only, downloaded on first use); results get `rerank_score`. Lexical modes use an in-process index built per owner/node type/filters on first query and rebuilt after any write (same on FalkorDB and ArcadeDB; DB-native full-text indexes are not used). See `tests/benchmarks/search_modes/REPORT.md`. `contradiction_pairs` lists hits that already have a `CONTRADICTS` edge. Always returns `suggested_next`, `do_not`, `budget`. Bumps access counters.
**`find_similar`** — `fact_id`; opt `limit` (default 5), `similarity_threshold` (default `SEMANTIC_SIMILARITY_THRESHOLD`, 0.55).
**`get_context`** — `node_id`; opt `depth` (default `SUBGRAPH_DEFAULT_DEPTH`, 1), `max_nodes`, `offset`, `page` (optional, default false), `include_outdated`, `include_ttl`, `relation_priority`, `exclude_relation_types`. Omitting `page` with `offset=0` is the priority walk and returns the same nodes, edges, order, and fields as before `page` existed. No index or data migration. That walk fills with `CONTRADICTS`, then `FOLLOWS_FROM`, then other allowlist edges; `MENTIONS` last. It does not return `offset` or `has_more`. `offset>0` or `page=true` reads stable id-ordered pages. `offset>0`, and a full page (`page=true` included), also return edges from that page back to the rest of the context: the seed plus the first context ids, up to the hard node cap. Callers that only paginated with `offset` therefore see extra edges; the node list for that offset is unchanged. Default cap 20 (hard 50). A negative `offset` is clamped to 0. Paginated responses include `offset` and `has_more`. `has_more` is true only when the next page has at least one node that would be returned; hidden outdated or TTL neighbors do not count. An offset past the end returns no nodes, `has_more` false, and echoes `offset`. Outdated and archived neighbors are hidden by default; pass `include_outdated=true` to include them. Neighbors with `ttl_days` are omitted unless `include_ttl=true`. Each node includes `status`, `created_at`, and `metadata` when the store has them.
**`recall_context`** — shortcut search+expand (small graphs). Opt `depth` (default `RECALL_CONTEXT_DEFAULT_DEPTH`, 2), `limit` (default `RECALL_CONTEXT_SEED_LIMIT`, 5), `max_nodes` (default 20, hard 50), `similarity_threshold` (default 0.55), `include_outdated`, `include_ttl`, `search_type`, `include_paths`, `metadata_filter`, `compact`, `relation_priority`, `exclude_relation_types`, time-aware ranking weights. Same neighbor order and TTL filter as `get_context`. Always returns `suggested_next` / `do_not` / `budget`.
**`get_trace`** — `from_id`, `to_id`; opt `max_depth`, `directed` (default true). Path nodes include `status`, `created_at`, `updated_at`, and `metadata` when the store has them.
**`get_brief`** — warm-up, not search and not a degree ranking: `about` if a fact with `source.ref` = `ABOUT_SOURCE_REF` (default `about`) exists, else null. Recent active facts with no TTL, by time (`limit` cuts only these). Open `CONTRADICTS` between those durable facts. `stale_count` is active facts not accessed for `STALE_FACTS_DAYS`. `include_stale=true` returns their texts; leave it false at task start. Stats, including `outdated_facts` and `archived_facts`. Returns the resolved `owner_id`. Facts with `ttl_days` are omitted.
**`get_stats`** — Fact/Entity counts, plus `active_facts`, `outdated_facts`, and `archived_facts`. **`health_check`** — backend, embeddings, and `healthy`. `vector_index_missing` lists up to 20 owners that have no vector index yet; that list does not clear `healthy`. The first write creates both indexes. Search creates them if they are still missing.

## Admin (HTTP; not MCP by default)

See [admin.md](./admin.md): `/admin/health`, `/admin/ensure-indexes`, export/import (incl. `FactVersion` when `include_versions=true`), delete/prune owners, `/metrics`, `/internal/graph-size`.
Set `MCP_EXPOSE_ADMIN_TOOLS=true` to also expose `test_connection`, `ensure_vector_indexes`, `export_owner`, `import_owner` as MCP tools. An `import_owner` tool call is a streamable HTTP POST; the SDK rejects a body larger than 4 MiB. `/admin/import` is not under that limit.
