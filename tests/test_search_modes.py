"""search_mode: exact, fuzzy, bm25, hybrid on both stores; lexical index unit tests."""

from __future__ import annotations

import uuid

import pytest

from graph_memory_mcp.graph_memory.lexical_search import (
    LexicalIndex,
    normalize_hybrid_modes,
    normalize_search_mode,
    rrf_fuse,
)
from graph_memory_mcp.graph_memory.mcp_handlers_nodes import create_node
from graph_memory_mcp.graph_memory.mcp_handlers_search import search

ROWS = [
    [
        "a",
        "Fact",
        "Deploy job RUN_20250301T101500Z failed on node gpu-7",
        None,
        1,
        "{}",
        0.0,
    ],
    ["b", "Fact", "The billing service moved to PostgreSQL 16", None, 1, "{}", 0.0],
    [
        "c",
        "Fact",
        "Kubernetes cluster autoscaler limits set to 48Gi",
        None,
        1,
        "{}",
        0.0,
    ],
    ["d", "Fact", "Alice prefers dark roast coffee", None, 1, "{}", 0.0],
]


def _ids(hits):
    return [row[0] for row, _ in hits]


def test_bm25_ranks_keyword_match_first():
    idx = LexicalIndex.build(ROWS)
    assert _ids(idx.search("billing postgresql", "bm25", 10))[0] == "b"
    assert idx.search("nothing matches here", "bm25", 10) == []


def test_fuzzy_tolerates_typos_bm25_does_not():
    idx = LexicalIndex.build(ROWS)
    assert idx.search("kubernetis autoscaller", "bm25", 10) == []
    assert _ids(idx.search("kubernetis autoscaller", "fuzzy", 10))[0] == "c"


def test_exact_requires_phrase():
    idx = LexicalIndex.build(ROWS)
    assert _ids(idx.search("dark roast", "exact", 10)) == ["d"]
    assert idx.search("roast dark", "exact", 10) == []
    assert _ids(idx.search('"gpu-7" "deploy job"', "exact", 10)) == ["a"]


def test_mode_parsing_and_rrf():
    assert normalize_search_mode(None) == "embedding"
    assert normalize_search_mode("vector") == "embedding"
    assert normalize_search_mode("BM25") == "bm25"
    with pytest.raises(ValueError):
        normalize_search_mode("magic")
    assert normalize_hybrid_modes(None) == ("embedding", "bm25")
    with pytest.raises(ValueError):
        normalize_hybrid_modes(["hybrid"])
    fused = rrf_fuse([["x", "y"], ["y", "z"]])
    assert max(fused, key=fused.get) == "y"


def test_search_modes_on_store(memory_backend):
    cfg, db = memory_backend.config, memory_backend.db
    owner = f"pytest_modes_{uuid.uuid4().hex[:8]}"
    texts = [r[2] for r in ROWS]
    ids = {}
    for t in texts:
        r = create_node(db, cfg, text=t, owner_id=owner, auto_link=False)
        ids[t] = r["node"]["node_id"]
    try:
        legacy = search(db, cfg, query="coffee preferences", owner_id=owner)
        explicit = search(
            db, cfg, query="coffee preferences", owner_id=owner, search_mode="embedding"
        )
        assert [x["node_id"] for x in legacy["results"]] == [
            x["node_id"] for x in explicit["results"]
        ]
        assert legacy["results"][0]["node_id"] == ids[texts[3]]

        r = search(
            db, cfg, query="RUN_20250301T101500Z", owner_id=owner, search_mode="exact"
        )
        assert [x["node_id"] for x in r["results"]] == [ids[texts[0]]]
        assert r["results"][0]["match"] == "exact"

        r = search(
            db, cfg, query="postgres billing", owner_id=owner, search_mode="bm25"
        )
        assert r["results"][0]["node_id"] == ids[texts[1]]

        r = search(
            db, cfg, query="kubernetis autoscaller", owner_id=owner, search_mode="fuzzy"
        )
        assert r["results"][0]["node_id"] == ids[texts[2]]

        r = search(
            db, cfg, query="billing database", owner_id=owner, search_mode="hybrid"
        )
        assert r["results"][0]["node_id"] == ids[texts[1]]
        assert "score" in r["results"][0]

        # A write must be visible to the cached lexical index.
        new = create_node(
            db, cfg, text="Zebra migration finished", owner_id=owner, auto_link=False
        )
        r = search(db, cfg, query="zebra", owner_id=owner, search_mode="bm25")
        assert [x["node_id"] for x in r["results"]] == [new["node"]["node_id"]]

        bad = search(db, cfg, query="x", owner_id=owner, search_mode="nope")
        assert bad["success"] is False
    finally:
        db.delete_owner_graph(owner)


def test_reranker_orders_relevant_first():
    """Cross-encoder puts the on-topic text first (skips if the model is unavailable)."""
    from graph_memory_mcp.graph_memory import reranker

    try:
        reranker.load_reranker(None)
    except Exception as exc:  # offline CI, no cached model
        pytest.skip(f"reranker model unavailable: {exc}")
    results = [
        {"node_id": "a", "text": "The cafeteria serves lunch at noon."},
        {"node_id": "b", "text": "Postgres was upgraded to version 16 in March."},
    ]
    out = reranker.rerank(
        "which postgres version do we run", results, model_name=None, top_k=10
    )
    assert out[0]["node_id"] == "b"
    assert "rerank_score" in out[0]
