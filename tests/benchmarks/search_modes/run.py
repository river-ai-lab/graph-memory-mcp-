#!/usr/bin/env python3
"""Search-mode benchmark on Scenario 1 data (no chat LLM).

Compares search_mode = embedding | exact | fuzzy | bm25 | hybrid, with and
without the cross-encoder reranker, on the Scenario 1 structural questions
plus two probe sets added here:

* typo: core facts queried with misspelled names (one transposition per name);
* ident: 24 ticket facts whose ids differ by one digit, queried by bare id.

Noise levels are nested (0, +1k, +9k) on one owner. Tokens use the Scenario 1
regex proxy. Metrics per question: MRR of the first gold fact, R@5 (share of
gold in the top 5), recall within a 1000-token budget, tokens to collect all
gold, latency (median, warm lexical index; index build time is reported).

  GRAPH_BACKEND=falkordb uv run python tests/benchmarks/search_modes/run.py
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
LEVELS = (0, 1000, 10000)
BUDGET = 1000
TOKEN_RE = re.compile(r"\w+|[^\w\s]")
OWNER = "searchmodes"
LIMIT = 100

logging.basicConfig(level=logging.ERROR)

SYSTEMS = {
    "embedding": dict(search_mode="embedding"),
    "exact": dict(search_mode="exact"),
    "fuzzy": dict(search_mode="fuzzy"),
    "bm25": dict(search_mode="bm25"),
    "hybrid(emb+bm25)": dict(search_mode="hybrid"),
    "hybrid(emb+bm25+fuzzy)": dict(
        search_mode="hybrid", hybrid_modes=["embedding", "bm25", "fuzzy"]
    ),
    "embedding+rerank": dict(search_mode="embedding", rerank=True),
    "hybrid(emb+bm25)+rerank": dict(search_mode="hybrid", rerank=True),
}


def ntok(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def mangle(word: str) -> str:
    return word[0] + word[2] + word[1] + word[3:] if len(word) >= 4 else word


def probe_sets(core_facts: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(7)
    typo = []
    for f in rng.sample([f for f in core_facts if f["status"] == "active"], 16):
        words = f["text"].rstrip(".").split()
        q = " ".join(mangle(w) if w[:1].isupper() else w for w in words)
        typo.append(
            {"id": f"t{len(typo):02d}", "type": "typo", "q": q, "gold": [f["id"]]}
        )
    ident_facts, ident = [], []
    services = ["ledger-api", "auth-gw", "search-svc", "billing-core"]
    for i in range(12):
        base = 48200 + i * 7
        for j, num in enumerate((base, base + 1)):  # one-digit-apart twin ids
            fid = f"x{i:02d}{j}"
            svc = services[(i + j) % 4]
            ident_facts.append(
                {
                    "id": fid,
                    "status": "active",
                    "entities": [],
                    "text": f"Ticket OPS-{num} was closed after fixing a stale cache in {svc}.",
                }
            )
        ident.append(
            {
                "id": f"i{i:02d}",
                "type": "ident",
                "q": f"OPS-{base}",
                "gold": [f"x{i:02d}0"],
            }
        )
    return typo, ident, ident_facts


def score(ctx: list[tuple[str, str]], gold: list[str], lat: float) -> dict:
    gold_s = set(gold)
    cum, found, all_at, rr = 0, {}, None, 0.0
    for rank, (fid, text) in enumerate(ctx, start=1):
        cum += ntok(text)
        if fid in gold_s and fid not in found:
            found[fid] = cum
            if not rr:
                rr = 1.0 / rank
            if len(found) == len(gold_s):
                all_at = cum
    top5 = {fid for fid, _ in ctx[:5]}
    return {
        "mrr": rr,
        "r@5": len(top5 & gold_s) / len(gold_s),
        f"recall@{BUDGET}": sum(1 for t in found.values() if t <= BUDGET) / len(gold_s),
        "tokens_all_gold": all_at,
        "n_items": len(ctx),
        "latency_ms": lat,
    }


def aggregate(rows: list[dict]) -> dict:
    out = {"n": len(rows)}
    for k in ("mrr", "r@5", f"recall@{BUDGET}"):
        out[k] = round(sum(r[k] for r in rows) / len(rows), 3)
    toks = [r["tokens_all_gold"] for r in rows if r["tokens_all_gold"] is not None]
    out["all_gold_found"] = f"{len(toks)}/{len(rows)}"
    out["median_tokens_all_gold"] = statistics.median(toks) if toks else None
    out["median_items"] = statistics.median(r["n_items"] for r in rows)
    out["latency_ms_median"] = round(
        statistics.median(r["latency_ms"] for r in rows), 1
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "results.json"))
    ap.add_argument("--levels", default=",".join(map(str, LEVELS)))
    ap.add_argument("--systems", default=",".join(SYSTEMS))
    args = ap.parse_args()
    sys.path.insert(0, str(HERE))
    from generate import build

    data = build()
    typo, ident, ident_facts = probe_sets(data["core"]["facts"])
    questions = data["questions"] + typo + ident

    from graph_memory_mcp.config import load_mcp_server_config
    from graph_memory_mcp.server import GraphMemoryMCP

    cfg = load_mcp_server_config()
    server = GraphMemoryMCP(cfg)
    server.db_client.delete_owner_graph(OWNER)
    uid, rev = {}, {}

    def bulk(items: list[dict], node_type: str) -> None:
        for i in range(0, len(items), 200):
            chunk = items[i : i + 200]
            r = server.create_nodes(
                items=[
                    {"text": x["text"], "metadata": {"bench_id": x["id"]}}
                    for x in chunk
                ],
                node_type=node_type,
                owner_id=OWNER,
            )
            for x, n in zip(chunk, r["nodes"]):
                uid[x["id"]] = n["node_id"]
                rev[n["node_id"]] = x["id"]

    def add(facts, ents, edges):
        bulk([e for e in ents if e["id"] not in uid], "Entity")
        bulk(facts, "Fact")
        for f in facts:
            if f["status"] == "outdated":
                server.mark_outdated(
                    fact_id=uid[f["id"]], reason="superseded", owner_id=OWNER
                )
        for e in edges:
            a, b = uid.get(e["from"]), uid.get(e["to"])
            if a and b:
                server.create_relation(
                    from_id=a, to_id=b, relation_type=e["type"], owner_id=OWNER
                )

    systems = {k: SYSTEMS[k] for k in args.systems.split(",")}
    results = {"backend": cfg.graph_backend, "limit": LIMIT, "levels": {}}
    core, noise, prev = data["core"], data["noise"], 0
    for lvl in [int(x) for x in args.levels.split(",")]:
        t0 = time.time()
        if not uid:
            add(core["facts"] + ident_facts, core["entities"], core["edges"])
        if lvl > prev:
            facts = noise["facts"][prev:lvl]
            ids = {f["id"] for f in facts}
            ments = {e for f in facts for e in f["entities"]}
            ents = [e for e in noise["entities"] if e["id"] in ments]
            known = set(uid) | ids | ments
            edges = [
                e
                for e in noise["edges"]
                if (
                    e["from"] in ids
                    or e["to"] in ids
                    or (e["from"] in ments and e["to"] in ments)
                )
                and e["from"] in known
                and e["to"] in known
            ]
            add(facts, ents, edges)
        prev = lvl
        print(f"level {lvl}: ingested in {time.time() - t0:.1f}s", flush=True)
        level_out = {"systems": {}}
        for name, kw in systems.items():
            t = time.perf_counter()
            server.search(
                query="warm up query",
                owner_id=OWNER,
                limit=LIMIT,
                node_types=["Fact"],
                **kw,
            )
            warm_ms = (time.perf_counter() - t) * 1000
            rows = []
            for q in questions:
                t = time.perf_counter()
                r = server.search(
                    query=q["q"], owner_id=OWNER, limit=LIMIT, node_types=["Fact"], **kw
                )
                lat = (time.perf_counter() - t) * 1000
                ctx = [
                    (rev.get(x["node_id"], "?"), x.get("text") or "")
                    for x in r.get("results") or []
                ]
                row = score(ctx, q["gold"], lat)
                row.update(id=q["id"], type=q["type"])
                rows.append(row)
            groups = {
                "scenario1": [r for r in rows if r["type"] not in ("typo", "ident")],
                "chain2": [r for r in rows if r["type"] == "chain2"],
                "typo": [r for r in rows if r["type"] == "typo"],
                "ident": [r for r in rows if r["type"] == "ident"],
            }
            agg = {g: aggregate(v) for g, v in groups.items() if v}
            level_out["systems"][name] = {
                "warmup_ms": round(warm_ms, 1),
                "agg": agg,
                "rows": rows,
            }
            s1 = agg["scenario1"]
            print(
                f"  {name:26s} S1 mrr={s1['mrr']} r@5={s1['r@5']} rec@1k={s1[f'recall@{BUDGET}']} "
                f"typo mrr={agg['typo']['mrr']} ident mrr={agg['ident']['mrr']} "
                f"lat={s1['latency_ms_median']}ms warm={warm_ms:.0f}ms",
                flush=True,
            )
        results["levels"][str(lvl)] = level_out
        Path(args.out).write_text(json.dumps(results, indent=1, default=str) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
