"""In-process lexical search modes: exact phrase, BM25, and typo-tolerant BM25.

Both stores return the owner's candidate rows (same filters as vector search)
through ``text_rows``. This module ranks them in Python, so the modes behave
the same on FalkorDB and ArcadeDB. An index per (owner, node type, filters) is
kept until the next write (``CacheManager.generation``) changes.

Scoring:

* ``exact``: case- and whitespace-insensitive substring match of the whole
  query (or of each ``"quoted phrase"`` when the query has quotes). Matches
  are ordered by BM25 of the query terms; non-matches are dropped.
* ``bm25``: Okapi BM25 (k1=1.2, b=0.75) over lower-cased word tokens.
* ``fuzzy``: each query term is expanded to vocabulary terms whose character
  trigram Jaccard similarity is at least ``FUZZY_MIN_SIM``. A matched term's
  BM25 contribution is scaled by that similarity, so exact hits still win.
"""

from __future__ import annotations

import math
import re
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_QUOTED = re.compile(r'"([^"]+)"')
_WS = re.compile(r"\s+")
K1 = 1.2
B = 0.75
FUZZY_MIN_SIM = 0.45
FUZZY_MAX_EXPANSIONS = 8
RRF_K = 60

LEXICAL_MODES = ("exact", "fuzzy", "bm25")
SEARCH_MODES = ("embedding", "exact", "fuzzy", "bm25", "hybrid")
DEFAULT_HYBRID_MODES = ("embedding", "bm25")


def tokenize(text: Optional[str]) -> List[str]:
    return TOKEN_RE.findall((text or "").lower())


def normalize_phrase(text: Optional[str]) -> str:
    return _WS.sub(" ", (text or "").lower()).strip()


def trigrams(token: str) -> set:
    padded = f"  {token} "
    return {padded[i : i + 3] for i in range(len(padded) - 2)}


def edit_similarity(a: str, b: str) -> float:
    """1 - OSA (Damerau) distance / max length; transpositions cost 1."""
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if not la or not lb or abs(la - lb) > 2:
        return 0.0
    prev2: List[int] = []
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        prev2, prev = prev, cur
    return 1.0 - prev[lb] / max(la, lb)


def normalize_search_mode(value: Optional[str]) -> str:
    """Return a mode from SEARCH_MODES. None or empty means ``embedding``."""
    if value is None or not str(value).strip():
        return "embedding"
    mode = str(value).strip().lower().replace("-", "_")
    aliases = {
        "vector": "embedding",
        "semantic": "embedding",
        "text": "bm25",
        "fulltext": "bm25",
        "full_text": "bm25",
        "keyword": "bm25",
        "phrase": "exact",
        "substring": "exact",
        "typo": "fuzzy",
    }
    mode = aliases.get(mode, mode)
    if mode not in SEARCH_MODES:
        raise ValueError(
            f"Invalid search_mode {value!r}; expected one of {', '.join(SEARCH_MODES)}"
        )
    return mode


def normalize_hybrid_modes(values: Optional[Sequence[str]]) -> Tuple[str, ...]:
    if not values:
        return DEFAULT_HYBRID_MODES
    out: List[str] = []
    for value in values:
        mode = normalize_search_mode(value)
        if mode == "hybrid":
            raise ValueError("hybrid_modes cannot contain 'hybrid'")
        if mode not in out:
            out.append(mode)
    return tuple(out)


@dataclass
class LexicalIndex:
    """BM25 statistics over one candidate set. ``rows`` keep the store row shape."""

    rows: List[List[Any]]
    doc_tf: List[Counter] = field(default_factory=list)
    doc_len: List[int] = field(default_factory=list)
    phrases: List[str] = field(default_factory=list)
    df: Counter = field(default_factory=Counter)
    postings: Dict[str, List[int]] = field(default_factory=lambda: defaultdict(list))
    trigram_vocab: Dict[str, List[str]] = field(
        default_factory=lambda: defaultdict(list)
    )
    avgdl: float = 0.0

    @classmethod
    def build(cls, rows: Iterable[List[Any]]) -> "LexicalIndex":
        index = cls(rows=list(rows))
        for i, row in enumerate(index.rows):
            text = row[2] if len(row) > 2 else ""
            toks = tokenize(text)
            tf = Counter(toks)
            index.doc_tf.append(tf)
            index.doc_len.append(len(toks))
            index.phrases.append(normalize_phrase(text))
            for term in tf:
                index.df[term] += 1
                index.postings[term].append(i)
        for term in index.df:
            for gram in trigrams(term):
                index.trigram_vocab[gram].append(term)
        n = len(index.rows)
        index.avgdl = (sum(index.doc_len) / n) if n else 0.0
        return index

    def idf(self, term: str) -> float:
        n = len(self.rows)
        df = self.df.get(term, 0)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def _bm25(self, weighted_terms: Dict[str, float]) -> Dict[int, float]:
        scores: Dict[int, float] = defaultdict(float)
        avgdl = self.avgdl or 1.0
        for term, weight in weighted_terms.items():
            idf = self.idf(term)
            for doc in self.postings.get(term, ()):
                tf = self.doc_tf[doc][term]
                denom = tf + K1 * (1 - B + B * self.doc_len[doc] / avgdl)
                scores[doc] += weight * idf * tf * (K1 + 1) / denom
        return scores

    def expand_fuzzy(self, term: str) -> Dict[str, float]:
        """Vocabulary terms similar to ``term`` with trigram Jaccard weights."""
        if term in self.df and len(term) <= 3:
            return {term: 1.0}
        grams = trigrams(term)
        overlap: Counter = Counter()
        for gram in grams:
            for cand in self.trigram_vocab.get(gram, ()):
                overlap[cand] += 1
        scored = []
        for cand, inter in overlap.items():
            union = len(grams) + len(trigrams(cand)) - inter
            sim = inter / union if union else 0.0
            if sim < FUZZY_MIN_SIM and len(term) >= 4:
                # trigrams miss transpositions in short words; fall back to edits
                sim = max(sim, edit_similarity(term, cand) - 0.2)
            if sim >= FUZZY_MIN_SIM:
                scored.append((sim, cand))
        scored.sort(key=lambda x: (-x[0], x[1]))
        out = {cand: sim for sim, cand in scored[:FUZZY_MAX_EXPANSIONS]}
        if term in self.df:
            out[term] = 1.0
        return out

    def search(
        self, query: str, mode: str, limit: int
    ) -> List[Tuple[List[Any], float]]:
        terms = tokenize(query)
        if mode == "bm25":
            weighted = {t: float(c) for t, c in Counter(terms).items()}
            scores = self._bm25(weighted)
        elif mode == "fuzzy":
            weighted: Dict[str, float] = defaultdict(float)
            for term, count in Counter(terms).items():
                for cand, sim in self.expand_fuzzy(term).items():
                    weighted[cand] = max(weighted[cand], sim * count)
            scores = self._bm25(weighted)
        elif mode == "exact":
            phrases = [normalize_phrase(p) for p in _QUOTED.findall(query)]
            phrases = [p for p in phrases if p] or [normalize_phrase(query)]
            if not phrases[0]:
                return []
            base = self._bm25({t: 1.0 for t in terms})
            scores = {}
            for doc, hay in enumerate(self.phrases):
                if all(p in hay for p in phrases):
                    # 1.0 floor so a match without BM25 overlap still ranks.
                    scores[doc] = 1.0 + base.get(doc, 0.0)
        else:
            raise ValueError(f"Not a lexical mode: {mode}")
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
        return [(self.rows[doc], score) for doc, score in ranked if score > 0]


_INDEX_LOCK = threading.Lock()
_MAX_INDEXES = 32


def cached_index(db: Any, key: Tuple, loader) -> LexicalIndex:
    """Index for ``key`` (owner, type, filters), rebuilt after any write."""
    generation = getattr(db.cache, "generation", None)
    with _INDEX_LOCK:
        store: dict = getattr(db, "_lexical_indexes", None)
        if store is None:
            store = {}
            setattr(db, "_lexical_indexes", store)
        hit = store.get(key)
        if hit is not None and generation is not None and hit[0] == generation:
            return hit[1]
    index = LexicalIndex.build(loader())
    with _INDEX_LOCK:
        if len(store) >= _MAX_INDEXES:
            store.pop(next(iter(store)))
        store[key] = (generation, index)
    return index


def rrf_fuse(ranked_lists: Sequence[Sequence[str]], k: int = RRF_K) -> Dict[str, float]:
    """Reciprocal rank fusion: sum of 1 / (k + rank), rank from 1."""
    scores: Dict[str, float] = defaultdict(float)
    for ranking in ranked_lists:
        for rank, node_id in enumerate(ranking, start=1):
            scores[node_id] += 1.0 / (k + rank)
    return dict(scores)
