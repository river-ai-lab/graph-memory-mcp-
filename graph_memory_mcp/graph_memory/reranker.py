"""Optional cross-encoder reranker (ONNX Runtime, no PyTorch).

Off by default. ``SEARCH_RERANKER_MODEL`` names a Hugging Face repo with
``onnx/model.onnx`` and ``tokenizer.json`` (default
``Xenova/ms-marco-MiniLM-L-6-v2``, about 90 MB, English). The model scores
(query, text) pairs; the top ``SEARCH_RERANK_TOP_K`` results are reordered by
that score. Uses the same runtime and download helpers as the embedding
backend, so it adds no dependency.
"""

from __future__ import annotations

import threading
from typing import Dict, List

import numpy as np

from graph_memory_mcp.graph_memory.embedding_backend import (
    _download_required,
    _import_onnxruntime,
    _load_tokenizer,
    _ort_int_dtype,
)

DEFAULT_RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
_LOCK = threading.Lock()
_LOADED: Dict[str, "CrossEncoderReranker"] = {}
_MAX_LEN = 256
_BATCH = 32


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        ort = _import_onnxruntime()
        self.model_name = model_name
        self._tokenizer = _load_tokenizer(model_name)
        self._tokenizer.enable_truncation(_MAX_LEN)
        path = _download_required(model_name, "onnx/model.onnx")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        self._session = ort.InferenceSession(
            path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._inputs = list(self._session.get_inputs())

    def score(self, query: str, texts: List[str]) -> np.ndarray:
        out: List[float] = []
        for start in range(0, len(texts), _BATCH):
            chunk = texts[start : start + _BATCH]
            enc = self._tokenizer.encode_batch([(query, t or "") for t in chunk])
            ids = np.asarray([e.ids for e in enc])
            mask = np.asarray([e.attention_mask for e in enc])
            types = np.asarray([e.type_ids for e in enc])
            feed = {}
            for spec in self._inputs:
                dtype = _ort_int_dtype(spec.type)
                src = {
                    "input_ids": ids,
                    "attention_mask": mask,
                    "token_type_ids": types,
                }.get(spec.name)
                if src is None:
                    raise RuntimeError(f"Unsupported reranker input {spec.name!r}")
                feed[spec.name] = src.astype(dtype, copy=False)
            logits = np.asarray(self._session.run(None, feed)[0], dtype=np.float32)
            out.extend(logits.reshape(len(chunk), -1)[:, 0].tolist())
        return np.asarray(out, dtype=np.float32)


def load_reranker(model_name: str | None) -> CrossEncoderReranker:
    name = model_name or DEFAULT_RERANKER_MODEL
    with _LOCK:
        hit = _LOADED.get(name)
        if hit is None:
            hit = CrossEncoderReranker(name)
            _LOADED[name] = hit
        return hit


def rerank(
    query: str, results: List[Dict], *, model_name: str | None, top_k: int
) -> List[Dict]:
    """Reorder the first ``top_k`` results by cross-encoder score; keep the tail."""
    if not results:
        return results
    head, tail = results[:top_k], results[top_k:]
    scores = load_reranker(model_name).score(query, [r.get("text") or "" for r in head])
    ranked = []
    for item, s in zip(head, scores):
        updated = dict(item)
        updated["rerank_score"] = round(float(s), 6)
        ranked.append(updated)
    ranked.sort(key=lambda r: -r["rerank_score"])
    return ranked + tail
