"""Text representations for the intent classifier (task 3.5, design section 5).

Two representations live here, both behind `encode(...)`-shaped seams so the classifier and the
split dedup (3.3) read them the same way:

- The design-locked semantic representation is `paraphrase-multilingual-MiniLM-L12-v2` (384-d,
  CPU). One multilingual space covers es+pt, which is what makes Portuguese work with no PT
  training data. The real encoder is **≈470 MB of weights, more resident once torch is loaded**,
  so it is loaded LAZILY (never at import) and cached with `functools.lru_cache` (ponytail: a
  one-line stdlib cache, no hand-rolled cache class). It is never imported or downloaded in tests.
- A TF-IDF char-ngram variant (`analyzer="char_wb"`, 3-5) is the required comparison. It shares no
  space across es/pt the way MiniLM does, so the report reads it as the "no cross-lingual transfer"
  baseline (NIT-3): a large es-vs-pt gap here versus a small one on MiniLM is the contrast the
  comparison exists to surface.

Test path: `StubEncoder` returns small deterministic hash-based vectors and is selected when
`CORA_NLU_STUB=1`. Unit tests exercise the classifier plumbing on the stub with no network and no
470 MB download; the real encoder produces the committed offline metrics (REQ-47).

    # ponytail: torch is a heavy transitive dep for one 384-d encoder, justified because the
    # multilingual embedding is the design-locked representation and is what makes PT work with
    # zero PT training data. If a smaller ONNX export proves installable, swap it in behind
    # encode() without touching callers.
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache

import numpy as np

__all__ = ["STUB_DIM", "StubEncoder", "build_tfidf", "encode", "stub_enabled"]

# The real MiniLM dimensionality (model_card.json). The stub keeps a smaller, fixed dim: the
# classifier only needs a consistent vector length, and a smaller stub space keeps tests fast.
EMBED_DIM = 384
STUB_DIM = 64

# HF model id + revision are pinned in data/nlu/model_card.json; keep the id here only so the lazy
# loader has one literal to pass to sentence-transformers. It is NOT a hard-coded secret or a
# runtime-tunable; the real pin of record is the model card.
_HF_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def stub_enabled() -> bool:
    """True when tests/offline runs force the deterministic stub encoder (`CORA_NLU_STUB=1`)."""
    return os.getenv("CORA_NLU_STUB") == "1"


class StubEncoder:
    """Deterministic hash-based sentence vectors for tests (no network, no model download).

    Not a semantic encoder: it maps each utterance to a fixed-dim L2-normalized bag-of-char-trigram
    hash vector. Identical strings map to identical vectors and near-identical strings to similar
    ones, which is all the classifier *plumbing* tests and the 3.3 dedup fallback need. Seeded by
    the text itself, so results are reproducible across runs and machines.
    """

    def __init__(self, dim: int = STUB_DIM) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), self.dim), dtype=np.float64)
        for i, text in enumerate(texts):
            norm = " ".join(text.lower().split())
            for j in range(max(len(norm) - 2, 0)):
                tri = norm[j : j + 3]
                bucket = int(hashlib.blake2b(tri.encode("utf-8"), digest_size=8).hexdigest(), 16) % self.dim
                vecs[i, bucket] += 1.0
        lengths = np.linalg.norm(vecs, axis=1, keepdims=True)
        lengths[lengths == 0] = 1.0  # a (near-)empty string stays zero, not a divide-by-zero
        return vecs / lengths


@lru_cache(maxsize=1)
def _real_encoder():  # noqa: ANN202 - the ST type is only known once torch is importable
    """Load and cache the real MiniLM encoder. Imported here, never at module import (lazy)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(_HF_MODEL_ID, device="cpu")


@lru_cache(maxsize=1)
def _stub_encoder() -> StubEncoder:
    return StubEncoder()


def encode(texts: list[str]) -> np.ndarray:
    """Embed utterances, returning an (n, dim) float array. Uses the stub when `CORA_NLU_STUB=1`.

    This is the seam `splits.py` already imports for cross-split dedup and the classifier uses for
    training/inference, so both read the same representation.
    """
    if stub_enabled():
        return _stub_encoder().encode(list(texts))
    # normalize_embeddings so cosine == dot product downstream (dedup) and the LogReg sees unit
    # vectors (the design's frozen-embedding + linear-head setup).
    return np.asarray(
        _real_encoder().encode(list(texts), normalize_embeddings=True, show_progress_bar=False),
        dtype=np.float64,
    )


def build_tfidf():  # noqa: ANN201 - returns an unfitted sklearn TfidfVectorizer
    """The char-ngram TF-IDF variant for the required comparison (the "no cross-lingual" baseline).

    `char_wb` 3-5 grams: robust to the morphology/typo variety of short es/pt utterances, but with
    no shared cross-lingual space, which is exactly the contrast against MiniLM the report reads.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)


if __name__ == "__main__":  # tiny self-check on the stub (no network)
    os.environ["CORA_NLU_STUB"] = "1"
    _stub_encoder.cache_clear()
    v = encode(["hola", "hola", "quiero pagar"])
    assert v.shape == (3, STUB_DIM), v.shape
    assert np.allclose(v[0], v[1]), "identical strings must map to identical stub vectors"
    assert abs(np.linalg.norm(v[0]) - 1.0) < 1e-9, "stub vectors must be L2-normalized"
    print("embeddings stub self-check OK")
