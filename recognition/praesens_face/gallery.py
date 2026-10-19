"""In-memory FAISS gallery of enrolled guest embeddings.

Each enrollment photo becomes one vector, so a guest enrolled with three
photos has three vectors. A search returns the best-scoring photo per guest.
Vectors are L2-normalised, so inner product (IndexFlatIP) is cosine similarity.

Nothing here is written to disk. After a restart, rebuild the gallery from the
stored enrollment photos (see FaceEngine.rebuild).
"""

from __future__ import annotations

import hashlib
import itertools
import threading
from dataclasses import dataclass

import faiss
import numpy as np


@dataclass
class GalleryHit:
    guest_id: str
    score: float


class Gallery:
    def __init__(self, dim: int = 512):
        self.dim = dim
        self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        self._vec_to_guest: dict[int, str] = {}
        self._guest_to_vecs: dict[str, list[int]] = {}
        self._next_id = itertools.count(1)
        self._lock = threading.RLock()

    # --- writes -------------------------------------------------------------
    def add(self, guest_id: str, embeddings: np.ndarray) -> int:
        """Add one or more embeddings (shape (n, dim)) for a guest."""
        emb = _as_matrix(embeddings, self.dim)
        with self._lock:
            ids = np.array([next(self._next_id) for _ in range(len(emb))], dtype=np.int64)
            self._index.add_with_ids(emb, ids)
            for vid in ids.tolist():
                self._vec_to_guest[vid] = guest_id
            self._guest_to_vecs.setdefault(guest_id, []).extend(ids.tolist())
            return len(ids)

    def replace(self, guest_id: str, embeddings: np.ndarray) -> int:
        with self._lock:
            self.remove(guest_id)
            return self.add(guest_id, embeddings)

    def remove(self, guest_id: str) -> int:
        """Remove every vector for a guest. Returns how many were removed."""
        with self._lock:
            vids = self._guest_to_vecs.pop(guest_id, [])
            if vids:
                self._index.remove_ids(np.array(vids, dtype=np.int64))
                for vid in vids:
                    del self._vec_to_guest[vid]
            return len(vids)

    def clear(self) -> None:
        with self._lock:
            self._index.reset()
            self._vec_to_guest.clear()
            self._guest_to_vecs.clear()

    # --- reads --------------------------------------------------------------
    def search(self, queries: np.ndarray, top_k: int = 1) -> list[list[GalleryHit]]:
        """For each query embedding, return up to top_k guests, best first."""
        q = _as_matrix(queries, self.dim)
        with self._lock:
            n = self._index.ntotal
            if n == 0:
                return [[] for _ in range(len(q))]
            # Ask for extra neighbours because one guest can own several vectors.
            k = min(n, max(top_k * 5, top_k))
            scores, vids = self._index.search(q, k)
            mapping = dict(self._vec_to_guest)
        results = []
        for row_scores, row_ids in zip(scores, vids):
            best: dict[str, float] = {}
            for s, vid in zip(row_scores.tolist(), row_ids.tolist()):
                if vid == -1:
                    continue
                g = mapping[vid]
                if g not in best:  # FAISS returns results sorted best-first
                    best[g] = s
                if len(best) == top_k:
                    break
            results.append([GalleryHit(g, s) for g, s in best.items()])
        return results

    def guests(self) -> dict[str, int]:
        with self._lock:
            return {g: len(v) for g, v in self._guest_to_vecs.items()}

    def __contains__(self, guest_id: str) -> bool:
        with self._lock:
            return guest_id in self._guest_to_vecs

    @property
    def size(self) -> int:
        return int(self._index.ntotal)

    def fingerprint(self) -> str:
        """SHA-256 over the gallery contents (guest ids + their vectors).

        Independent of insertion order and internal ids, so the same set of
        enrollments always gives the same hash. Useful for the purge check:
        enroll A and B, record the hash, enroll C, delete C, and the hash must
        match the recorded one again.
        """
        with self._lock:
            h = hashlib.sha256()
            for g in sorted(self._guest_to_vecs):
                vecs = [self._index.reconstruct(vid) for vid in self._guest_to_vecs[g]]
                h.update(g.encode())
                for v in sorted(np.round(np.asarray(v, dtype=np.float32), 6).tobytes() for v in vecs):
                    h.update(v)
            return h.hexdigest()


def _as_matrix(x: np.ndarray, dim: int) -> np.ndarray:
    m = np.ascontiguousarray(np.atleast_2d(x), dtype=np.float32)
    if m.shape[1] != dim:
        raise ValueError(f"expected embeddings of dim {dim}, got {m.shape[1]}")
    return m
