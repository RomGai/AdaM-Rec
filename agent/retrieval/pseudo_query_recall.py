"""Retrieve historical targets with each pseudo query in two embedding spaces."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


class PseudoQueryDualRetriever:
    """Reuse normalized item caches; encode queries separately for each branch.

    Each matrix must follow its own item ID order. Queries are text-only even
    for the multimodal encoder: the historical target image must not be passed
    as part of a query whose purpose is to retrieve that target.
    """

    def __init__(
        self,
        *,
        text_item_ids: Sequence[str],
        text_item_embeddings: np.ndarray,
        vl_item_ids: Sequence[str],
        vl_item_embeddings: np.ndarray,
        encode_text: Callable[[str], np.ndarray],
        encode_multimodal: Callable[[str], np.ndarray],
    ) -> None:
        self.text_embeddings, self.text_positions = self._catalog(text_item_ids, text_item_embeddings)
        self.vl_embeddings, self.vl_positions = self._catalog(vl_item_ids, vl_item_embeddings)
        self.encode_text = encode_text
        self.encode_multimodal = encode_multimodal

    @staticmethod
    def _catalog(ids: Sequence[str], embeddings: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
        matrix = np.asarray(embeddings)
        positions = {str(iid): index for index, iid in enumerate(ids)}
        if matrix.ndim != 2 or matrix.shape[0] != len(ids) or matrix.shape[1] == 0:
            raise ValueError("Item embedding rows must match their branch's item IDs.")
        if len(positions) != len(ids):
            raise ValueError("Duplicate item IDs in embedding cache.")
        return matrix, positions

    @staticmethod
    def _target_rank(matrix: np.ndarray, embedding: np.ndarray, index: int) -> int:
        query = np.asarray(embedding, dtype=np.float32)
        if query.ndim == 2 and query.shape[0] == 1:
            query = query[0]
        if query.ndim != 1 or query.shape[0] != matrix.shape[1]:
            raise ValueError("Query embedding must be a single vector in the branch's embedding space.")
        norm = float(np.linalg.norm(query))
        if not np.isfinite(query).all() or not np.isfinite(norm) or norm <= 0:
            raise ValueError("Query embedding must be finite and nonzero.")
        scores = matrix @ (query / norm)
        if not np.isfinite(scores).all():
            raise ValueError("Non-finite retrieval scores.")
        # Exact one-based rank, with cache order breaking ties. Avoid sorting
        # the entire catalog when only the historical target's rank is needed.
        target_score = scores[index]
        return 1 + int(np.count_nonzero(scores > target_score)) + int(
            np.count_nonzero(scores[:index] == target_score)
        )

    def target_ranks(self, pseudo_query: str, target_item_id: str) -> tuple[int | None, int | None]:
        if not pseudo_query.strip():
            raise ValueError("Pseudo query must not be empty.")
        text_index = self.text_positions.get(target_item_id)
        vl_index = self.vl_positions.get(target_item_id)
        # Cache absence is not a retrieval failure. Skip incomplete pairs rather
        # than manufacturing a poor rank and biasing routing against a modality.
        if text_index is None or vl_index is None:
            return None, None
        text_rank = self._target_rank(self.text_embeddings, self.encode_text(pseudo_query), text_index)
        vl_rank = self._target_rank(self.vl_embeddings, self.encode_multimodal(pseudo_query), vl_index)
        return text_rank, vl_rank
