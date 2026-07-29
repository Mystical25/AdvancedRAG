"""
Embedding model wrapper used across indexing, retrieval, and evaluation.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    DEVICE,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    EMBEDDING_QUERY_PREFIX,
    NORMALIZE_EMBEDDINGS,
)
from utils.logger import get_logger


logger = get_logger(__name__)


class EmbeddingModel:
    """
    Thin wrapper around SentenceTransformer for local embeddings.
    """

    def __init__(self) -> None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        try:
            self.model = SentenceTransformer(
                EMBEDDING_MODEL,
                device=DEVICE,
                local_files_only=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Embedding model {EMBEDDING_MODEL} could not be loaded: {exc}"
            ) from exc

    def encode(
        self,
        texts: Sequence[str],
    ):
        items = list(texts)
        if not items:
            return np.empty((0, self.dimension), dtype=np.float32)

        assert self.model is not None
        return self.model.encode(
            items,
            batch_size=EMBEDDING_BATCH_SIZE,
            normalize_embeddings=NORMALIZE_EMBEDDINGS,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

    def encode_query(
        self,
        text: str,
    ):
        assert self.model is not None
        query = text.strip()
        if EMBEDDING_QUERY_PREFIX:
            query = f"{EMBEDDING_QUERY_PREFIX} {query}"
        return self.model.encode(
            query,
            normalize_embeddings=NORMALIZE_EMBEDDINGS,
            convert_to_numpy=True,
        )

    @property
    def dimension(self) -> int:
        assert self.model is not None
        return self.model.get_sentence_embedding_dimension()
