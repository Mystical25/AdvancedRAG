"""
==================================================
Hybrid Retrieval Pipeline
==================================================

Responsibilities
----------------
1. Load indexed artifacts
2. Embed the query
3. Dense retrieval (FAISS)
4. Sparse retrieval (BM25)
5. Merge candidates with reciprocal-rank fusion
6. Cross-encoder reranking
7. Return final chunks

==================================================
"""

from __future__ import annotations

from collections import defaultdict
import re

import faiss
import numpy as np

from config import BM25_TOP_K, FINAL_TOP_K, RERANK_TOP_K, VECTOR_TOP_K
from models.embedding import EmbeddingModel
from models.reranker import Reranker
from pipeline.indexing import IndexedChunk, IndexingPipeline
from utils.helpers import normalize_whitespace
from utils.logger import get_logger


logger = get_logger(__name__)


class HybridRetriever:
    """
    Hybrid dense + sparse retriever with reranking.
    """

    def __init__(self) -> None:
        if not IndexingPipeline.artifacts_exist():
            raise FileNotFoundError(
                "Retrieval artifacts were not found. Build the index before querying."
            )

        logger.info("Loading retrieval artifacts...")
        self.embedding_model = EmbeddingModel()
        self.reranker = Reranker()
        self.chunks = IndexingPipeline.load_chunks()

        if self._needs_chunk_repair(self.chunks):
            logger.warning(
                "Legacy image descriptions were detected; rebuilding retrieval indexes in memory."
            )
            self.chunks = self._repair_chunks(self.chunks)
            self.faiss_index, self.bm25 = self._build_indexes(self.chunks)
        else:
            self.faiss_index = IndexingPipeline.load_faiss()
            self.bm25 = IndexingPipeline.load_bm25()

    def retrieve(
        self,
        query: str,
    ) -> list[IndexedChunk]:
        """
        Retrieve the most relevant chunks for a query.
        """

        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Query must not be empty.")

        dense_candidates = self._dense_search(normalized_query)
        sparse_candidates = self._sparse_search(normalized_query)
        merged = self._merge_candidates(dense_candidates, sparse_candidates)
        return self._rerank(normalized_query, merged)

    def _dense_search(
        self,
        query: str,
    ) -> list[tuple[IndexedChunk, float]]:
        query_embedding = self.embedding_model.encode_query(query).astype(np.float32)
        query_embedding = np.expand_dims(query_embedding, axis=0)
        faiss.normalize_L2(query_embedding)

        scores, indices = self.faiss_index.search(query_embedding, VECTOR_TOP_K)

        results: list[tuple[IndexedChunk, float]] = []
        for score, index in zip(scores[0], indices[0]):
            if index == -1:
                continue
            results.append((self.chunks[index], float(score)))

        return results

    def _sparse_search(
        self,
        query: str,
    ) -> list[tuple[IndexedChunk, float]]:
        tokens = IndexingPipeline.tokenize_text(query)
        scores = self.bm25.get_scores(tokens)
        ranked = np.argsort(scores)[::-1][:BM25_TOP_K]

        results: list[tuple[IndexedChunk, float]] = []
        for index in ranked:
            results.append((self.chunks[int(index)], float(scores[index])))

        return results

    @staticmethod
    def _merge_candidates(
        dense: list[tuple[IndexedChunk, float]],
        sparse: list[tuple[IndexedChunk, float]],
    ) -> list[IndexedChunk]:
        """
        Merge retrieval results with reciprocal-rank fusion.
        """

        fused_scores: dict[str, float] = defaultdict(float)
        merged_chunks: dict[str, IndexedChunk] = {}
        rrf_constant = 60

        for rank, (chunk, _) in enumerate(dense, start=1):
            fused_scores[chunk.id] += 1.0 / (rrf_constant + rank)
            merged_chunks[chunk.id] = chunk

        for rank, (chunk, _) in enumerate(sparse, start=1):
            fused_scores[chunk.id] += 1.0 / (rrf_constant + rank)
            merged_chunks[chunk.id] = chunk

        sorted_ids = sorted(
            fused_scores,
            key=fused_scores.get,
            reverse=True,
        )

        return [merged_chunks[chunk_id] for chunk_id in sorted_ids[:RERANK_TOP_K]]

    @staticmethod
    def _needs_chunk_repair(chunks: list[IndexedChunk]) -> bool:
        for chunk in chunks:
            if chunk.source_type != "image":
                continue
            text = " ".join(
                part
                for part in (
                    chunk.raw_text or "",
                    chunk.text or "",
                )
                if part
            )
            if re.search(
                r"(?is)\bsystem\b.*?\buser\b.*?\bassistant\b",
                text,
            ):
                return True
        return False

    @staticmethod
    def _repair_chunks(chunks: list[IndexedChunk]) -> list[IndexedChunk]:
        repaired: list[IndexedChunk] = []
        for chunk in chunks:
            if chunk.source_type != "image":
                repaired.append(chunk)
                continue

            description = HybridRetriever._extract_image_description(chunk)
            if not description:
                repaired.append(chunk)
                continue

            image_id = chunk.metadata.get("image_id", chunk.id)
            contextualized_text = IndexingPipeline._build_chunk_text(
                body=(
                    f"Image description from {chunk.document_name}. "
                    f"Image {image_id}. {description}"
                ),
                headings=list(chunk.headings) or ["Image description"],
                source_type="text",
            )
            repaired.append(
                IndexedChunk(
                    id=chunk.id,
                    text=contextualized_text,
                    raw_text=description,
                    source_type=chunk.source_type,
                    document_name=chunk.document_name,
                    embedding_text=contextualized_text,
                    headings=list(chunk.headings),
                    page_numbers=list(chunk.page_numbers),
                    metadata=dict(chunk.metadata),
                )
            )

        return repaired

    @staticmethod
    def _extract_image_description(chunk: IndexedChunk) -> str:
        text = normalize_whitespace(chunk.raw_text or chunk.text)
        if not text:
            return ""

        transcript_match = re.search(
            r"(?is)\bsystem\b.*?\buser\b.*?\bassistant\b[:\s]*(.*)$",
            text,
        )
        if transcript_match:
            return normalize_whitespace(transcript_match.group(1))

        assistant_match = re.search(r"(?is)\bassistant\b[:\s]*(.*)$", text)
        if assistant_match:
            return normalize_whitespace(assistant_match.group(1))

        return text

    def _build_indexes(
        self,
        chunks: list[IndexedChunk],
    ) -> tuple[object, object]:
        embeddings = self.embedding_model.encode(
            [chunk.embedding_text or chunk.text for chunk in chunks]
        ).astype(np.float32)
        faiss.normalize_L2(embeddings)
        return (
            IndexingPipeline._build_faiss(embeddings),
            IndexingPipeline._build_bm25(chunks),
        )

    def _rerank(
        self,
        query: str,
        chunks: list[IndexedChunk],
    ) -> list[IndexedChunk]:
        """
        Cross-encoder reranking.
        """

        if not chunks:
            return []

        documents = [chunk.embedding_text or chunk.text or chunk.raw_text for chunk in chunks]
        ranked_indices = self.reranker.rank(query, documents)

        safe_indices: list[int] = []
        seen: set[int] = set()
        for index in ranked_indices:
            if not isinstance(index, (int, np.integer)):
                continue
            if index < 0 or index >= len(chunks) or index in seen:
                continue
            safe_indices.append(int(index))
            seen.add(int(index))

        if not safe_indices:
            raise RuntimeError("Reranker returned no valid chunk indices.")

        return [chunks[index] for index in safe_indices[:FINAL_TOP_K]]
