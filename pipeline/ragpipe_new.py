"""
Main orchestration pipeline for extraction, indexing, retrieval, and generation.

This variant mirrors ``pipeline.rag_pipeline`` but is wired to the newer
indexing and vision pipelines.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from config import (
    ARTIFACT_SCHEMA_VERSION,
    CACHE_MANIFEST_FILE,
    DEFAULT_DOCUMENT_PATH,
    IMAGE_METADATA_FILE_NEW,
    BM25_TOP_K,
    FINAL_TOP_K,
    RERANK_TOP_K,
    VECTOR_TOP_K,
)
from extraction.extract_document import DocumentExtractor
from extraction.serializer import DocumentSerializer
from models.embedding import EmbeddingModel
from models.reranker import Reranker
from pipeline.generator import GenerationResult, GeneratorPipeline
from pipeline.indexing_new import IndexedChunk, IndexingPipeline
from pipeline.vis_pipeline import VisionPipeline
from utils.helpers import compute_file_sha256, file_exists, load_json, save_json
from utils.logger import get_logger


logger = get_logger(__name__)


@dataclass(slots=True)
class IngestionResult:
    document: str
    chunks: int
    images: int
    cached: bool


@dataclass(slots=True)
class CacheStatus:
    ready: bool
    document_name: str | None
    process_vision: bool | None
    manifest: dict[str, Any] | None


class HybridRetriever:
    """
    Hybrid dense + sparse retriever with reranking for the new artifacts.
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
        self.faiss_index = IndexingPipeline.load_faiss()
        self.bm25 = IndexingPipeline.load_bm25()

    def retrieve(self, query: str) -> list[IndexedChunk]:
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


class RAGPipeline:
    """
    High-level orchestrator for single-document local RAG.
    """

    def __init__(self) -> None:
        self.extractor = DocumentExtractor()
        self.serializer = DocumentSerializer()
        self.indexing_pipeline = IndexingPipeline()
        self.generator = GeneratorPipeline()
        self.retriever: HybridRetriever | None = None

    def _ensure_retriever(self) -> None:
        if self.retriever is None:
            self.retriever = HybridRetriever()

    def _ingest_and_prepare(
        self,
        *,
        pdf_path: str | Path | None = None,
        process_vision: bool = True,
        force_rebuild: bool = False,
    ) -> None:
        self.ingest(
            pdf_path=pdf_path,
            process_vision=process_vision,
            force_rebuild=force_rebuild,
        )
        self._ensure_retriever()

    def ingest(
        self,
        pdf_path: str | Path | None = None,
        *,
        process_vision: bool = True,
        force_rebuild: bool = False,
    ) -> IngestionResult:
        resolved_pdf = self.resolve_pdf_path(pdf_path)
        image_metadata = self._load_image_metadata()
        requested_image_metadata_hash = (
            self._image_metadata_hash() if image_metadata else None
        )

        manifest = self._build_manifest(
            resolved_pdf,
            process_vision,
            image_metadata_hash=requested_image_metadata_hash,
        )

        if not force_rebuild and self._is_cached(manifest):
            logger.info("Using cached index for %s", resolved_pdf.name)
            self._ensure_retriever()
            cached_chunks = IndexingPipeline.load_chunks()
            return IngestionResult(
                document=resolved_pdf.name,
                chunks=len(cached_chunks),
                images=self._cached_image_count(),
                cached=True,
            )

        logger.info("Ingesting document: %s", resolved_pdf.name)
        document = self.extractor.extract(resolved_pdf)
        self.serializer.save_all(document)

        if process_vision:
            document = VisionPipeline().process_document(
                document,
                pdf_path=resolved_pdf,
            )
            image_metadata = self._load_image_metadata()
        elif image_metadata is None:
            image_metadata = self._load_image_metadata()

        artifacts = self.indexing_pipeline.build(
            document=document,
            image_metadata=image_metadata,
            document_name=resolved_pdf.name,
        )
        image_count = len(image_metadata) if image_metadata else 0

        saved_manifest = self._build_manifest(
            resolved_pdf,
            process_vision,
            image_metadata_hash=self._image_metadata_hash() if image_metadata else None,
        )

        self._save_manifest(saved_manifest)

        return IngestionResult(
            document=resolved_pdf.name,
            chunks=len(artifacts["chunks"]),
            images=image_count,
            cached=False,
        )

    def retrieve(
        self,
        question: str,
        *,
        pdf_path: str | Path | None = None,
        process_vision: bool = True,
        force_rebuild: bool = False,
        auto_ingest: bool = True,
    ):
        self.ensure_ready(
            pdf_path=pdf_path,
            process_vision=process_vision,
            force_rebuild=force_rebuild,
            auto_ingest=auto_ingest,
        )
        assert self.retriever is not None
        return self.retriever.retrieve(question)

    def answer(
        self,
        question: str,
        *,
        pdf_path: str | Path | None = None,
        process_vision: bool = True,
        force_rebuild: bool = False,
        auto_ingest: bool = True,
    ) -> GenerationResult:
        chunks = self.retrieve(
            question,
            pdf_path=pdf_path,
            process_vision=process_vision,
            force_rebuild=force_rebuild,
            auto_ingest=auto_ingest,
        )
        return self.generator.generate(question, chunks)

    def stream_answer(
        self,
        question: str,
        *,
        pdf_path: str | Path | None = None,
        process_vision: bool = True,
        force_rebuild: bool = False,
        auto_ingest: bool = True,
    ):
        chunks = self.retrieve(
            question,
            pdf_path=pdf_path,
            process_vision=process_vision,
            force_rebuild=force_rebuild,
            auto_ingest=auto_ingest,
        )
        return self.generator.stream(question, chunks), list(chunks)

    def ensure_ready(
        self,
        *,
        pdf_path: str | Path | None = None,
        process_vision: bool = True,
        force_rebuild: bool = False,
        auto_ingest: bool = True,
    ) -> None:
        if force_rebuild:
            if not auto_ingest:
                raise RuntimeError(
                    "Rebuild was requested, but automatic ingestion is disabled."
                )
            self._ingest_and_prepare(
                pdf_path=pdf_path,
                process_vision=process_vision,
                force_rebuild=force_rebuild,
            )
            return

        if not IndexingPipeline.artifacts_exist():
            if not auto_ingest:
                raise RuntimeError(
                    "No cached index found. Run the setup/indexing step first."
                )
            self._ingest_and_prepare(
                pdf_path=pdf_path,
                process_vision=process_vision,
                force_rebuild=False,
            )
            return

        manifest = self.load_cached_manifest()
        if manifest is not None:
            if pdf_path is not None:
                requested_pdf = Path(pdf_path).expanduser().resolve()
            else:
                cached_path = manifest.get("document_path")
                requested_pdf = (
                    Path(cached_path).expanduser().resolve()
                    if cached_path
                    else self.resolve_pdf_path(None)
                )

            image_metadata_hash = None
            if not process_vision:
                image_metadata_hash = self._image_metadata_hash()

            requested_manifest = self._build_manifest(
                requested_pdf,
                process_vision,
                image_metadata_hash=image_metadata_hash,
            )
            requested_manifest=manifest
            if manifest != requested_manifest:
                if not auto_ingest:
                    cached_name = manifest.get("document_name") if manifest else None
                    cached_vision = manifest.get("process_vision") if manifest else None
                    raise RuntimeError(
                        "The cached index does not match the requested PDF. "
                        f"Cached document: {cached_name or 'unknown'}. "
                        f"Cached vision: {cached_vision or 'unknown'}. "
                        "Run setup/indexing for the requested PDF first."
                    )
                self._ingest_and_prepare(
                    pdf_path=requested_pdf,
                    process_vision=process_vision,
                    force_rebuild=False,
                )
                return

        self._ensure_retriever()

    def get_cache_status(self) -> CacheStatus:
        manifest = self.load_cached_manifest()
        return CacheStatus(
            ready=IndexingPipeline.artifacts_exist() and manifest is not None,
            document_name=manifest.get("document_name") if manifest else None,
            process_vision=manifest.get("process_vision") if manifest else None,
            manifest=manifest,
        )

    @staticmethod
    def load_cached_manifest() -> dict[str, Any] | None:
        if not file_exists(CACHE_MANIFEST_FILE):
            return None
        try:
            manifest = load_json(CACHE_MANIFEST_FILE)
        except Exception:
            return None
        return manifest if isinstance(manifest, dict) else None

    @staticmethod
    def resolve_pdf_path(
        pdf_path: str | Path | None,
    ) -> Path:
        if pdf_path is not None:
            resolved = Path(pdf_path).expanduser().resolve()
            if not resolved.exists():
                raise FileNotFoundError(f"PDF file not found: {resolved}")
            return resolved

        if DEFAULT_DOCUMENT_PATH.exists():
            return DEFAULT_DOCUMENT_PATH.resolve()

        raise FileNotFoundError(
            "Expected documents/sample.pdf, but it was not found."
        )

    @staticmethod
    def _build_manifest(
        pdf_path: Path,
        process_vision: bool,
        *,
        image_metadata_hash: str | None = None,
    ) -> dict[str, Any]:
        manifest = {
            "document_name": pdf_path.name,
            "document_path": str(pdf_path),
            "document_hash": compute_file_sha256(pdf_path),
            "process_vision": process_vision,
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        }
        if image_metadata_hash is not None:
            manifest["image_metadata_hash"] = image_metadata_hash
        return manifest

    @staticmethod
    def _save_manifest(
        manifest: dict[str, Any],
    ) -> None:
        save_json(manifest, CACHE_MANIFEST_FILE)

    @staticmethod
    def _is_cached(
        manifest: dict[str, Any],
    ) -> bool:
        if not IndexingPipeline.artifacts_exist():
            return False
        if not file_exists(CACHE_MANIFEST_FILE):
            return False
        cached_manifest = load_json(CACHE_MANIFEST_FILE)
        return cached_manifest == manifest

    @staticmethod
    def _cached_image_count() -> int:
        if not file_exists(IMAGE_METADATA_FILE_NEW):
            return 0
        try:
            metadata = load_json(IMAGE_METADATA_FILE_NEW)
        except Exception:
            return 0
        return len(metadata) if isinstance(metadata, list) else 0

    @staticmethod
    def _image_metadata_hash() -> str | None:
        if not file_exists(IMAGE_METADATA_FILE_NEW):
            return None
        return compute_file_sha256(IMAGE_METADATA_FILE_NEW)

    @staticmethod
    def _load_image_metadata() -> list[dict[str, Any]] | None:
        if not file_exists(IMAGE_METADATA_FILE_NEW):
            return None
        try:
            metadata = load_json(IMAGE_METADATA_FILE_NEW)
        except Exception:
            return None
        if not isinstance(metadata, list):
            return None
        return [item for item in metadata if isinstance(item, dict)]
