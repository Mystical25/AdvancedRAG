"""
==================================================
Indexing Pipeline
==================================================

Responsibilities
----------------
1. Chunk the DoclingDocument using HybridChunker, with a custom
   picture serializer so that PictureDescriptionData annotations
   (written by VisionPipeline) are pulled into chunk text in
   reading order, alongside the surrounding narrative/tables.
2. Split chunk text with tokenizer-aware windows and configurable overlap.
3. Generate embeddings
4. Build FAISS index
5. Build BM25 index
6. Save all artifacts

This pipeline no longer builds a separate stream of
image-only chunks from image_metadata.json. Image
descriptions now enter the index exclusively through the
document itself (as picture annotations), chunked by
HybridChunker in the same pass as everything else.

Contextualized chunking is still available as an opt-in flag on
IndexingPipeline, but it is off by default because it was inflating
chunks and duplicating headings/context.

==================================================
"""

from __future__ import annotations

import pickle
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from pydantic import PrivateAttr
from typing_extensions import override

from config import (
    BM25_INDEX_FILE,
    CHUNKS_DIR,
    CHUNKS_NEW_JSON,
    CHUNK_OVERLAP,
    INDEXING_USE_CONTEXTUALIZE,
    EMBEDDING_MODEL,
    EMBEDDING_SAFE_MAX_TOKENS,
    EMBEDDINGS_FILE,
    FAISS_INDEX_FILE,
    MAX_CONSECUTIVE_PICTURE_DESCRIPTIONS,
    MAX_CHUNK_SIZE,
    MIN_TRAILING_CHUNK_TOKENS,
)
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.serializer.base import BaseDocSerializer, SerializationResult
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.markdown import MarkdownPictureSerializer
from docling_core.types.doc.document import (
    DoclingDocument,
    PictureDescriptionData,
    PictureItem,
    TableItem,
)
from models.embedding import EmbeddingModel
from utils.helpers import load_json, normalize_whitespace, save_json, save_numpy
from utils.logger import get_logger


logger = get_logger(__name__)


# ==================================================
# Picture serializer: pulls annotation text into chunk output
# ==================================================


class DescriptionPictureSerializer(MarkdownPictureSerializer):
    """
    Extends Docling's default markdown picture serializer to also
    emit the text of any PictureDescriptionData annotation attached
    to the picture (written earlier by VisionPipeline). Without this,
    HybridChunker's default serialization ignores annotations and a
    picture contributes little more than a placeholder to chunk text.
    """

    _last_picture_run_key: tuple[Any, ...] | None = PrivateAttr(default=None)
    _picture_run_count: int = PrivateAttr(default=0)

    @override
    def serialize(
        self,
        *,
        item: PictureItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        separator: Optional[str] = None,
        **kwargs: Any,
    ) -> SerializationResult:
        parent_res = super().serialize(
            item=item,
            doc_serializer=doc_serializer,
            doc=doc,
            separator=separator,
            **kwargs,
        )

        description = self._extract_description(item)
        parent_text = normalize_whitespace(parent_res.text).casefold()
        description_text = normalize_whitespace(description).casefold()

        if self._picture_run_key(item) != self._last_picture_run_key:
            self._last_picture_run_key = self._picture_run_key(item)
            self._picture_run_count = 0

        should_append = bool(description_text) and description_text not in parent_text
        if should_append and self._picture_run_count < MAX_CONSECUTIVE_PICTURE_DESCRIPTIONS:
            text_parts: list[str] = [parent_res.text] if parent_res.text else []
            text_parts.append(f"Image description: {normalize_whitespace(description)}")
            self._picture_run_count += 1

            return create_ser_result(
                text="\n".join(part for part in text_parts if part),
                span_source=[parent_res],
            )

        return create_ser_result(
            text=parent_res.text,
            span_source=[parent_res],
        )

    @staticmethod
    def _extract_description(item: PictureItem) -> str:
        for annotation in getattr(item, "annotations", None) or []:
            if isinstance(annotation, PictureDescriptionData) and annotation.text:
                return normalize_whitespace(annotation.text)
        return ""

    @staticmethod
    def _picture_run_key(item: PictureItem) -> tuple[Any, ...]:
        page_numbers = [
            int(getattr(prov, "page_no"))
            for prov in getattr(item, "prov", []) or []
            if isinstance(getattr(prov, "page_no", None), int)
        ]
        if page_numbers:
            return tuple(sorted(page_numbers))
        return (getattr(item, "self_ref", ""),)


class ImageAwareSerializerProvider(ChunkingSerializerProvider):
    """
    Wires DescriptionPictureSerializer into the serializer that
    HybridChunker uses when it walks the document.

    Note: if `picture_serializer` isn't accepted by ChunkingDocSerializer
    in your installed docling-core version, run
    `help(ChunkingDocSerializer.__init__)` to check the current
    constructor signature -- this is a fast-moving part of the API.
    """

    def get_serializer(self, doc: DoclingDocument) -> ChunkingDocSerializer:
        return ChunkingDocSerializer(
            doc=doc,
            picture_serializer=DescriptionPictureSerializer(),
        )


@dataclass(slots=True)
class IndexedChunk:
    """
    Minimal retrieval payload stored on disk.
    """

    id: str
    text: str
    raw_text: str
    source_type: str
    document_name: str
    embedding_text: str | None = None
    headings: list[str] = field(default_factory=list)
    page_numbers: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def citation(self) -> str:
        if not self.page_numbers:
            return "Page unavailable"
        if len(self.page_numbers) == 1:
            return f"Page {self.page_numbers[0]}"
        pages = ", ".join(str(page) for page in self.page_numbers)
        return f"Pages {pages}"


class IndexingPipeline:
    """
    Build and persist retrieval artifacts for a document.
    """

    def __init__(
        self,
        *,
        use_contextualize: bool = INDEXING_USE_CONTEXTUALIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        min_trailing_tokens: int = MIN_TRAILING_CHUNK_TOKENS,
    ) -> None:
        logger.info("Initializing indexing pipeline...")
        self.embedding_model = EmbeddingModel()
        self.chunker = HybridChunker(
            tokenizer=EMBEDDING_MODEL,
            max_tokens=MAX_CHUNK_SIZE,
            serializer_provider=ImageAwareSerializerProvider(),
        )
        self.use_contextualize = use_contextualize
        self.chunk_overlap = max(0, chunk_overlap)
        self.min_trailing_tokens = max(0, min_trailing_tokens)

    def build(
        self,
        document,
        image_metadata: list[dict[str, Any]] | None = None,
        document_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Build every index required by the RAG system.

        `document` must already have picture annotations attached
        (i.e. VisionPipeline.process_document has run on it, or it
        was reloaded from a JSON export taken after that step).
        Text, tables, and image descriptions are all chunked in a
        single HybridChunker pass over `document` -- there is no
        separate image-chunk stream anymore.
        """

        resolved_name = document_name or getattr(document, "name", "document")

        logger.info("Chunking document...")
        indexed_chunks = self._chunk_document(document=document, document_name=resolved_name)
        indexed_chunks.extend(
            self._build_image_chunks(
                image_metadata=image_metadata or [],
                document_name=resolved_name,
                start_index=len(indexed_chunks),
            )
        )
        indexed_chunks = self._enforce_embedding_limit(indexed_chunks)
        indexed_chunks = self._reindex_chunks(indexed_chunks)

        if not indexed_chunks:
            raise ValueError("No chunks were generated for indexing.")

        logger.info("Generated %s chunks.", len(indexed_chunks))

        embeddings = self._generate_embeddings(indexed_chunks)
        faiss_index = self._build_faiss(embeddings)
        bm25 = self._build_bm25(indexed_chunks)

        self._save_chunks(indexed_chunks)
        self._save_embeddings(embeddings)
        self._save_faiss(faiss_index)
        self._save_bm25(bm25)

        logger.info("Indexing complete.")

        return {
            "chunks": indexed_chunks,
            "embeddings": embeddings,
            "faiss": faiss_index,
            "bm25": bm25,
        }

    def _chunk_document(
        self,
        document,
        document_name: str,
    ) -> list[IndexedChunk]:
        chunks: list[IndexedChunk] = []

        for index, chunk in enumerate(self.chunker.chunk(document)):
            raw_source = str(chunk.text or "")
            body_source = raw_source.strip()
            headings = self._extract_headings(chunk)
            source_type = self._classify_source_type(chunk, raw_source)

            if self.use_contextualize:
                body_source = str(self.chunker.contextualize(chunk) or "").strip()
                headings = []

            if not normalize_whitespace(body_source):
                continue

            heading_prefix = self._build_heading_prefix(headings=headings, source_type=source_type)
            body_budget = self._body_token_budget(heading_prefix=heading_prefix)
            split_texts = self._split_text_for_embedding(
                body_source,
                max_tokens=body_budget,
                overlap=self.chunk_overlap,
                min_trailing_tokens=self.min_trailing_tokens,
            )

            for split_index, split_text in enumerate(split_texts):
                split_body = split_text.strip() if source_type == "table" else normalize_whitespace(split_text)
                if not split_body:
                    continue

                chunk_text = self._build_chunk_text(
                    body=split_body,
                    headings=headings,
                    source_type=source_type,
                )
                chunk_text = self._ensure_token_limit(chunk_text, max_tokens=EMBEDDING_SAFE_MAX_TOKENS)
                if not chunk_text:
                    continue

                chunks.append(
                    IndexedChunk(
                        id=f"text-{index:04d}-{split_index:02d}",
                        text=chunk_text,
                        raw_text=split_body,
                        source_type=source_type,
                        document_name=document_name,
                        embedding_text=chunk_text,
                        headings=headings,
                        page_numbers=self._extract_page_numbers(chunk),
                        metadata={
                            "origin": getattr(
                                getattr(chunk.meta, "origin", None),
                                "filename",
                                document_name,
                            ),
                            "split_index": split_index,
                            "content_type": source_type,
                        },
                    )
                )

        return chunks

    @staticmethod
    def _build_image_chunks(
        image_metadata: list[dict[str, Any]],
        document_name: str,
        start_index: int,
    ) -> list[IndexedChunk]:
        image_chunks: list[IndexedChunk] = []

        for offset, item in enumerate(image_metadata):
            description = normalize_whitespace(item.get("description", ""))
            if not description:
                continue

            image_id = item.get("image_id", offset)
            page_numbers = sorted(
                {
                    int(page)
                    for page in item.get("page_numbers", [])
                    if str(page).isdigit()
                }
            )

            contextualized_text = IndexingPipeline._build_chunk_text(
                body=(
                    f"Image description from {document_name}. "
                    f"Image {image_id}. {description}"
                ),
                headings=["Image description"],
                source_type="text",
            )

            image_chunks.append(
                IndexedChunk(
                    id=f"image-{start_index + offset:04d}",
                    text=contextualized_text,
                    raw_text=description,
                    source_type="image",
                    document_name=document_name,
                    embedding_text=contextualized_text,
                    headings=["Image description"],
                    page_numbers=page_numbers,
                    metadata={
                        "image_id": image_id,
                        "image_path": item.get("image_path"),
                    },
                )
            )

        return image_chunks

    @staticmethod
    def _classify_source_type(chunk, raw_source: str) -> str:
        """
        Inspect the doc items backing this chunk to tag it as
        "table", "image" (picture-only), "text_with_image" (picture
        interleaved with narrative -- the common case now that
        descriptions flow through the document), or plain "text".
        """

        doc_items = list(getattr(chunk.meta, "doc_items", None) or [])

        has_table = (
            any(isinstance(item, TableItem) for item in doc_items)
            or "|" in raw_source
            or "---" in raw_source
        )
        if has_table:
            return "table"

        has_picture = any(isinstance(item, PictureItem) for item in doc_items)
        if not has_picture:
            return "text"

        has_other_text = any(
            not isinstance(item, (TableItem, PictureItem)) for item in doc_items
        )
        return "text_with_image" if has_other_text else "image"

    def _generate_embeddings(
        self,
        chunks: list[IndexedChunk],
    ) -> np.ndarray:
        logger.info("Generating embeddings...")
        texts = [chunk.embedding_text or chunk.text for chunk in chunks]
        embeddings = self.embedding_model.encode(texts).astype(np.float32)
        faiss.normalize_L2(embeddings)
        return embeddings

    def _enforce_embedding_limit(
        self,
        chunks: list[IndexedChunk],
    ) -> list[IndexedChunk]:
        safe_chunks: list[IndexedChunk] = []
        for chunk in chunks:
            safe_chunks.extend(
                self._split_chunk_if_needed(
                    chunk=chunk,
                    max_tokens=EMBEDDING_SAFE_MAX_TOKENS,
                )
            )
        return safe_chunks

    def _split_chunk_if_needed(
        self,
        *,
        chunk: IndexedChunk,
        max_tokens: int,
    ) -> list[IndexedChunk]:
        text = normalize_whitespace(chunk.text)
        raw_text = normalize_whitespace(chunk.raw_text or "")
        if not text:
            return []

        if self._token_count(text) <= max_tokens:
            return [
                IndexedChunk(
                    id=chunk.id,
                    text=chunk.text,
                    raw_text=chunk.raw_text,
                    source_type=chunk.source_type,
                    document_name=chunk.document_name,
                    embedding_text=self._ensure_token_limit(
                        chunk.embedding_text or chunk.text,
                        max_tokens=max_tokens,
                    ),
                    headings=list(chunk.headings),
                    page_numbers=list(chunk.page_numbers),
                    metadata=dict(chunk.metadata),
                )
            ]

        split_texts = self._split_text_for_embedding(
            text,
            max_tokens=max_tokens,
            overlap=self.chunk_overlap,
            min_trailing_tokens=self.min_trailing_tokens,
        )
        if not split_texts:
            return [chunk]

        raw_split_texts = self._split_text_for_embedding(
            raw_text or text,
            max_tokens=max_tokens,
            overlap=self.chunk_overlap,
            min_trailing_tokens=self.min_trailing_tokens,
        )

        expanded: list[IndexedChunk] = []
        for split_index, split_text in enumerate(split_texts):
            normalized_split_text = normalize_whitespace(split_text)
            if not normalized_split_text:
                continue

            split_raw_text = (
                normalize_whitespace(raw_split_texts[split_index])
                if split_index < len(raw_split_texts)
                else normalized_split_text
            )

            expanded.append(
                IndexedChunk(
                    id=f"{chunk.id}-s{split_index:02d}",
                    text=normalized_split_text,
                    raw_text=split_raw_text,
                    source_type=chunk.source_type,
                    document_name=chunk.document_name,
                    embedding_text=self._ensure_token_limit(
                        normalized_split_text,
                        max_tokens=max_tokens,
                    ),
                    headings=list(chunk.headings),
                    page_numbers=list(chunk.page_numbers),
                    metadata={
                        **chunk.metadata,
                        "split_index": split_index,
                        "split_from": chunk.id,
                    },
                )
            )

        return expanded or [chunk]

    def _split_text_for_embedding(
        self,
        text: str,
        *,
        max_tokens: int = EMBEDDING_SAFE_MAX_TOKENS,
        overlap: int | None = None,
        min_trailing_tokens: int | None = None,
    ) -> list[str]:
        normalized = text.strip()
        if not normalize_whitespace(normalized):
            return []

        token_ids = self._token_ids(normalized)
        if not token_ids:
            return []

        if len(token_ids) <= max_tokens:
            return [self._decode_token_ids(token_ids)]

        token_overlap = self.chunk_overlap if overlap is None else max(0, overlap)
        trailing_min = (
            self.min_trailing_tokens
            if min_trailing_tokens is None
            else max(0, min_trailing_tokens)
        )

        segments: list[str] = []
        total = len(token_ids)
        start = 0
        step = max(1, max_tokens - token_overlap)

        while start < total:
            remaining = total - start
            if remaining <= max_tokens:
                end = total
            else:
                end = min(start + max_tokens, total)
                remainder = total - end
                if 0 < remainder < trailing_min:
                    adjusted_end = total - trailing_min
                    if adjusted_end > start:
                        end = adjusted_end

            if end <= start:
                end = min(total, start + max_tokens)

            segment_ids = token_ids[start:end]
            if not segment_ids:
                break

            segment_text = self._decode_token_ids(segment_ids)
            if segment_text:
                segments.append(segment_text)

            if end >= total:
                break

            next_start = max(0, end - token_overlap)
            if next_start <= start:
                next_start = min(total, start + step)
            start = next_start

        return self._merge_tiny_trailing_chunks(
            segments,
            max_tokens=max_tokens,
            min_trailing_tokens=trailing_min,
        )

    @staticmethod
    def _build_heading_prefix(
        *,
        headings: list[str],
        source_type: str,
    ) -> str:
        normalized_headings = [
            normalize_whitespace(heading)
            for heading in headings
            if normalize_whitespace(heading)
        ]
        if not normalized_headings:
            return ""

        heading_text = " | ".join(normalized_headings)
        if source_type == "table":
            return f"{heading_text}\n"

        return f"{heading_text}. "

    @staticmethod
    def _build_chunk_text(
        *,
        body: str,
        headings: list[str],
        source_type: str,
    ) -> str:
        normalized_body = body.strip() if source_type == "table" else normalize_whitespace(body)
        heading_prefix = IndexingPipeline._build_heading_prefix(
            headings=headings,
            source_type=source_type,
        )
        if not heading_prefix:
            return normalized_body
        if source_type == "table":
            return f"{heading_prefix}{normalized_body}".strip()
        return normalize_whitespace(f"{heading_prefix}{normalized_body}")

    def _body_token_budget(
        self,
        *,
        heading_prefix: str,
        max_tokens: int = EMBEDDING_SAFE_MAX_TOKENS,
    ) -> int:
        heading_tokens = self._token_count(heading_prefix) if heading_prefix else 0
        budget = max_tokens - heading_tokens - 8
        return max(1, budget)

    def _merge_tiny_trailing_chunks(
        self,
        segments: list[str],
        *,
        max_tokens: int,
        min_trailing_tokens: int,
    ) -> list[str]:
        if len(segments) < 2:
            return [segment for segment in segments if normalize_whitespace(segment)]

        trailing = normalize_whitespace(segments[-1])
        if not trailing or self._token_count(trailing) >= min_trailing_tokens:
            return [segment for segment in segments if normalize_whitespace(segment)]

        merged_tail = normalize_whitespace(f"{segments[-2]} {trailing}")
        if self._token_count(merged_tail) <= max_tokens:
            segments = [*segments[:-2], merged_tail]

        return [segment for segment in segments if normalize_whitespace(segment)]

    def _token_ids(self, text: str) -> list[str]:
        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            return re.findall(r"\S+", normalize_whitespace(text))

        try:
            tokens = tokenizer.tokenize(text)
            return list(tokens)
        except Exception:
            return re.findall(r"\S+", normalize_whitespace(text))

    def _decode_token_ids(self, token_ids: list[str]) -> str:
        tokenizer = self._get_tokenizer()
        if tokenizer is not None:
            try:
                if hasattr(tokenizer, "convert_tokens_to_string"):
                    return tokenizer.convert_tokens_to_string(token_ids).strip()
            except Exception:
                pass

        return normalize_whitespace(" ".join(str(token) for token in token_ids))

    def _get_tokenizer(self):
        model = getattr(self.embedding_model, "model", None)
        return getattr(model, "tokenizer", None)

    def _token_count(self, text: str) -> int:
        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            return len(re.findall(r"\S+", normalize_whitespace(text)))

        try:
            return len(tokenizer.tokenize(text))
        except Exception:
            return len(re.findall(r"\S+", normalize_whitespace(text)))

    def _truncate_to_tokens(
        self,
        text: str,
        *,
        max_tokens: int,
    ) -> str:
        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            return normalize_whitespace(" ".join(re.findall(r"\S+", normalize_whitespace(text))[:max_tokens]))

        try:
            tokens = tokenizer.tokenize(text)
            input_tokens = tokens[:max_tokens]
            if not input_tokens:
                return normalize_whitespace(" ".join(re.findall(r"\S+", normalize_whitespace(text))[:max_tokens]))

            if hasattr(tokenizer, "convert_tokens_to_string"):
                return tokenizer.convert_tokens_to_string(input_tokens).strip()
        except Exception:
            pass

        return normalize_whitespace(" ".join(re.findall(r"\S+", normalize_whitespace(text))[:max_tokens]))

    def _ensure_token_limit(
        self,
        text: str,
        *,
        max_tokens: int,
    ) -> str:
        normalized = normalize_whitespace(text)
        if not normalized:
            return ""
        if self._token_count(normalized) <= max_tokens:
            return normalized
        return self._truncate_to_tokens(normalized, max_tokens=max_tokens)

    def _reindex_chunks(
        self,
        chunks: list[IndexedChunk],
    ) -> list[IndexedChunk]:
        reindexed: list[IndexedChunk] = []
        for index, chunk in enumerate(chunks):
            reindexed.append(
                IndexedChunk(
                    id=f"chunk-{index:05d}",
                    text=chunk.text,
                    raw_text=chunk.raw_text,
                    source_type=chunk.source_type,
                    document_name=chunk.document_name,
                    embedding_text=chunk.embedding_text,
                    headings=list(chunk.headings),
                    page_numbers=list(chunk.page_numbers),
                    metadata=dict(chunk.metadata),
                )
            )
        return reindexed

    @staticmethod
    def _build_faiss(embeddings: np.ndarray):
        logger.info("Building FAISS index...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        return index

    @staticmethod
    def _build_bm25(
        chunks: list[IndexedChunk],
    ) -> BM25Okapi:
        logger.info("Building BM25 index...")
        corpus = [IndexingPipeline.tokenize_text(chunk.text) for chunk in chunks]
        return BM25Okapi(corpus)

    @staticmethod
    def tokenize_text(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", normalize_whitespace(text).lower())

    @staticmethod
    def _extract_headings(chunk) -> list[str]:
        headings = getattr(chunk.meta, "headings", None) or []
        return [normalize_whitespace(heading) for heading in headings if heading]

    @staticmethod
    def _extract_page_numbers(chunk) -> list[int]:
        page_numbers: set[int] = set()

        for doc_item in getattr(chunk.meta, "doc_items", []):
            for prov in getattr(doc_item, "prov", []):
                page_no = getattr(prov, "page_no", None)
                if isinstance(page_no, int):
                    page_numbers.add(page_no)

        return sorted(page_numbers)

    @staticmethod
    def _save_chunks(chunks: list[IndexedChunk]) -> None:
        save_json([chunk.to_dict() for chunk in chunks], CHUNKS_NEW_JSON)

    @staticmethod
    def _save_embeddings(embeddings: np.ndarray) -> None:
        save_numpy(embeddings, EMBEDDINGS_FILE)

    @staticmethod
    def _save_faiss(index) -> None:
        faiss.write_index(index, str(FAISS_INDEX_FILE))

    @staticmethod
    def _save_bm25(bm25: BM25Okapi) -> None:
        with open(BM25_INDEX_FILE, "wb") as file:
            pickle.dump(bm25, file)

    @staticmethod
    def artifacts_exist() -> bool:
        required_files: Iterable = (
            CHUNKS_NEW_JSON,
            EMBEDDINGS_FILE,
            FAISS_INDEX_FILE,
            BM25_INDEX_FILE,
        )
        return all(path.exists() for path in required_files)

    @staticmethod
    def load_chunks() -> list[IndexedChunk]:
        records = load_json(CHUNKS_NEW_JSON)
        return [IndexedChunk(**record) for record in records]

    @staticmethod
    def load_faiss():
        return faiss.read_index(str(FAISS_INDEX_FILE))

    @staticmethod
    def load_bm25():
        with open(BM25_INDEX_FILE, "rb") as file:
            return pickle.load(file)
