"""
==================================================
Vision Pipeline
==================================================

Processes every figure contained in a DoclingDocument.

Responsibilities

- Extract images
- Save images to disk (for UI/debug purposes only)
- Generate captions using Qwen2.5-VL
- Write each caption back onto the DoclingDocument as a
  PictureDescriptionData annotation on the corresponding
  PictureItem
- Save image metadata to disk as a secondary, human-readable
  artifact (thumbnails, debugging, citations UI)

The DoclingDocument IS modified by this pipeline: every
picture that receives a description (VLM-generated or
fallback) gets that description appended to its
`.annotations` list. This is what allows HybridChunker to
pick up the description later, in reading order, without any
separate image-chunk bookkeeping downstream.

If your orchestration runs extraction / vision / indexing as
separate process invocations (rather than passing the same
in-memory `document` object through), you must persist and
reload the document via `DoclingDocument.save_as_json()` /
`DoclingDocument.load_from_json()` between stages so the
annotations survive the round trip. A plain
`export_to_dict()` + `json.load()` round trip is not
guaranteed to reconstruct the annotation objects with their
correct types.

==================================================
"""

import json
import re
from pathlib import Path
from typing import Any

from config import (
    IMAGE_METADATA_FILE,
    IMAGE_METADATA_FILE_NEW,
    IMAGES_DIR,
    MIN_IMAGE_DIMENSION_FOR_DESCRIPTION,
)
from docling_core.types.doc.document import PictureDescriptionData
from utils.helpers import normalize_whitespace
from utils.logger import get_logger


logger = get_logger(__name__)

# Provenance tags recorded on each annotation so you can later tell,
# per-image, whether the description came from the vision model or
# from a fallback (useful when auditing retrieval quality).
VLM_PROVENANCE = "vision_pipeline:qwen2.5-vl"
FALLBACK_PROVENANCE = "vision_pipeline:fallback-caption"


class VisionPipeline:

    def __init__(self):
        from models.vision import VisionModel

        self.vision_model = VisionModel()

    def process_document(
        self,
        document,
        pdf_path: Path | None = None,
    ):
        """
        Process every image in a DoclingDocument.

        Mutates `document` in place: every PictureItem that gets a
        description (VLM-generated or fallback) has a
        PictureDescriptionData annotation appended to it.

        Returns the same `document` object (now annotated), so callers
        can pass it straight into the indexing pipeline. The
        per-image metadata list is still written to disk
        (IMAGE_METADATA_FILE) as a secondary artifact for UI/debug
        use, but indexing no longer depends on it.
        """

        image_metadata: list[dict[str, Any]] = []
        pictures = list(document.pictures)

        logger.info("Found %s image(s).", len(pictures))

        for index, picture in enumerate(pictures):
            metadata = self._process_picture(
                document=document,
                picture=picture,
                index=index,
                pdf_path=pdf_path,
            )

            if metadata is not None:
                image_metadata.append(metadata)

        self._save_metadata(image_metadata)

        logger.info(
            "Vision processing completed. %s picture(s) annotated on the document.",
            len(image_metadata),
        )

        return document

    def _process_picture(
        self,
        document,
        picture,
        index: int,
        pdf_path: Path | None = None,
    ) -> dict[str, Any] | None:
        image_path = IMAGES_DIR / f"image_{index:04d}.png"
        page_numbers = self._extract_page_numbers(picture)

        image = self._extract_picture_image(
            document=document,
            picture=picture,
            pdf_path=pdf_path,
        )

        if image is None:
            logger.warning(
                "No raster image was available for picture %d; using fallback description only.",
                index,
            )
            fallback_description = self._build_fallback_description(
                picture=picture,
                index=index,
                page_numbers=page_numbers,
            )
            self._annotate_picture(
                picture=picture,
                description=fallback_description,
                provenance=FALLBACK_PROVENANCE,
            )
            return {
                "image_id": index,
                "image_path": None,
                "description": fallback_description,
                "page_numbers": page_numbers,
            }

        if self._is_too_small(image):
            logger.info(
                "Skipping picture %d because the extracted image is too small (%sx%s).",
                index,
                image.width,
                image.height,
            )
            return None

        image.save(image_path)
        logger.info("Saved %s", image_path.name)

        provenance = VLM_PROVENANCE
        try:
            description = self.vision_model.describe_image(str(image_path))
        except Exception as exc:
            logger.warning(
                "Qwen image description failed for %s: %s. Falling back to caption text.",
                image_path.name,
                exc,
            )
            description = ""

        description = self._sanitize_description(description)
        if not description:
            description = self._build_fallback_description(
                picture=picture,
                index=index,
                page_numbers=page_numbers,
            )
            provenance = FALLBACK_PROVENANCE

        self._annotate_picture(
            picture=picture,
            description=description,
            provenance=provenance,
        )

        return {
            "image_id": index,
            "image_path": str(image_path),
            "description": description,
            "page_numbers": page_numbers,
        }

    # ==================================================
    # Document annotation
    # ==================================================

    @staticmethod
    def _annotate_picture(
        picture,
        description: str,
        provenance: str,
    ) -> None:
        """
        Write the description onto the PictureItem itself so it
        travels with the document (and therefore through
        HybridChunker) instead of living only in a side JSON file.
        """

        description = normalize_whitespace(description)
        if not description:
            return

        try:
            picture.annotations.append(
                PictureDescriptionData(
                    kind="description",
                    text=description,
                    provenance=provenance,
                )
            )
        except Exception as exc:
            # Don't let an annotation-schema mismatch (e.g. a docling-core
            # version drift) take down the whole vision pass; log loudly
            # so it's easy to notice and fix.
            logger.error(
                "Failed to attach PictureDescriptionData to picture %s: %s",
                getattr(picture, "self_ref", "unknown"),
                exc,
            )

    @staticmethod
    def _sanitize_description(description: str) -> str:
        """
        Normalize whitespace and strip any echoed chat transcript.
        """

        text = normalize_whitespace(description)
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

    @staticmethod
    def _is_too_small(image) -> bool:
        """
        Skip images that are too small to be useful for retrieval.
        """

        try:
            width, height = image.size
        except Exception:
            return False

        return (
            width < MIN_IMAGE_DIMENSION_FOR_DESCRIPTION
            or height < MIN_IMAGE_DIMENSION_FOR_DESCRIPTION
        )

    def _extract_picture_image(
        self,
        document,
        picture,
        pdf_path: Path | None,
    ):
        for prov_index, _ in enumerate(getattr(picture, "prov", []) or []):
            try:
                image = picture.get_image(document, prov_index=prov_index)
            except Exception as exc:
                logger.debug(
                    "Docling image extraction failed for picture %s provenance %s: %s",
                    getattr(picture, "self_ref", "unknown"),
                    prov_index,
                    exc,
                )
                image = None

            if image is not None:
                return image

        if pdf_path is None:
            return None

        return self._crop_picture_from_pdf(pdf_path, picture)

    @staticmethod
    def _crop_picture_from_pdf(
        pdf_path: Path,
        picture,
    ):
        try:
            import pypdfium2 as pdfium
        except ModuleNotFoundError:
            return None

        pdf_document = pdfium.PdfDocument(str(pdf_path))
        try:
            for prov in getattr(picture, "prov", []) or []:
                page_no = getattr(prov, "page_no", None)
                bbox = getattr(prov, "bbox", None)
                if not isinstance(page_no, int) or bbox is None:
                    continue

                page_index = page_no - 1
                if page_index < 0 or page_index >= len(pdf_document):
                    continue

                page = pdf_document[page_index]
                page_width, page_height = page.get_size()

                left = max(0.0, float(getattr(bbox, "l", 0.0)))
                bottom = max(0.0, float(getattr(bbox, "b", 0.0)))
                right = max(0.0, float(page_width - getattr(bbox, "r", page_width)))
                top = max(0.0, float(page_height - getattr(bbox, "t", page_height)))

                bitmap = page.render(
                    scale=2.0,
                    crop=(left, bottom, right, top),
                )
                return bitmap.to_pil()
        finally:
            pdf_document.close()

        return None

    @staticmethod
    def _build_fallback_description(
        picture,
        index: int,
        page_numbers: list[int],
    ) -> str:
        caption = VisionPipeline._resolve_caption_text(picture)
        caption = normalize_whitespace(caption)
        if caption:
            return caption

        if page_numbers:
            pages = ", ".join(str(page) for page in page_numbers)
            return f"Figure {index + 1} extracted from page(s) {pages}."

        return f"Figure {index + 1} extracted from the document."

    @staticmethod
    def _resolve_caption_text(picture) -> str:
        caption_text = getattr(picture, "caption_text", "")

        if callable(caption_text):
            try:
                caption_text = caption_text()
            except TypeError:
                caption_text = ""
            except Exception:
                caption_text = ""

        if caption_text is None:
            return ""

        return str(caption_text)

    @staticmethod
    def _extract_page_numbers(
        picture,
    ) -> list[int]:
        page_numbers = sorted(
            {
                int(getattr(prov, "page_no"))
                for prov in getattr(picture, "prov", [])
                if isinstance(getattr(prov, "page_no", None), int)
            }
        )
        return page_numbers

    def _save_metadata(
        self,
        metadata: list[dict],
    ) -> None:
        """
        Secondary artifact only (UI thumbnails, debugging, audits).
        Indexing no longer reads this file.
        """

        for metadata_path in (IMAGE_METADATA_FILE, IMAGE_METADATA_FILE_NEW):
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with open(
                metadata_path,
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    metadata,
                    file,
                    indent=4,
                    ensure_ascii=False,
                )