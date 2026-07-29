"""
==================================================
Document Extraction
==================================================

This module converts a PDF into a DoclingDocument.

The returned DoclingDocument is the canonical
representation used throughout the project.

Responsibilities

• Load PDF
• Convert using Docling
• Return DoclingDocument

This module does NOT

• Save files
• Extract images
• Generate captions
• Create chunks
• Generate embeddings

==================================================
"""

# --------------------------------------------------
# Imports
# --------------------------------------------------

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DOCUMENT_MD, IMAGE_METADATA_FILE, SUPPORTED_EXTENSIONS
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import DoclingDocument

from extraction.serializer import DocumentSerializer
from utils.logger import get_logger

# --------------------------------------------------
# Logger
# --------------------------------------------------

logger = get_logger(__name__)


# ==================================================
# Document Extractor
# ==================================================


class DocumentExtractor:
    """
    Wrapper around Docling's DocumentConverter.
    """

    def __init__(self) -> None:
        logger.info("Initializing Docling DocumentConverter...")
        try:
            # Use a text-first PDF pipeline so indexing does not depend on OCR
            # auto-selection. The default Docling OCR path can fail on this
            # environment because RapidOCR may pick an unsupported backend.
            pdf_options = PdfPipelineOptions(
                do_ocr=False,
                do_table_structure=True,
                force_backend_text=True,
            )
            self.converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pdf_options,
                    )
                }
            )
        except Exception as exc:
            raise RuntimeError(
                f"Docling converter could not be initialized: {exc}"
            ) from exc

    # ==================================================
    # Extract Document
    # ==================================================

    def extract(
        self,
        pdf_path: Path,
    ):
        """
        Convert a PDF into a DoclingDocument.

        Parameters
        ----------
        pdf_path : Path
            Path to the input PDF.

        Returns
        -------
        DoclingDocument
        """

        self._validate_input(pdf_path)

        logger.info(f"Extracting document: {pdf_path.name}")

        try:
            page_count = self._get_page_count(pdf_path)
            converted_documents: list[DoclingDocument] = []

            for start_page, end_page in self._iter_page_batches(page_count):
                batch_document = self._convert_page_range(
                    pdf_path=pdf_path,
                    start_page=start_page,
                    end_page=end_page,
                )
                if batch_document is not None:
                    converted_documents.append(batch_document)

            if not converted_documents:
                raise RuntimeError(
                    f"Docling extraction produced no documents for {pdf_path.name}."
                )

            if len(converted_documents) == 1:
                document = converted_documents[0]
            else:
                document = DoclingDocument.concatenate(converted_documents)

            logger.info("Extraction completed successfully.")
            return document
        except Exception as exc:
            raise RuntimeError(
                f"Docling extraction failed for {pdf_path.name}: {exc}"
            ) from exc

    def _convert_page_range(
        self,
        *,
        pdf_path: Path,
        start_page: int,
        end_page: int,
    ) -> DoclingDocument | None:
        try:
            conversion_result = self.converter.convert(
                str(pdf_path),
                page_range=(start_page, end_page),
            )
            return conversion_result.document
        except Exception as exc:
            if start_page == end_page:
                raise RuntimeError(
                    f"Docling extraction failed on page {start_page}: {exc}"
                ) from exc

            midpoint = start_page + (end_page - start_page) // 2
            logger.warning(
                "Docling batch %s-%s failed: %s. Retrying as %s-%s and %s-%s.",
                start_page,
                end_page,
                exc,
                start_page,
                midpoint,
                midpoint + 1,
                end_page,
            )

            left = self._convert_page_range(
                pdf_path=pdf_path,
                start_page=start_page,
                end_page=midpoint,
            )
            right = self._convert_page_range(
                pdf_path=pdf_path,
                start_page=midpoint + 1,
                end_page=end_page,
            )

            batch_documents = [doc for doc in (left, right) if doc is not None]
            if not batch_documents:
                return None
            if len(batch_documents) == 1:
                return batch_documents[0]
            return DoclingDocument.concatenate(batch_documents)

    @staticmethod
    def _iter_page_batches(
        page_count: int,
        batch_size: int = 10,
    ):
        for start_page in range(1, page_count + 1, batch_size):
            end_page = min(page_count, start_page + batch_size - 1)
            yield start_page, end_page

    @staticmethod
    def _get_page_count(pdf_path: Path) -> int:
        try:
            import pypdfium2 as pdfium

            pdf_document = pdfium.PdfDocument(str(pdf_path))
            try:
                return len(pdf_document)
            finally:
                pdf_document.close()
        except ModuleNotFoundError:
            pass

        try:
            from pypdf import PdfReader

            return len(PdfReader(str(pdf_path)).pages)
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Unable to determine the PDF page count because neither "
                "pypdfium2 nor pypdf is available."
            ) from exc

    # ==================================================
    # Validation
    # ==================================================

    @staticmethod
    def _validate_input(
        pdf_path: Path,
    ) -> None:
        """
        Validate the input document.
        """

        if not pdf_path.exists():
            raise FileNotFoundError(
                f"File not found: {pdf_path}"
            )

        if pdf_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {pdf_path.suffix}"
            )

# ==================================================
# Standalone Test
# ==================================================

if __name__ == "__main__":

    from config import DOCUMENTS_DIR, DEFAULT_DOCUMENT_PATH
    from pipeline.vis_pipeline import VisionPipeline
    from scripts.document_enricher import MarkdownImageEnricher

    pdf_files = list(
        DOCUMENTS_DIR.glob("*.pdf")
    )

    if not pdf_files:
        raise FileNotFoundError(
            "No PDF files found inside the documents folder."
        )

    extractor = DocumentExtractor()
    pdf_path = DEFAULT_DOCUMENT_PATH

    document = extractor.extract(
        DEFAULT_DOCUMENT_PATH
    )

    serializer = DocumentSerializer()
    serializer.save_all(document)

    logger.info(
        f"Document title: {document.name}"
    )

    logger.info("Running vision pipeline...")
    VisionPipeline().process_document(
        document=document,
        pdf_path=pdf_path,
    )

    logger.info("Running document enricher...")
    MarkdownImageEnricher().enrich(
        markdown_path=DOCUMENT_MD,
        metadata_path=IMAGE_METADATA_FILE,
        output_path=Path("data/document_enriched.md"),
    )

    logger.info(
        "Extraction finished."
    )
