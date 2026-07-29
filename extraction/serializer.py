"""Serialize a Docling document into JSON, Markdown, and plain text."""

import json
from pathlib import Path

from config import DOCUMENT_JSON, DOCUMENT_MD, DOCUMENT_TXT
from utils.logger import get_logger

logger = get_logger(__name__)


class DocumentSerializer:
    """Serialize a Docling document into the supported artifact formats."""

    def save_all(self, document) -> None:
        self.save_json(document)
        self.save_markdown(document)
        self.save_text(document)
        logger.info("Document serialization completed.")

    def save_json(self, document, output_path: Path = DOCUMENT_JSON) -> None:
        logger.info("Saving JSON -> %s", output_path.name)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(document.export_to_dict(), handle, indent=4, ensure_ascii=False)

    def save_markdown(self, document, output_path: Path = DOCUMENT_MD) -> None:
        logger.info("Saving Markdown -> %s", output_path.name)
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(document.export_to_markdown())

    def save_text(self, document, output_path: Path = DOCUMENT_TXT) -> None:
        logger.info("Saving Text -> %s", output_path.name)
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(document.export_to_text())