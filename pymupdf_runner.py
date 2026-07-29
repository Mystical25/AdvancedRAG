from extraction.pymupdf_extractor import MarkdownExtractor
from pipeline.pymupdf_vision import VisionPipeline
from config import (
    PYMUDF_PDF_PATH,
    PYMUDF_OUTPUT_DIR,
    PYMUDF_ENRICHED_MARKDOWN,
    PYMUDF_IMAGE_DIR,
)

# All file and directory names for the PyMuPDF workflow are configured
# centrally in config.py. Do not hardcode paths in this runner.


def main():

    print("Extracting PDF...")

    extractor = MarkdownExtractor(
        pdf_path=PYMUDF_PDF_PATH,
        output_dir=PYMUDF_OUTPUT_DIR,
    )

    extractor.extract()

    print("Running Vision Pipeline...")

    pipeline = VisionPipeline()

    pipeline.process_document(
        markdown_path=PYMUDF_ENRICHED_MARKDOWN,
        image_dir=PYMUDF_IMAGE_DIR,
    )

    print("Done!")
    print(f"Enriched markdown saved to {PYMUDF_ENRICHED_MARKDOWN}")


if __name__ == "__main__":
    main()