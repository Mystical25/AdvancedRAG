from pathlib import Path
from difflib import SequenceMatcher
import hashlib
import json
import re
import shutil

import fitz
import pymupdf4llm


class MarkdownExtractor:

    # Figures smaller than this (in px, after 2x render) are considered
    # degenerate crops and are skipped instead of being sent to the vision
    # model (these are the most likely cause of device-side CUDA asserts).
    MIN_PIXMAP_DIM = 10

    # Figures larger than this (in px, longest side) are downscaled before
    # rendering, to avoid huge tensors reaching the vision model.
    MAX_PIXMAP_DIM = 2000

    # Minimum SequenceMatcher ratio (0-1) between a normalized markdown
    # paragraph and a normalized PDF text block before we trust the match
    # enough to use the block's y-position as the paragraph's anchor.
    # Tune down if figures are landing too far from their true position;
    # tune up if figures are anchoring to spurious matches.
    PARAGRAPH_MATCH_THRESHOLD = 0.35

    # If the exact same rendered image (by pixel hash) shows up more than
    # this many times across the document, treat it as recurring
    # boilerplate (letterhead, watermark, repeated header/footer logo)
    # rather than real content, and stop emitting new figure chunks/vision
    # calls for it. The first few occurrences are kept in case it's a
    # small number of legitimately repeated figures rather than true
    # boilerplate.
    REPEATED_IMAGE_OCCURRENCE_LIMIT = 3

    def __init__(
        self,
        pdf_path,
        output_dir="output",
        clean=True,
    ):

        self.pdf_path = Path(pdf_path)

        self.output_dir = Path(output_dir)

        self.markdown_dir = self.output_dir / "markdown"
        self.image_dir = self.output_dir / "images"

        if clean:
            # Wipe any images/markdown left over from a previous run.
            # Without this, re-running with a different figure-detection
            # result leaves stale PNGs on disk that have no matching
            # placeholder in the freshly-written markdown, which shows up
            # downstream as spurious "Placeholder not found" warnings.
            if self.image_dir.exists():
                shutil.rmtree(self.image_dir)
            if self.markdown_dir.exists():
                shutil.rmtree(self.markdown_dir)

        self.markdown_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.image_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.image_metadata = []
        self._image_hash_counts = {}

    def extract(self):

        document = fitz.open(self.pdf_path)

        final_markdown = []

        # Extract markdown for the WHOLE document in a single call rather
        # than once per page. pymupdf4llm decides heading levels (#/##/###)
        # by comparing font sizes within whatever it's given - calling it
        # per-page means each page independently decides what counts as
        # "the biggest heading on this page," so heading levels drift
        # across the document (page 1's title and page 9's subheading can
        # both end up marked H1). A single call keeps that comparison
        # consistent document-wide, which matters a lot for a chunker
        # (e.g. Docling's HybridChunker) that attaches the heading trail
        # to each chunk for context. page_chunks=True still gives us the
        # text broken out per page, which is what _insert_figures() needs.
        #
        # ignore_images/ignore_graphics=True stops pymupdf4llm from also
        # emitting its own inline image tags for the same figures - those
        # would otherwise be dangling references (no file written, since
        # we save figures ourselves) sitting in the middle of the page
        # text, which breaks downstream markdown parsers.
        page_dicts = pymupdf4llm.to_markdown(
            str(self.pdf_path),
            page_chunks=True,
            ignore_images=True,
            ignore_graphics=True,
        )

        for page_number in range(len(document)):

            page = document[page_number]

            page_md = page_dicts[page_number]["text"].rstrip()

            # Page-provenance marker. Once this document goes through
            # Docling as markdown, chunk metadata will reference positions
            # in the markdown, not the original PDF - there's no page
            # number or bbox back-reference like there'd be from Docling's
            # native PDF backend. An HTML comment survives standard
            # markdown parsing as an invisible/dropped node (it won't
            # intrude on the heading hierarchy the way a fake "## Page N"
            # heading would), and downstream you can recover the source
            # page for any chunk by finding the nearest preceding marker
            # via a substring search on this file's raw text.
            final_markdown.append(f"<!-- page: {page_number+1} -->")

            # ----------------------------
            # Figures on this page
            # ----------------------------

            figure_regions = self._extract_page_figures(
                page,
                page_number,
                document,
            )

            figures = []
            figure_index = 0

            for rect in figure_regions:

                pix = self._render_figure(page, rect)

                if pix is None:
                    # Degenerate crop (near-zero size) - skip it rather
                    # than sending garbage to the vision model.
                    continue

                image_hash = hashlib.md5(pix.tobytes()).hexdigest()
                occurrence = self._image_hash_counts.get(image_hash, 0) + 1
                self._image_hash_counts[image_hash] = occurrence

                if occurrence > self.REPEATED_IMAGE_OCCURRENCE_LIMIT:
                    # The exact same image has now shown up more times
                    # than any real, distinct figure reasonably would -
                    # almost certainly a letterhead/watermark/repeated
                    # logo. Skip it rather than generating another
                    # near-duplicate chunk and burning a vision-model call
                    # on content that adds no retrieval value.
                    print(
                        f"page {page_number+1}: skipping recurring "
                        f"image (seen {occurrence}x) at {rect} - "
                        f"looks like boilerplate"
                    )
                    continue

                figure_index += 1

                filename = (
                    f"page_{page_number+1}_img_{figure_index}.png"
                )

                pix.save(self.image_dir / filename)

                self.image_metadata.append(
                    {
                        "page": page_number + 1,
                        "image": filename,
                        "bbox": [
                            rect.x0,
                            rect.y0,
                            rect.x1,
                            rect.y1,
                        ],
                    }
                )

                # A short heading anchors this figure in the document's
                # section hierarchy (so a chunker's heading trail reads
                # like "...> Figure 8.1" instead of inheriting whatever
                # unrelated heading happened to precede it), and the
                # italic line directly under the image is the most
                # widely-recognized "caption" pattern in plain markdown -
                # no parser understands custom tags like
                # <ImageDescription>, but "image immediately followed by
                # an italic paragraph" is a well-established convention
                # (it's literally how Pandoc infers a figure caption).
                figure_markdown = "\n".join(
                    [
                        f"Figure {page_number+1}.{figure_index} Description:",
                        "",
                        f"![{Path(filename).stem}](../images/{filename})",
                        "",
                        "*TO_BE_FILLED_BY_VISION_PIPELINE*",
                    ]
                )

                figures.append(
                    {
                        "y": rect.y0,
                        "markdown": figure_markdown,
                    }
                )

            page_md = self._insert_figures(
                page=page,
                markdown=page_md,
                figures=figures,
            )

            final_markdown.append(page_md)
            final_markdown.append("")

        document.close()

        markdown_path = (
            self.markdown_dir /
            "enriched_document.md"
        )

        markdown_path.write_text(
            "\n".join(final_markdown),
            encoding="utf-8",
        )

        with open(
            self.output_dir / "image_metadata.json",
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                self.image_metadata,
                f,
                indent=4,
            )

        print(f"Saved markdown to {markdown_path}")

        return markdown_path

    ############################################################
    # Figure placement: approximate each markdown paragraph's page
    # position by matching it against the PDF's own text blocks, then
    # slot each figure in right after the paragraph immediately above
    # it (by y-position).
    ############################################################

    def _insert_figures(self, page, markdown, figures):

        if not figures:
            return markdown

        paragraphs = re.split(r"\n\s*\n", markdown) if markdown else [""]

        blocks = [
            b for b in page.get_text("blocks")
            if b[6] == 0 and b[4].strip()
            # (x0, y0, x1, y1, text, block_no, block_type)
        ]

        anchors = [
            self._best_matching_block_y(paragraph, blocks)
            for paragraph in paragraphs
        ]

        anchors = self._fill_anchor_gaps(anchors)

        # For each figure, find the last paragraph whose anchor sits at or
        # above the figure's top edge - that's the paragraph the figure
        # visually follows. -1 means "before every paragraph on the page."
        insert_after = {}

        for fig in sorted(figures, key=lambda f: f["y"]):

            target_index = -1

            for i, anchor_y in enumerate(anchors):
                if anchor_y <= fig["y"]:
                    target_index = i

            insert_after.setdefault(target_index, []).append(
                fig["markdown"]
            )

        output_parts = []

        output_parts.extend(insert_after.get(-1, []))

        for i, paragraph in enumerate(paragraphs):
            output_parts.append(paragraph)
            output_parts.extend(insert_after.get(i, []))

        return "\n\n".join(
            part for part in output_parts
            if part is not None and part.strip() != ""
        )

    def _best_matching_block_y(self, paragraph_text, blocks):
        """
        Find the PDF text block whose content best matches this markdown
        paragraph, and return that block's top y-coordinate as the
        paragraph's approximate position on the page. Returns None if no
        block clears PARAGRAPH_MATCH_THRESHOLD (e.g. the paragraph is a
        markdown table pymupdf4llm has reformatted heavily, or a heading
        pymupdf4llm has re-styled beyond what normalization recovers).
        """

        norm_paragraph = self._normalize_for_matching(paragraph_text)

        if not norm_paragraph:
            return None

        best_ratio = 0.0
        best_y = None

        for (x0, y0, x1, y1, block_text, block_no, block_type) in blocks:

            norm_block = self._normalize_for_matching(block_text)

            if not norm_block:
                continue

            ratio = SequenceMatcher(
                None, norm_paragraph, norm_block
            ).ratio()

            if ratio > best_ratio:
                best_ratio = ratio
                best_y = y0

        if best_ratio >= self.PARAGRAPH_MATCH_THRESHOLD:
            return best_y

        return None

    @staticmethod
    def _normalize_for_matching(text):
        """
        Strip markdown syntax and collapse whitespace so a markdown
        paragraph (e.g. "**Revenue** grew | 12% |") and its source PDF
        text block (e.g. "Revenue grew 12%") compare on content rather
        than formatting. Truncated for matching speed - full paragraphs
        can be long, and we only need enough signal to disambiguate
        between blocks on the same page.
        """

        text = re.sub(r"[#*_`>|]+", " ", text)
        text = re.sub(r"^[\s\-]+", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s+", " ", text).strip().lower()

        return text[:250]

    @staticmethod
    def _fill_anchor_gaps(anchors):
        """
        Paragraphs with no confident block match (None) get the nearest
        known neighbor's y-position, so every paragraph has *some* anchor
        to compare figures against. Forward-fill first, then back-fill
        any leading Nones from the first known value.
        """

        anchors = list(anchors)

        last_known = None
        for i, y in enumerate(anchors):
            if y is not None:
                last_known = y
            elif last_known is not None:
                anchors[i] = last_known

        first_known = next((y for y in anchors if y is not None), 0.0)
        for i, y in enumerate(anchors):
            if y is None:
                anchors[i] = first_known

        return anchors

    ############################################################

    def _render_figure(self, page, rect):
        """
        Render a figure region to a pixmap, guarding against degenerate
        (near-zero) crops and capping the max render dimension so we never
        hand the vision model something that could trigger an OOM-style
        device-side assert.
        """

        base_scale = 2.0

        try:
            pix = page.get_pixmap(
                clip=rect,
                matrix=fitz.Matrix(base_scale, base_scale),
                alpha=False,
            )
        except Exception as e:
            print(f"Failed to render figure at {rect}: {e}")
            return None

        if pix.width < self.MIN_PIXMAP_DIM or pix.height < self.MIN_PIXMAP_DIM:
            print(
                f"Skipping degenerate figure at {rect} "
                f"({pix.width}x{pix.height}px)"
            )
            return None

        longest_side = max(pix.width, pix.height)

        if longest_side > self.MAX_PIXMAP_DIM:

            scale = base_scale * (self.MAX_PIXMAP_DIM / longest_side)

            try:
                pix = page.get_pixmap(
                    clip=rect,
                    matrix=fitz.Matrix(scale, scale),
                    alpha=False,
                )
            except Exception as e:
                print(f"Failed to re-render oversized figure at {rect}: {e}")
                return None

        return pix

    def _extract_page_figures(
    self,
    page,
    page_number,
    document,
    ):
        page_rect = page.rect

        figures = []

        # =====================================================
        # 1. Embedded raster images
        # =====================================================
        for img in page.get_images(full=True):

            xref = img[0]

            try:
                image_rects = page.get_image_rects(xref)
            except Exception as e:
                print(
                    f"page {page_number+1}: get_image_rects failed for "
                    f"xref {xref}: {e}"
                )
                continue

            for rect in image_rects:

                rect = rect & page_rect

                if rect.is_empty:
                    continue

                if rect.width < 80 or rect.height < 80:
                    continue

                figures.append(rect)

        # =====================================================
        # 2. Cluster vector drawings (charts, diagrams, etc.)
        # =====================================================
        try:

            clusters = page.cluster_drawings()

        except Exception as e:

            print(
                f"page {page_number+1}: cluster_drawings failed "
                f"(charts on this page will be missed): {e}"
            )
            clusters = []

        for rect in clusters:

            rect = rect & page_rect

            if rect.is_empty:
                continue

            if rect.width < 120 or rect.height < 120:
                continue

            figures.append(rect)

        # =====================================================
        # 3. Remove duplicates / junk regions
        # =====================================================
        filtered = []

        page_area = page_rect.get_area()

        for rect in figures:

            w = rect.width
            h = rect.height

            if w < 80 or h < 80:
                continue

            ratio = max(w / h, h / w)

            # Ignore extremely thin regions. Qwen2-VL's patch-alignment
            # resize can round a very thin/wide crop's short side down to
            # 0 patches, which is a known trigger for the
            # "Assertion `input[0] != 0` failed" CUDA crash - so this
            # isn't just a quality filter, it's load-bearing for the
            # vision step too. Keep it conservative if you loosen it.
            if ratio > 8:
                continue

            # Ignore essentially-full-page crops (raised from 0.70 -> 0.95
            # so legitimate large charts/diagrams aren't discarded).
            if rect.get_area() > 0.95 * page_area:
                continue

            duplicate = False

            for existing in filtered:

                inter = existing & rect

                if inter.is_empty:
                    continue

                overlap = (
                    inter.get_area()
                    / min(existing.get_area(), rect.get_area())
                )

                if overlap > 0.90:
                    duplicate = True
                    break

            if not duplicate:
                filtered.append(rect)

        filtered.sort(
            key=lambda r: (r.y0, r.x0)
        )

        return filtered