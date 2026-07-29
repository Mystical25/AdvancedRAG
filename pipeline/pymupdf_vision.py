import json
import re
from pathlib import Path

from models.vision import VisionModel

try:
    from PIL import Image
except ImportError:
    Image = None


# Substrings that indicate a CUDA device-side assert has fired. Once this
# happens the CUDA context is corrupted for the rest of the process - every
# subsequent model call will throw the same generic error regardless of
# input, so there's no point continuing to call the model.
_CUDA_POISON_MARKERS = (
    "device-side assert",
    "CUDA error",
)


class VisionPipeline:

    def __init__(self):

        self.vision_model = VisionModel()
        self._cuda_poisoned = False

    ############################################################

    def process_document(
        self,
        markdown_path,
        image_dir,
    ):

        markdown_path = Path(markdown_path)
        image_dir = Path(image_dir)

        lines = markdown_path.read_text(
            encoding="utf-8"
        ).splitlines()

        metadata = []

        images = sorted(
            image_dir.glob("*.png"),
            key=self._natural_sort_key,
        )

        for image in images:

            print(f"Processing {image.name}")

            description = self._describe_image(image)

            self._insert_description(
                lines,
                image.name,
                description,
            )

            metadata.append(
                {
                    "image": image.name,
                    "description": description,
                }
            )

        markdown_path.write_text(
            "\n".join(lines),
            encoding="utf-8",
        )

        with open(
            image_dir.parent / "image_metadata.json",
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                metadata,
                f,
                indent=4,
                ensure_ascii=False,
            )

        print("Vision enrichment completed.")

    ############################################################

    def _describe_image(
        self,
        image_path,
    ):

        if self._cuda_poisoned:
            # The GPU context is already corrupted from an earlier image
            # in this run. Calling the model again would just throw the
            # same generic error regardless of this image's content, so
            # skip straight to a distinguishable placeholder instead of
            # burning time on a doomed call and mislabeling a perfectly
            # fine image as having "no visual description available."
            return "SKIPPED_GPU_CONTEXT_CORRUPTED - needs reprocessing"

        try:

            description = self.vision_model.describe_image(
                str(image_path)
            )

        except Exception as e:

            error_text = str(e)

            try:
                size = image_path.stat().st_size
            except OSError:
                size = "unknown"

            dims = self._image_dims(image_path)

            print(
                f"Failed to describe {image_path.name} "
                f"(size={size} bytes, dims={dims}): {e}"
            )

            if any(marker in error_text for marker in _CUDA_POISON_MARKERS):
                self._cuda_poisoned = True
                print(
                    f"CUDA device-side assert detected on "
                    f"{image_path.name} (dims={dims}) - GPU context is "
                    f"now corrupted, aborting further model calls for "
                    f"this run. Fix/exclude this image and re-run to "
                    f"pick up the remaining descriptions."
                )
                return "SKIPPED_GPU_CONTEXT_CORRUPTED - needs reprocessing"

            description = ""

        description = self._sanitize(description)

        if not description:

            description = "No visual description available."

        return description

    ############################################################

    @staticmethod
    def _image_dims(image_path):
        """Best-effort width x height lookup for diagnostics."""

        if Image is None:
            return "unknown (Pillow not installed)"

        try:
            with Image.open(image_path) as img:
                w, h = img.size
                ratio = max(w / h, h / w) if h and w else float("inf")
                return f"{w}x{h} (ratio={ratio:.1f})"
        except Exception:
            return "unknown"

    ############################################################

    _CAPTION_LINE_RE = re.compile(r"^\*TO_BE_FILLED_BY_VISION_PIPELINE\*$")

    def _insert_description(
        self,
        lines,
        image_name,
        description,
    ):

        image_id = Path(image_name).stem

        image_line = (
            f"![{image_id}](../images/{image_name})"
        )

        escaped_description = self._escape_markdown_emphasis(description)

        for i, line in enumerate(lines):

            if line.strip() != image_line:
                continue

            # The caption line is a few lines below the image (image,
            # blank line, caption) - search a small window rather than
            # assuming an exact offset, so this stays robust to minor
            # formatting changes upstream.
            for j in range(i + 1, min(i + 6, len(lines))):

                if not self._CAPTION_LINE_RE.match(lines[j].strip()):
                    continue

                lines[j] = f"*{escaped_description}*"

                return

            print(
                f"Warning: image found but caption line not found for "
                f"{image_name}"
            )
            return

        print(f"Warning: Placeholder not found for {image_name}")

    ############################################################

    @staticmethod
    def _escape_markdown_emphasis(text):
        """
        The caption line uses *asterisks* for markdown emphasis. If the
        model's description itself contains an asterisk, it would
        prematurely close the emphasis span and corrupt the line - so
        escape it. Newlines are also flattened since this must stay a
        single markdown line.
        """

        text = text.replace("*", "\\*")
        text = re.sub(r"\s+", " ", text).strip()

        return text

    ############################################################

    @staticmethod
    def _natural_sort_key(path):
        """
        Sort filenames like page_1_img_2.png, page_1_img_10.png in
        numeric order rather than lexicographic order (plain sorted()
        would put img_10 before img_2).
        """

        return [
            int(chunk) if chunk.isdigit() else chunk
            for chunk in re.split(r"(\d+)", path.stem)
        ]

    ############################################################

    @staticmethod
    def _sanitize(text):

        if text is None:
            return ""

        text = re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

        # Strip a leading chat-template artifact like "assistant: <desc>".
        # Anchored to the START of the string (re.match, not re.search) so
        # a description that legitimately contains the word "assistant"
        # mid-sentence (e.g. "a virtual assistant icon") is no longer
        # truncated down to whatever follows that word.
        assistant = re.match(
            r"^\s*assistant[:\s]*(.*)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if assistant:
            return assistant.group(1).strip()

        return text