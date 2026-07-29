"""
==================================================
Vision Model
==================================================

Loads the Qwen2.5-VL model used for understanding
images extracted from PDF documents.

Responsibilities

• Load the model once
• Generate image descriptions
• Perform OCR
• Explain charts
• Explain graphs
• Explain diagrams

This module does NOT perform image extraction.
==================================================
"""

# --------------------------------------------------
# Imports
# --------------------------------------------------

from PIL import Image

import re

import torch

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

from config import (
    DEVICE,
    TORCH_DTYPE,
    VISION_MODEL,
    VISION_MAX_NEW_TOKENS,
    IMAGE_DESCRIPTION_PROMPT,
)


# ==================================================
# Vision Model
# ==================================================

class VisionModel:
    """
    Wrapper around Qwen2.5-VL.
    """

    def __init__(self) -> None:
        print(f"Loading vision model: {VISION_MODEL}")

        try:
            self.processor = AutoProcessor.from_pretrained(
                VISION_MODEL
            )

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                VISION_MODEL,
                torch_dtype=TORCH_DTYPE,
            )
            self.model.to("cuda")

            self.model.eval()
        except Exception as exc:
            raise RuntimeError(
                f"Qwen vision model {VISION_MODEL} could not be loaded: {exc}"
            ) from exc

    # ==================================================
    # Image Description
    # ==================================================

    def describe_image(
        self,
        image_path: str,
        prompt: str = IMAGE_DESCRIPTION_PROMPT,
    ) -> str:
        """
        Generate a detailed description of an image.

        Parameters
        ----------
        image_path:
            Path to the image.

        prompt:
            Instruction given to Qwen.

        Returns
        -------
        str
            Model response.
        """

        image = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.model.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=VISION_MAX_NEW_TOKENS,
            )

        input_token_count = inputs["input_ids"].shape[-1]
        answer_ids = generated_ids[:, input_token_count:]

        output = self.processor.batch_decode(
            answer_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]

        return self._extract_generated_answer(output)

    @staticmethod
    def _extract_generated_answer(
        output: str,
    ) -> str:
        """
        Remove the echoed prompt and keep only the newly generated answer.
        """

        text = output.strip()
        if not text:
            return ""

        transcript_match = re.search(
            r"(?is)\bsystem\b.*?\buser\b.*?\bassistant\b[:\s]*(.*)$",
            text,
        )
        if transcript_match:
            return transcript_match.group(1).strip()

        assistant_match = re.search(r"(?is)\bassistant\b[:\s]*(.*)$", text)
        if assistant_match:
            return assistant_match.group(1).strip()

        return text
