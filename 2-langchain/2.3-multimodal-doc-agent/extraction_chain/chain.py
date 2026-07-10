"""
Multimodal extraction chain — ingests documents and extracts structured
data using llama.cpp vision models.

Usage::

    from pydantic import BaseModel, Field
    from extraction_chain import MultimodalExtractionChain
    from loaders import ImageLoader

    class Invoice(BaseModel):
        vendor: str = Field(description="Vendor name")
        total: float = Field(description="Total amount")
        date: str = Field(description="Invoice date")

    chain = MultimodalExtractionChain(
        base_url="http://127.0.0.1:8080",
        schema=Invoice,
        system_prompt="Extract invoice fields from the image. Output valid JSON.",
    )

    # Load images
    loader = ImageLoader()
    images = loader.load_folder("invoices/")

    # Extract from each image
    for img in images:
        result = chain.extract(img.data_uri)
        print(result)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel

from .vision_model import VisionChatModel


class MultimodalExtractionChain:
    """
    LangChain-compatible chain for vision-based document extraction.

    Combines:
    1. Document loading (images, PDFs)
    2. Vision model inference (llama.cpp ``/v1/chat/completions`` multimodal)
    3. Structured JSON parsing into Pydantic models
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        schema: Optional[Type[BaseModel]] = None,
        system_prompt: Optional[str] = None,
        prompt_template: str = "Extract the requested information from this image. Output valid JSON matching the expected schema.",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        request_timeout: float = 120.0,
    ):
        """
        Args:
            base_url: llama.cpp server URL.
            schema: Optional Pydantic model for structured output.
            system_prompt: Optional system instruction.
            prompt_template: Text prompt sent alongside images.
            temperature: Sampling temperature.
            max_tokens: Max tokens for generation.
            request_timeout: HTTP request timeout.
        """
        self.base_url = base_url
        self.schema = schema
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout

        self._model = VisionChatModel(
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            request_timeout=request_timeout,
        )

    def extract(
        self,
        image_uri: str,
        prompt: Optional[str] = None,
        schema: Optional[Type[BaseModel]] = None,
    ) -> Union[str, BaseModel]:
        """
        Extract information from a single image.

        Args:
            image_uri: Base64 data URI of the image.
            prompt: Optional override for the prompt template.
            schema: Optional override for the output schema.

        Returns:
            If ``schema`` is set: a validated Pydantic instance.
            Otherwise: raw text response.
        """
        text = self._model.invoke_with_images(
            prompt=prompt or self.prompt_template,
            images=[image_uri],
            system_prompt=self.system_prompt,
        )

        # Parse into Pydantic if schema provided
        target_schema = schema or self.schema
        if target_schema is not None:
            return self._parse_json(text, target_schema)

        return text

    def extract_batch(
        self,
        image_uris: List[str],
        prompt: Optional[str] = None,
    ) -> List[Union[str, BaseModel]]:
        """
        Extract from multiple images sequentially.

        Args:
            image_uris: List of base64 data URIs.
            prompt: Optional override for the prompt template.

        Returns:
            List of results (raw text or Pydantic instances).
        """
        return [self.extract(uri, prompt=prompt) for uri in image_uris]

    def extract_with_context(
        self,
        image_uri: str,
        context: str,
        question: str,
    ) -> Union[str, BaseModel]:
        """
        Extract with additional context (e.g. "This is page 2 of 5").

        Args:
            image_uri: Base64 data URI.
            context: Extra context string added to the prompt.
            question: Specific question to answer.

        Returns:
            Result text or Pydantic instance.
        """
        prompt = f"{context}\n\n{question}\n\n{self.prompt_template}"
        return self.extract(image_uri, prompt=prompt)

    @staticmethod
    def _parse_json(text: str, schema: Type[BaseModel]) -> BaseModel:
        """Parse JSON text into a Pydantic model."""
        import json
        import re

        # Strip markdown code fences
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse vision model output as JSON: {e}\n"
                f"Output: {text[:500]}"
            ) from e

        return schema.model_validate(data)

    def close(self) -> None:
        self._model.close()
