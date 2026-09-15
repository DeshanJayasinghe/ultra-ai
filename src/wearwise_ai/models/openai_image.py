from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Protocol

from wearwise_ai.application.outfit_visual_generation_service import (
    OUTFIT_VISUAL_MODEL_FAMILY,
    OUTFIT_VISUAL_MODEL_NAME,
    OUTFIT_VISUAL_MODEL_VERSION,
    OutfitVisualGenerationRequest,
    OutfitVisualOutput,
)
from wearwise_ai.application.wardrobe_item_visual_generation_service import (
    WARDROBE_ITEM_VISUAL_MODEL_FAMILY,
    WARDROBE_ITEM_VISUAL_MODEL_NAME,
    WARDROBE_ITEM_VISUAL_MODEL_VERSION,
    WardrobeItemVisualGenerationRequest,
    WardrobeItemVisualOutput,
)
from wearwise_ai.models.background_removal.birefnet_adapter import ModelUnavailableError
from wearwise_ai.models.openai_usage import image_usage_metrics, log_usage


class OpenAIImagesClient(Protocol):
    def generate(self, **kwargs: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class OpenAIOutfitVisualConfig:
    model_name: str = OUTFIT_VISUAL_MODEL_NAME
    model_version: str = OUTFIT_VISUAL_MODEL_VERSION
    size: str = "1024x1024"
    quality: str = "low"
    output_format: str = "webp"
    prompt_version: str = "outfit-visual-flatlay-v1"


class OpenAIOutfitVisualGenerator:
    def __init__(
        self,
        *,
        config: OpenAIOutfitVisualConfig | None = None,
        images_client: OpenAIImagesClient | None = None,
    ) -> None:
        self._config = config or OpenAIOutfitVisualConfig()
        self._images_client = images_client

    def generate(self, request: OutfitVisualGenerationRequest) -> OutfitVisualOutput:
        response = self._generate_response(request)
        log_usage(
            image_usage_metrics(response, model_name=self._config.model_name),
            job_type="outfit_visual_generation",
        )
        image_bytes = _response_image_bytes(response)
        width, height = _size_dimensions(self._config.size)

        return OutfitVisualOutput(
            image_bytes=image_bytes,
            content_type=f"image/{self._config.output_format}",
            width=width,
            height=height,
            model_family=OUTFIT_VISUAL_MODEL_FAMILY,
            model_name=self._config.model_name,
            model_version=self._config.model_version,
            prompt_version=self._config.prompt_version,
        )

    def _generate_response(self, request: OutfitVisualGenerationRequest) -> Any:
        client = self._client()
        prompt = _prompt(request)
        edit = getattr(client, "edit", None)
        if callable(edit):
            image_files = []
            try:
                for index, item in enumerate(request.items, start=1):
                    image_file = BytesIO(item.image_bytes)
                    image_file.name = f"outfit-item-{index}.{_extension(item.content_type)}"
                    image_files.append(image_file)

                return edit(
                    model=self._config.model_name,
                    image=image_files,
                    prompt=prompt,
                    size=self._config.size,
                    quality=self._config.quality,
                    output_format=self._config.output_format,
                )
            finally:
                for image_file in image_files:
                    image_file.close()

        return client.generate(
            model=self._config.model_name,
            prompt=prompt,
            size=self._config.size,
            quality=self._config.quality,
            output_format=self._config.output_format,
        )

    def _client(self) -> OpenAIImagesClient:
        if self._images_client is not None:
            return self._images_client

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ModelUnavailableError(
                "OpenAI SDK is not installed. Install wearwise-ai dependencies "
                "and set OPENAI_API_KEY."
            ) from exc

        self._images_client = OpenAI().images
        return self._images_client


@dataclass(frozen=True, slots=True)
class OpenAIWardrobeItemVisualConfig:
    model_name: str = WARDROBE_ITEM_VISUAL_MODEL_NAME
    model_version: str = WARDROBE_ITEM_VISUAL_MODEL_VERSION
    size: str = "1024x1024"
    quality: str = "low"
    output_format: str = "webp"
    prompt_version: str = "wardrobe-item-presentation-v1"


class OpenAIWardrobeItemVisualGenerator:
    def __init__(
        self,
        *,
        config: OpenAIWardrobeItemVisualConfig | None = None,
        images_client: OpenAIImagesClient | None = None,
    ) -> None:
        self._config = config or OpenAIWardrobeItemVisualConfig()
        self._images_client = images_client

    def generate(self, request: WardrobeItemVisualGenerationRequest) -> WardrobeItemVisualOutput:
        response = self._generate_response(request)
        log_usage(
            image_usage_metrics(response, model_name=self._config.model_name),
            job_type="wardrobe_item_visual_generation",
        )
        image_bytes = _response_image_bytes(response)
        width, height = _size_dimensions(self._config.size)

        return WardrobeItemVisualOutput(
            image_bytes=image_bytes,
            content_type=f"image/{self._config.output_format}",
            width=width,
            height=height,
            model_family=WARDROBE_ITEM_VISUAL_MODEL_FAMILY,
            model_name=self._config.model_name,
            model_version=self._config.model_version,
            prompt_version=self._config.prompt_version,
        )

    def _generate_response(self, request: WardrobeItemVisualGenerationRequest) -> Any:
        client = self._client()
        prompt = _wardrobe_item_prompt(request)
        edit = getattr(client, "edit", None)
        if callable(edit):
            image_file = BytesIO(request.image_bytes)
            image_file.name = f"wardrobe-item.{_extension(request.content_type)}"
            try:
                return edit(
                    model=self._config.model_name,
                    image=image_file,
                    prompt=prompt,
                    size=self._config.size,
                    quality=self._config.quality,
                    output_format=self._config.output_format,
                )
            finally:
                image_file.close()

        return client.generate(
            model=self._config.model_name,
            prompt=prompt,
            size=self._config.size,
            quality=self._config.quality,
            output_format=self._config.output_format,
        )

    def _client(self) -> OpenAIImagesClient:
        if self._images_client is not None:
            return self._images_client

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ModelUnavailableError(
                "OpenAI SDK is not installed. Install wearwise-ai dependencies "
                "and set OPENAI_API_KEY."
            ) from exc

        self._images_client = OpenAI().images
        return self._images_client


def _prompt(request: OutfitVisualGenerationRequest) -> str:
    items = "\n".join(f"- {item.category_code}: {item.name}" for item in request.items)
    title = request.title or "complete outfit"
    style = request.style or "premium flat-lay outfit card"

    return (
        "Create a premium fashion outfit flat-lay image for a mobile outfit suggestion card.\n"
        f"Outfit title: {title}\n"
        f"Visual style: {style}\n"
        "Use the listed wardrobe pieces and provided item images as the complete outfit:\n"
        f"{items}\n"
        "Arrange a complete wearable outfit on a clean dark editorial background. "
        "Preserve the visible colors, garment types, patterns, and realistic proportions. "
        "Do not add a person, mannequin, body, face, hanger, labels, text, watermark, "
        "or extra clothing categories. Make every piece look tidy, smoothed, and deliberately "
        "styled. Center the outfit with top or jacket in the upper area, bottom centered below, "
        "shoes at the bottom, and accessories to the side with balanced spacing."
    )


def _wardrobe_item_prompt(request: WardrobeItemVisualGenerationRequest) -> str:
    name = request.name or "wardrobe item"
    category = request.category_code or "clothing"
    subcategory = request.subcategory_code or "unspecified"

    return (
        "Create a clean catalog wardrobe image of this exact clothing item for a mobile "
        "wardrobe app.\n"
        f"Item name: {name}\n"
        f"Category: {category}\n"
        f"Subcategory: {subcategory}\n"
        "Use the provided item image as the source garment. Preserve the original color, "
        "fabric texture, silhouette, pattern, logos, seams, stitching, pockets, buttons, "
        "and realistic proportions. Make the item tidy, centered, front-facing, and pleasant. "
        "Remove messy folds, harsh wrinkles, clutter, hangers, and leftover background artifacts. "
        "For tops, straighten the body and place sleeves naturally. For trousers or jeans, align "
        "the waistband and legs neatly. For shoes, arrange the pair together at a slight "
        "three-quarter angle. Use a transparent or soft neutral studio background with balanced "
        "margins. Do not add a person, mannequin, body, face, hanger, text, watermark, props, "
        "or extra accessories."
    )


def _response_image_bytes(response: Any) -> bytes:
    data = getattr(response, "data", None)
    if not isinstance(data, list) or data == []:
        raise ModelUnavailableError("OpenAI outfit visual generator returned no image data.")

    first = data[0]
    encoded = getattr(first, "b64_json", None)
    if not isinstance(encoded, str) or encoded.strip() == "":
        raise ModelUnavailableError("OpenAI outfit visual generator returned an empty image.")

    try:
        return base64.b64decode(encoded)
    except ValueError as exc:
        raise ModelUnavailableError("OpenAI outfit visual generator returned invalid base64.") from exc


def _size_dimensions(size: str) -> tuple[int, int]:
    parts = size.lower().split("x", maxsplit=1)
    if len(parts) != 2:
        return (1024, 1536)

    try:
        return (int(parts[0]), int(parts[1]))
    except ValueError:
        return (1024, 1536)


def _extension(content_type: str) -> str:
    return {
        "image/jpeg": "jpg",
        "image/png": "png",
        "image/webp": "webp",
    }.get(content_type, "png")
