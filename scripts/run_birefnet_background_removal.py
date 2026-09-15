#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from wearwise_ai.application.background_removal_service import BackgroundRemovalRequest
from wearwise_ai.models.background_removal import (
    HuggingFaceBiRefNetRunnerConfig,
    PillowAlphaMaskPostprocessor,
    TransformersBiRefNetPredictor,
)
from wearwise_ai.preprocessing import ImageProbe, probe_image_bytes, plan_normalization


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local BiRefNet background removal.")
    parser.add_argument("--input", required=True, help="Input garment image path.")
    parser.add_argument("--output", required=True, help="Output transparent cutout path.")
    parser.add_argument("--mask-output", help="Optional output mask PNG path.")
    parser.add_argument("--model-id", default="ZhengPeng7/BiRefNet")
    parser.add_argument("--revision")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--half", action="store_true", help="Use fp16 inference.")
    args = parser.parse_args()

    input_path = Path(args.input)
    image_bytes = input_path.read_bytes()
    probe = _probe_or_fallback(image_bytes)
    request = BackgroundRemovalRequest(
        asset_id=input_path.stem,
        image_bytes=image_bytes,
        normalization_plan=plan_normalization(probe),
    )

    runner_config = HuggingFaceBiRefNetRunnerConfig(
        model_id=args.model_id,
        revision=args.revision,
        device=args.device,
        use_half_precision=args.half,
        local_files_only=args.local_files_only,
    )

    predictor = TransformersBiRefNetPredictor()
    mask_bytes, confidence, metrics = predictor.predict_mask(
        image_bytes=image_bytes,
        runner_config=runner_config,
    )

    if args.mask_output:
        Path(args.mask_output).write_bytes(mask_bytes)

    cutout = PillowAlphaMaskPostprocessor().build_cutout(
        request=request,
        mask_bytes=mask_bytes,
        confidence=confidence,
        metrics=metrics,
    )
    Path(args.output).write_bytes(cutout.image_bytes)

    print(f"wrote cutout: {args.output}")
    if args.mask_output:
        print(f"wrote mask: {args.mask_output}")
    print(f"confidence: {cutout.confidence:.4f}")
    print(f"metrics: {cutout.metrics}")
    return 0


def _probe_or_fallback(image_bytes: bytes) -> ImageProbe:
    try:
        return probe_image_bytes(image_bytes)
    except Exception:
        from PIL import Image
        from io import BytesIO

        with Image.open(BytesIO(image_bytes)) as image:
            return ImageProbe(
                width=image.width,
                height=image.height,
                format_name=(image.format or "unknown").lower(),
                mime_type=Image.MIME.get(image.format, "application/octet-stream"),
            )


if __name__ == "__main__":
    raise SystemExit(main())
