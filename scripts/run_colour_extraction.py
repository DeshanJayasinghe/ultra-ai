#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from wearwise_ai.application.colour_extraction_service import (
    ColourExtractionRequest,
    ColourExtractionService,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run deterministic colour extraction on a garment cutout."
    )
    parser.add_argument("--input", required=True, help="Input garment image path.")
    parser.add_argument("--content-type", default="image/webp")
    args = parser.parse_args()

    input_path = Path(args.input)
    result = ColourExtractionService().extract(
        ColourExtractionRequest(
            asset_id=input_path.stem,
            image_bytes=input_path.read_bytes(),
            content_type=args.content_type,
        )
    )

    print(
        json.dumps(
            {
                "asset_id": result.asset_id,
                "status": result.status,
                "processing_version": result.processing_version,
                "result": result.to_ai_result(),
                "metrics": result.metrics,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
