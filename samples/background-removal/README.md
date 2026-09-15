# Background Removal Samples

These generated sample images are non-private validation fixtures for the WearWise Phase 4
background-removal pipeline.

## Samples

* `flat-lay-blue-tshirt.png` - easy flat-lay top with light background.
* `person-green-jacket.png` - worn garment with visible person context.
* `dark-hoodie-dark-background.png` - low-contrast dark garment on dark background.

## Validation Command

Use the pinned cached BiRefNet revision:

```bash
HF_HUB_DISABLE_XET=1 .venv/bin/python scripts/run_birefnet_background_removal.py \
  --input samples/background-removal/flat-lay-blue-tshirt.png \
  --output tmp/background-removal-validation/flat-lay-blue-tshirt-cutout.webp \
  --mask-output tmp/background-removal-validation/flat-lay-blue-tshirt-mask.png \
  --device cpu \
  --revision e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4 \
  --local-files-only
```

Generated masks and cutouts belong under `tmp/background-removal-validation/`.

Run colour extraction on a generated cutout:

```bash
.venv/bin/python scripts/run_colour_extraction.py \
  --input tmp/background-removal-validation/flat-lay-blue-tshirt-cutout.webp
```

The production colour-extraction job expects the cleaned cutout and background-removal quality
summary in `options.background_removed`:

```json
{
  "background_removed": {
    "storage_key": "ai-outputs/asset-1/background_removed_v4.webp",
    "content_type": "image/webp",
    "review_required": false,
    "quality_status": "accepted",
    "quality_reasons": []
  }
}
```

If `review_required` is `true`, the worker completes the colour-extraction job as review-required
without reading the cutout or suggesting colours.

For local filesystem validation, set:

```bash
WEARWISE_AI_ARTIFACT_ROOT=/path/to/artifact/root
```

Then `storage_key` values are resolved relative to that root.
