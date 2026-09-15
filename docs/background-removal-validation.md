# Background Removal Validation

Date: 2026-08-20

Model:

* Family: `birefnet`
* Model: `ZhengPeng7/BiRefNet`
* Tested revision: `e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4`
* Device: local CPU
* Input size: `1024 x 1024`

## Results

| Sample | Result | Confidence | Inference Time | Notes |
| --- | --- | ---: | ---: | --- |
| `flat-lay-blue-tshirt.png` | Pass with minor edge halo | 0.5194 | 14.749s | Garment shape preserved well. Some background bleed remains near soft edges. |
| `person-green-jacket.png` | Needs product handling | 0.6376 | 13.776s | Model segmented the salient foreground person plus clothing, not only the jacket. Worn-item uploads need garment localization, crop guidance, or manual review. |
| `dark-hoodie-dark-background.png` | Pass for hard sample with edge risk | 0.5063 | 13.921s | Hoodie remains mostly intact despite low contrast. Background bleed and edge softness need review in wardrobe-card UI. |

## Product Decision

BiRefNet is suitable as the MVP background-removal model for flat-lay and single-garment photos.
For worn photos, BiRefNet alone is not enough to isolate a specific garment from the person.

The pipeline now records a first-pass quality gate:

* `foreground_coverage_ratio`
* `transparent_coverage_ratio`
* `edge_foreground_ratio`
* `background_removal_quality_status`
* `background_removal_quality_reasons`

The colour-extraction worker consumes this via `options.background_removed`. When
`review_required` is `true`, it completes with `review_required: true` and no colour suggestions
instead of extracting misleading colours from a poor cutout.

The quality status is `accepted` when coverage, edge-touch and confidence are inside the MVP
thresholds. It becomes `needs_review` when the foreground is too small, too large, too close to
the image edges, or low-confidence.

Phase 4 should therefore:

* prefer flat-lay/captured-garment guidance in the mobile capture UX
* allow worn photos, but flag them for review when the foreground includes body regions
* add garment localization or crop guidance before background removal if worn-photo quality becomes important for MVP
* keep manual item creation and manual correction available when AI output is not clean enough

## Colour Extraction Follow-Up

Deterministic colour extraction was run against the generated cutouts:

| Sample | Primary | Secondary | Notes |
| --- | --- | --- | --- |
| `flat-lay-blue-tshirt-cutout.webp` | `blue` | `navy` | Good MVP output for a blue shirt. |
| `dark-hoodie-dark-background-cutout.webp` | `black` | none | Good MVP output for a black hoodie. |
| `person-green-jacket-cutout.webp` | `black` | none | Poor product output because the cutout includes dark pants/person regions. Downstream colour extraction should not blindly continue when background removal is `needs_review`. |

## Commands Used

```bash
HF_HUB_DISABLE_XET=1 .venv/bin/python scripts/run_birefnet_background_removal.py \
  --input samples/background-removal/flat-lay-blue-tshirt.png \
  --output tmp/background-removal-validation/flat-lay-blue-tshirt-cutout.webp \
  --mask-output tmp/background-removal-validation/flat-lay-blue-tshirt-mask.png \
  --device cpu \
  --revision e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4 \
  --local-files-only
```

Repeat the same command for:

* `samples/background-removal/person-green-jacket.png`
* `samples/background-removal/dark-hoodie-dark-background.png`
