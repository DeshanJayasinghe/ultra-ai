# AI Cost Optimization Playbook

Companion to [ai-cost-analysis.md](./ai-cost-analysis.md). This document holds the concrete code changes, in the order they should be shipped.

---

## Step 1 — Cap reasoning and output (ship first, ~30 min)

**Why:** `gpt-5` is a reasoning model. Nothing in `src/` sets `reasoning`, `verbosity`, or `max_output_tokens`, so every call defaults to medium reasoning and bills hidden reasoning tokens at $10/1M. For enum-constrained JSON extraction, reasoning buys nothing.

**Files:** `models/openai_vision.py`, `models/classification/openai_adapter.py`

Add these three parameters to each of the three `create(...)` calls:

```python
response = self._client().create(
    model=self._config.model_name,
    input=_vision_input(...),
    reasoning={"effort": "minimal"},          # <- add
    max_output_tokens=600,                    # <- add
    text={
        "verbosity": "low",                   # <- add
        "format": {
            "type": "json_schema",
            "name": "wearwise_colour_extraction",
            "strict": True,
            "schema": _colour_schema(),
        },
    },
)
```

Make them configurable rather than hardcoded, so they can be tuned per task:

```python
@dataclass(frozen=True, slots=True)
class OpenAIVisionExtractionConfig:
    model_name: str = COLOUR_EXTRACTION_MODEL_NAME
    model_version: str = COLOUR_EXTRACTION_MODEL_VERSION
    image_detail: str = "low"                 # was "high"
    reasoning_effort: str = "minimal"
    max_output_tokens: int = 600
    verbosity: str = "low"
```

**Watch for:** if a task starts returning `review_required: true` more often, raise that single task to `effort: "low"` rather than reverting everything.

---

## Step 2 — Make spend measurable (ship second)

**Why:** `resource_metrics` exists on every result payload but carries no token counts. Today you cannot attribute cost per job type, so you cannot verify any of the savings below.

Capture `response.usage` in each adapter and thread it into the result metrics:

```python
def _usage_metrics(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}

    details = getattr(usage, "output_tokens_details", None)
    return {
        "openai_input_tokens": getattr(usage, "input_tokens", 0),
        "openai_output_tokens": getattr(usage, "output_tokens", 0),
        "openai_cached_input_tokens": getattr(
            getattr(usage, "input_tokens_details", None), "cached_tokens", 0
        ),
        "openai_reasoning_tokens": getattr(details, "reasoning_tokens", 0),
    }
```

`RawColourExtraction` / `RawAttributeExtraction` / `RawCategoryPrediction` each need a `usage: dict[str, int]` field, merged into `metrics` in the corresponding service. The `resource_metrics` dict already flows through to Laravel unchanged, so nothing downstream breaks.

Once this lands, the reasoning-token count will confirm Step 1 worked, and give you a real per-job-type cost breakdown.

---

## Step 3 — Make wardrobe visual generation opt-in (largest single saving)

**Why:** ~62% of current spend. `gpt-image-1` at medium quality is ~$0.07 per item, charged automatically on every upload for a cosmetic feature the user never requested — on top of a BiRefNet cutout that is already free and already good.

**File:** `wearwise-api/app/Modules/Ai/Services/AiJobService.php:843`

```php
$this->queueBackgroundRemovalFollowUp($job, $backgroundRemoved, 'colour_extraction');
$this->queueBackgroundRemovalFollowUp($job, $backgroundRemoved, 'embedding_generation');
$this->queueBackgroundRemovalFollowUp($job, $backgroundRemoved, 'clothing_classification');
$this->queueBackgroundRemovalFollowUp($job, $backgroundRemoved, 'attribute_extraction');
// Removed from the automatic fan-out: queued on explicit user request instead.
// $this->queueBackgroundRemovalFollowUp($job, $backgroundRemoved, 'wardrobe_item_visual_generation');
```

Expose an endpoint that queues `wardrobe_item_visual_generation` for a given `media_asset_id`, and wire it to a "Generate clean image" action in the mobile app.

**If it must remain automatic**, apply Step 3b instead.

### Step 3b — Downgrade the image model

**File:** `models/openai_image.py:116-121`

```python
@dataclass(frozen=True, slots=True)
class OpenAIWardrobeItemVisualConfig:
    model_name: str = "gpt-image-1-mini"   # was gpt-image-1; 5x cheaper output tokens
    size: str = "1024x1024"
    quality: str = "low"                   # was "medium"; ~$0.004 vs ~$0.07
    output_format: str = "webp"
```

Also update the default in `run_worker.py:200` and the constant in `application/wardrobe_item_visual_generation_service.py:9`.

Validate on ~20 real garments across all five categories before committing — for a flat-lay on a neutral background, `-mini` at low is likely indistinguishable, but confirm it rather than assume it.

---

## Step 4 — Send smaller images

**Why:** `detail: "high"` on a 2048px image costs 765 input tokens; `detail: "low"` is a flat 85. Category and colour do not need 2048px.

**File:** `run_worker.py:135, 150, 172` — change all three `*_IMAGE_DETAIL` defaults from `"high"` to `"low"`.

**File:** `models/openai_vision.py` — downscale before encoding:

```python
def _data_url(*, content_type: str, image_bytes: bytes, max_edge: int = 512) -> str:
    image_bytes, content_type = _downscaled(image_bytes, content_type, max_edge)
    encoded = base64.b64encode(image_bytes).decode("ascii")
    safe_content_type = content_type if content_type.startswith("image/") else "image/png"
    return f"data:{safe_content_type};base64,{encoded}"


def _downscaled(image_bytes: bytes, content_type: str, max_edge: int) -> tuple[bytes, str]:
    from PIL import Image

    with Image.open(BytesIO(image_bytes)) as image:
        if max(image.size) <= max_edge:
            return image_bytes, content_type

        rgba = image.convert("RGBA")
        rgba.thumbnail((max_edge, max_edge), Image.LANCZOS)
        buffer = BytesIO()
        rgba.save(buffer, format="WEBP", quality=85)
        return buffer.getvalue(), "image/webp"
```

Keep the alpha channel — the background-removed transparency is exactly what tells the model where the garment is. Apply the same helper in `classification/openai_adapter.py`.

---

## Step 5 — Merge three calls into one

**Why:** Category, colour, and attributes are all one look at one garment. Three calls means 3× image tokens, 3× reasoning warm-up, 3× failure surface — and an ordering bug where `attribute_extraction` wants a `category_code` that `clothing_classification` has not produced yet, because they run in parallel.

Add `models/openai_garment_analysis.py` with a single `OpenAIGarmentAnalyser` whose schema is the union of the three existing schemas. `_colour_schema()` and `_attribute_schema()` already compose — merge their `properties` and `required` lists and add the classification fields:

```python
def _garment_schema() -> dict[str, object]:
    attribute = _attribute_schema()
    colour = _colour_schema()
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "category", "category_confidence", "alternatives",
            *[k for k in colour["required"] if k not in {"review_required", "review_reason"}],
            *[k for k in attribute["required"] if k not in {"review_required", "review_reason"}],
            "review_required", "review_reason",
        ],
        "properties": {
            "category": {"type": "string", "enum": [...]},
            "category_confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "alternatives": OPENAI_CLASSIFICATION_RESPONSE_SCHEMA["properties"]["alternatives"],
            **colour["properties"],
            **attribute["properties"],
        },
    }
```

**Keep the three Laravel job types.** The API contract and the mobile app stay unchanged. Instead, have the analyser write its result into a cache keyed by `sha256(cleaned_bytes) + processing_version`; whichever of the three jobs runs first performs the call, and the other two read the cached analysis and project out their slice of it. Each job's `to_complete_payload` is unchanged.

This is the change that also removes the category-hint ordering bug, since the model now sees category and attributes in the same pass.

---

## Step 6 — Tier the models

**Why:** Picking 1 of 6 enums does not need a frontier model. `gpt-5-nano` is 25× cheaper than `gpt-5` on both input and output.

### 6a — Un-park SigLIP for classification ($0)

**File:** `workers/run_worker.py:334-340`

The runner, adapter, and integration tests all exist, and the SigLIP weights are **already loaded in this process** for `embedding_generation` — classification through it is free.

```python
siglip_classifier_runner = TransformersSigLIPZeroShotCategoryRunner(
    runner_config=siglip_runner_config,
)
```

Wire it as the primary classifier with OpenAI as the escalation path, using the existing `min_confidence` threshold as the trigger:

```python
"clothing_classification": ClothingClassificationJobWorkflow(
    internal_api=internal_api,
    classification_service=ClothingClassificationService(
        classifier=ConfidenceRoutedClassifier(
            primary=SigLIPCategoryClassifier(runner=siglip_classifier_runner),
            fallback=OpenAICategoryClassifier(
                config=OpenAIClassificationConfig(model_name="gpt-5-mini", image_detail="high"),
            ),
            min_confidence=0.6,
        ),
    ),
    artifact_reader=artifact_store,
),
```

`ConfidenceRoutedClassifier` is a small new class: call `primary`, return it when `confidence >= min_confidence and not review_required`, otherwise call `fallback`. Log which path was taken in `resource_metrics` so you can track the escalation rate.

### 6b — Deterministic colour extraction ($0)

`ColourExtractionService` already contains a complete alpha-dominant-colour implementation that runs when `extractor is None` (`colour_extraction_service.py`). The artifact is alpha-masked, so dominant colour is a histogram over visible pixels — exact, instant, free.

Drop the `extractor=` argument in `run_worker.py:360-367` and let the deterministic path run. Escalate to `gpt-5-nano` only when the deterministic result sets `review_required` (e.g. ambiguous or multi-coloured garments).

### 6c — `gpt-5-nano` for the merged analysis

```python
MODEL_NAME = "gpt-5-nano"   # escalate to gpt-5-mini on low confidence
```

Set via `WEARWISE_OPENAI_*_MODEL` env vars first and A/B against the current output on a fixed sample of ~50 items before changing the constants.

**Brand detection is the one field that genuinely needs resolution** — reading a logo or wordmark. Route brand to the `gpt-5-mini` + `detail: "high"` escalation tier, and only when the item is flagged as likely-branded, rather than on every item.

---

## Step 7 — Caching, batching, and guardrails

### Content-hash cache

Key every AI result on `sha256(cleaned_artifact_bytes) + processing_version`. Re-uploads, duplicate garments, and post-failure retries become free. The `processing_version` constants (`ATTRIBUTE_EXTRACTION_PROCESSING_VERSION`, etc.) already exist and already change when logic changes — they are the correct cache-busting key.

### Prompt caching

The developer instruction and JSON schema are byte-identical on every call. Ensure they sit at the **front** of the input so OpenAI's automatic prompt caching can hit them — cached input is 10× cheaper. `_vision_input` already puts the developer message first; keep it that way and never interpolate per-item values (like `category_hint`) into the developer message, which would break the cache prefix. Move `category_hint` into the user message instead:

```python
# openai_vision.py — currently breaks the cache prefix:
instruction=(
    "Extract wardrobe item attributes ... "
    f"The current category hint is {category_hint}. ..."   # <- move this to the user turn
),
```

### Batch API — 50% off

Attribute extraction and visual generation are not latency-critical; the app already polls job status. Route them through the Batch API for a flat 50% discount on whatever remains.

### Spend guardrails

- **Stop retrying deterministic failures.** `run_worker.py:300` marks every failure `"retryable": True`. A `ModelUnavailableError` from `_response_json` (invalid JSON / non-object payload) will never succeed on retry — it just re-charges the call. Classify errors and only retry transport-level ones.
- Add a per-account daily spend ceiling, enforced in `AiJobService` before queuing.
- Alert when cost-per-item exceeds an expected band — using the usage metrics from Step 2.

---

## Verification checklist

Before/after each step, on a fixed sample of ~50 real garments:

- [ ] `openai_reasoning_tokens` drops to near zero after Step 1
- [ ] Per-item cost logged and trending down (Step 2 makes this observable)
- [ ] Category accuracy vs. current `gpt-5` baseline holds within tolerance
- [ ] `review_required` rate does not climb materially
- [ ] Colour accuracy holds after the deterministic switch (6b)
- [ ] Brand detection still works on the escalation tier (6c)
- [ ] Visual-generation quality accepted by product after model downgrade (3b)
