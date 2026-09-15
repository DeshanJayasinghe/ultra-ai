# AI Cost — Round 2 Findings

> **Status: all six fixes implemented on 2026-09-14.** See [§9 What shipped](#9-what-shipped).

**Date:** 2026-09-14 (after round-1 optimizations shipped)
**Observed:** 6 image uploads + outfit creation = 20 Responses calls, **$0.91**
**Reported symptom:** "credits deduct automatically and continuously"

Follow-up to [ai-cost-analysis.md](./ai-cost-analysis.md) and [ai-cost-optimization-playbook.md](./ai-cost-optimization-playbook.md).

---

## 1. First: there is no runaway background loop

The worker log shows the **opposite** of a leak. After the last job completes at 14:19:15, the tail is:

```
14:19:20 /internal/v1/ai/jobs/claim-next  ~ 1.96ms
14:19:25 /internal/v1/ai/jobs/claim-next  ~ 0.22ms
14:19:30 /internal/v1/ai/jobs/claim-next  ~ 0.24ms
... every 5s, forever
```

This is `--poll-interval-seconds 5.0` ([run_worker.py:218](../src/wearwise_ai/workers/run_worker.py#L218)) doing exactly its job: asking Laravel "any work?" and being told no. It is a **local HTTP call to your own API**, hits no OpenAI endpoint, and costs **$0.00**.

The same is true of the repeated `/api/v1/ai/jobs/{id}` lines every ~2s — that is the **mobile app polling job status** while it waits for a result. Also local, also free.

**Zero OpenAI calls happen between job completions.** Credits are not draining in the background. Every cent was spent by a job that appears in the log with a `claimed → completed` pair.

What makes it *feel* continuous is that the expensive jobs are **slow**: a single `outfit_visual_generation` took from 14:17:50 to 14:18:37 (~47s). Twelve of those run back-to-back for ~10 minutes of steady spending, while the polling noise scrolls past.

---

## 2. Where the $0.91 actually went

Your round-1 changes worked. Verified from the log detail screenshots:

- `Reasoning effort: minimal`, `Verbosity: low`, `Max output tokens: 900`, `detail: low` — all applied ✅
- "Reasoning: **Empty reasoning item**" — reasoning tokens eliminated ✅
- The merged analyser is live: one `"Analyse this garment"` call returns category **and** attributes together ✅
- Only 6 `Analyse this garment` calls for 6 images — down from 18 ✅

The breakdown:

| Work | Count | Model | Cost |
|---|---|---|---|
| Garment analysis | 6 | `gpt-5` @ 803 in / 323 out | ~$0.025 |
| Wardrobe item visuals | 6 | `gpt-image-1-mini` low | ~$0.07 |
| **Outfit visuals** | **12** | **`gpt-image-1` medium 1024×1536** | **~$0.84** |
| **Total** | | | **~$0.93** |

**~92% of the remaining spend is `outfit_visual_generation`** — the one path round 1 never touched.

---

## 3. Root cause A — outfit visual generation was never optimized

[`application/outfit_visual_generation_service.py:9`](../src/wearwise_ai/application/outfit_visual_generation_service.py#L9) and [`models/openai_image.py:31-35`](../src/wearwise_ai/models/openai_image.py#L31-L35) are still at round-0 settings:

```python
OUTFIT_VISUAL_MODEL_NAME = "gpt-image-1"   # not -mini
size: str = "1024x1536"                     # portrait, ~1.5x the tokens of square
quality: str = "medium"                     # ~$0.07/image
```

The wardrobe-item generator was downgraded to `gpt-image-1-mini` / `quality: low`; **its sibling was missed.** This is the single highest-value fix remaining.

---

## 4. Root cause B — 12 outfit jobs for one onboarding

[`style_step.dart:72`](../../wearwise-mobile/lib/features/onboarding/presentation/steps/style_step.dart#L72) hardcodes `count: 6`, then `_createOutfitVisualJobs` fires a `gpt-image-1` render **for every one of them in parallel** via `Future.wait` — before the user has seen a single result.

Your log shows this batch running **twice**: six jobs at 14:11 and six more at 14:17. Two full onboarding passes, 12 premium image generations, ~$0.84.

Three compounding defects:

**B1 — Speculative generation.** Six outfit cards are rendered whether or not the user looks at them. This is the same mistake round 1 fixed for wardrobe items, repeated at the outfit level.

**B2 — The submit button is not locked during the expensive work.** `busy` comes from `widget.submitting` ([style_step.dart:158](../../wearwise-mobile/lib/features/onboarding/presentation/steps/style_step.dart#L158)), which is only set by `submitStyle()` — and that runs **after** `_createOutfitVisualJobs` has already fired all six calls. During the ~30-60s window where the money is being spent, the button stays enabled. A second tap costs another six images.

**B3 — Idempotency keys are random, so they do nothing.** [`ai_job_repository.dart:84`](../../wearwise-mobile/lib/features/ai/data/ai_job_repository.dart#L84):

```dart
options: Options(headers: {'Idempotency-Key': _uuid.v4()}),
```

A fresh UUID per call means every retry is a brand-new job by definition. The header is present, looks correct, and provides **zero** protection. It must be derived from the request's content (e.g. a hash of `job_type` + sorted `item_id`s + `title`) so a repeat of the same logical request collapses.

---

## 5. Root cause C — the server has no dedup and no quota

`AiJobService::create()` ([AiJobService.php:180-230](../../wearwise-api/app/Modules/Ai/Services/AiJobService.php#L180-L230)) is the only path the app uses, and it performs **no duplicate check at all** — unlike `queueBackgroundRemovalFollowUp`, which does check `source_background_removal_job_id`.

And the quota gate is an empty stub ([AiJobCreationGate.php:33-36](../../wearwise-api/app/Modules/Ai/Services/AiJobCreationGate.php#L33-L36)):

```php
/**
 * Placeholder for Phase 4 quota and entitlement enforcement.
 */
protected function ensureEntitled(...): void
{
    //
}
```

Any client — a retry loop, a double tap, a bug — can queue unlimited paid image generations. **Nothing server-side stops it.** This is why round 1's savings could be undone by a single UI defect.

---

## 6. Fixes, in order

### Fix 1 — Downgrade outfit visuals (~$0.84 → ~$0.08 on this session)

```python
# application/outfit_visual_generation_service.py:9
OUTFIT_VISUAL_MODEL_NAME = "gpt-image-1-mini"

# models/openai_image.py:31-35
size: str = "1024x1024"     # was 1024x1536
quality: str = "low"        # was medium
```

Also update the three `WEARWISE_OPENAI_OUTFIT_VISUAL_*` defaults in [run_worker.py:181-197](../src/wearwise_ai/workers/run_worker.py#L181-L197). Ten minutes of work, ~90% of the remaining bill.

### Fix 2 — Generate outfit visuals lazily

Drop `count: 6` renders to **one** (the card actually on screen), and generate the rest on demand as the user swipes. If all six must exist eventually, queue them one at a time behind user interaction rather than in a `Future.wait` burst.

### Fix 3 — Lock the button before spending, not after

Set a local `_generating` flag at the top of `_submit()` and clear it in a `finally`, so it covers `_createOutfitVisualJobs` — not just `submitStyle`.

### Fix 4 — Make idempotency keys deterministic

```dart
final key = sha256.convert(utf8.encode(
  '$jobType|${itemIds.toList()..sort()}|$title',
)).toString();
options: Options(headers: {'Idempotency-Key': key}),
```

Then **honour it server-side** in `AiJobService::create()` — store the key on the job row with a unique index per account, and return the existing job on a repeat instead of creating a new one.

### Fix 5 — Implement the quota gate

Fill in `ensureEntitled()` with a per-account daily cap on paid job types (`outfit_visual_generation`, `wardrobe_item_visual_generation`). This is the backstop that makes every future client bug cheap instead of expensive.

### Fix 6 — Move garment analysis to `gpt-5-nano`

The log shows the merged analyser still running on `gpt-5-2025-08-07`. At the observed 803 in / 323 out, that call is **$0.00423** on `gpt-5` versus **$0.00017** on `gpt-5-nano` — 25× cheaper.

The constants were changed in round 1, so the worker is likely still picking up a `WEARWISE_OPENAI_*_MODEL` env var, or `openai_garment_analysis.py` has its own default. Check the running worker's environment:

```bash
grep -n "model_name" src/wearwise_ai/models/openai_garment_analysis.py
env | grep WEARWISE_OPENAI
```

The jeans output in your sample is high quality and well within `nano`'s reach — the schema is doing the work.

---

## 7. Projected

| | This session |
|---|---|
| Observed today | $0.91 |
| \+ Fix 1 (outfit → mini/low/square) | ~$0.12 |
| \+ Fix 2 (6 outfits → 1) | ~$0.04 |
| \+ Fix 6 (analysis → nano) | ~$0.02 |

Fixes 3-5 do not reduce the steady-state number; they stop it from silently doubling again.

---

## 8. On "continuous deduction" — how to prove it to yourself

Add the `response.usage` logging from playbook Step 2 (still outstanding — it is why this round needed manual reconstruction from screenshots). Then:

```bash
# Every paid call, with its cost, as it happens:
grep -E "openai_(input|output)_tokens" worker.log
```

If that output is silent while `claim-next` scrolls, nothing is being spent. Right now you have no way to distinguish a free poll from a paid call in the same log stream, which is exactly what made this look like a background leak.


---

## 9. What shipped

All six fixes are implemented and verified.

| # | Fix | Files |
|---|---|---|
| 1 | Outfit visuals → `gpt-image-1-mini`, `1024x1024`, `quality: low` | [outfit_visual_generation_service.py:9](../src/wearwise_ai/application/outfit_visual_generation_service.py#L9), [openai_image.py:31-35](../src/wearwise_ai/models/openai_image.py#L31-L35), [run_worker.py:181-193](../src/wearwise_ai/workers/run_worker.py#L181-L193) |
| 2 | Eager outfit visuals 6 → 2 (rest on demand) | [style_step.dart](../../wearwise-mobile/lib/features/onboarding/presentation/steps/style_step.dart) |
| 3 | `_generating` re-entry guard covering the paid window | [style_step.dart](../../wearwise-mobile/lib/features/onboarding/presentation/steps/style_step.dart) |
| 4 | Deterministic idempotency keys (content-derived UUIDv5) | [ai_job_repository.dart](../../wearwise-mobile/lib/features/ai/data/ai_job_repository.dart) |
| 5 | Per-account daily caps on paid job types | [AiJobCreationGate.php](../../wearwise-api/app/Modules/Ai/Services/AiJobCreationGate.php), [config/wearwise.php](../../wearwise-api/config/wearwise.php) |
| 6 | `.env` override removed (was forcing `gpt-5` / `detail: high`) | `wearwise-ai/.env` |
| — | Token + cost logging on every paid call | [openai_usage.py](../src/wearwise_ai/models/openai_usage.py) (new) |

### Root cause of Fix 6

The round-1 constants were correct all along — `CLOTHING_CLASSIFICATION_MODEL_NAME` was already `gpt-5-nano`. The worker's `.env` was overriding them:

```
WEARWISE_OPENAI_CLASSIFICATION_MODEL=gpt-5      → gpt-5-nano
WEARWISE_OPENAI_CLASSIFICATION_IMAGE_DETAIL=high → low
```

Because the colour and attribute defaults fall back to the classification env var ([run_worker.py:139-175](../src/wearwise_ai/workers/run_worker.py#L139-L175)), this single line silently forced **every** vision call onto the frontier model. Worth remembering: env overrides outrank code constants, so a "fixed" default can still be inert in a running deployment.

### Cost observability

Every paid call now prints one greppable line:

```
openai_call job_type=garment_analysis model=gpt-5-nano input_tokens=803 \
  cached_input_tokens=0 output_tokens=323 reasoning_tokens=0 estimated_cost=$0.000169
```

To audit spend, and to settle any future "is it leaking?" question in seconds:

```bash
grep openai_call worker.log                                    # every paid call
grep -c openai_call worker.log                                 # how many
grep openai_call worker.log | grep -o 'estimated_cost=\$[0-9.]*'  # what each cost
```

Silence while `claim-next` scrolls now definitively means **zero spend** — the ambiguity that made polling look like a leak is gone.

Prices live in `TEXT_PRICES_PER_MILLION` / `IMAGE_PRICES_PER_MILLION` in [openai_usage.py](../src/wearwise_ai/models/openai_usage.py) and are estimates for observability only; OpenAI billing remains authoritative. Dated snapshots (`gpt-5-2025-08-07`) price as their base model via longest-prefix match, so `gpt-5-nano-*` is never mispriced as `gpt-5`.

### Verification

- `php artisan test --filter=AiJobEndpointsTest` — **20 passed**, 243 assertions
- `PYTHONPATH=src python3 -m unittest discover -s tests` — **95 passed**, 3 skipped
- `tests/test_openai_usage.py` (new) — **9 passed**
- `flutter analyze` on both modified Dart files — **no issues**
- Idempotency key verified stable across key/list reordering, and sensitive to real changes

### Note on Fix 4

The server **already** honoured `Idempotency-Key` correctly via `IdempotencyService` ([AiJobController.php:56](../../wearwise-api/app/Modules/Ai/Http/Controllers/AiJobController.php#L56)), including fingerprint conflict detection, a 24-hour retention window, and concurrent-reservation handling. The random `_uuid.v4()` on the client was defeating a fully working mechanism. Only the Dart change was needed.

### Still outstanding

- **Prompt caching** — `category_hint` is interpolated into the developer message in [openai_garment_analysis.py](../src/wearwise_ai/models/openai_garment_analysis.py), breaking the static cache prefix. Moving it into the user turn would make cached input 10× cheaper. Low priority at nano rates.
- **Batch API** — a further 50% on non-interactive work.
- **Rotate the OpenAI API key.** It is correctly gitignored and was never committed, but it was displayed during this session.
