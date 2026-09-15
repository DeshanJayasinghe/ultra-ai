# WearWise AI

Python AI inference services and model pipelines for WearWise.

## Responsibility

This repository owns AI capabilities for:

* Image validation
* Image normalization
* Background removal
* Clothing classification
* Colour extraction
* Attribute extraction
* Embedding generation
* Duplicate detection
* AI model versioning and evaluation

AI services must remain asynchronous, measurable, replaceable, and user-correction aware.

## Source of Truth

Before implementing, read:

* `AGENTS.md`
* `../wearwise-documentation/AI_ENGINEERING_ALIGNMENT_README.md`
* `../wearwise-documentation/Architecture designs/AI and Machine Learning Architecture.md`
* `../wearwise-documentation/Architecture designs/Image Processing, Media Storage and CDN Architecture.md`
* `../wearwise-documentation/Implementation Blueprint/Phase-wise implementation checklist.md`

## Phase 0 Status

This repository is in Phase 0 governance readiness. Product implementation should start only after FastAPI scaffolding, CI, health endpoints, structured logging, and AI service contract foundations are ready.
