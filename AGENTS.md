# WearWise AI Agent Instructions

This repository contains Python AI inference services and model pipelines for WearWise. Follow these instructions for every change in this repository.

## Source of Truth

Read and follow:

* `../wearwise-documentation/Architecture designs/AI and Machine Learning Architecture.md`
* `../wearwise-documentation/Architecture designs/Image Processing, Media Storage and CDN Architecture.md`
* `../wearwise-documentation/Architecture designs/Recommendation, Personalization and Search Architecture.md`
* `../wearwise-documentation/Architecture designs/Security, Privacy and Compliance Architecture.md`
* `../wearwise-documentation/Implementation Blueprint/API implementation order.md`
* `../wearwise-documentation/Implementation Blueprint/Database migration order.md`
* `../wearwise-documentation/Implementation Blueprint/Folder architecture.md`
* `../wearwise-documentation/Implementation Blueprint/Coding standards.md`
* `../wearwise-documentation/AI_ENGINEERING_ALIGNMENT_README.md`

Do not introduce alternative AI service frameworks, model lifecycle patterns, queue contracts, or storage conventions without approval.

## Before Writing Any Code

Before writing any code:

* Think through the entire implementation.
* Break the work into logical steps.
* Consider scalability, security, performance, maintainability, testing, and future extensibility.
* Prefer existing project patterns over creating new ones.
* Reuse existing components whenever possible.
* If you discover an architectural issue, stop and explain it before coding.
* Do not rush into implementation.
* The quality of the architecture is more important than the speed of delivery.
* Every piece of code should be production-ready on the first implementation.

## AI Rules

* Use FastAPI for service APIs.
* Keep AI processing asynchronous.
* Version every model and pipeline.
* Record confidence scores and model versions with results.
* Never overwrite user-confirmed data.
* Support user corrections and feedback loops.
* Keep models replaceable behind stable interfaces.
* Keep storage, queues, inference, validation, and result mapping separated.
* Validate image inputs before processing.
* Treat uploaded media as untrusted.
* Propagate request and correlation IDs.
* Use structured logging without exposing personal data or secrets.
* Design every pipeline for retries, failure states, and measurable performance.

## Testing and Completion

Every feature should include focused tests for validation, pipeline orchestration, failure handling, confidence/result mapping, model-version metadata, and service contracts.

Before completion, run the relevant formatter, type checks, linting, and tests when available. If a check cannot be run, report why.
