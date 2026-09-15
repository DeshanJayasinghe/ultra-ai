# Contributing to WearWise AI

Follow `AGENTS.md` before making any change.

## Workflow

* Create branches from `develop`.
* Use `feature/*`, `fix/*`, `chore/*`, `release/*`, or `hotfix/*`.
* Keep pull requests small and focused.
* Version AI models, pipelines and output schemas.
* Add tests for validation, orchestration, failure handling, metadata and service contracts.
* Update documentation when AI behaviour, model versions, queues, storage, or evaluation rules change.

## Pull Request Requirements

* Explain the AI capability or operational outcome.
* Include model-version and confidence handling where relevant.
* Describe retry, failure and user-correction behaviour.
* Include test or evaluation evidence.
* Do not include secrets, datasets with unapproved personal data, temporary code, or unrelated refactors.
