from __future__ import annotations

import argparse
from os import getenv
from uuid import uuid4

from wearwise_ai.application.background_removal_service import BackgroundRemovalService
from wearwise_ai.application.preprocessing_service import PreprocessingService
from wearwise_ai.core.env import load_environment_file
from wearwise_ai.infrastructure.internal_api_client import (
    InternalApiConfig,
    LaravelInternalApiClient,
)
from wearwise_ai.models.background_removal.birefnet_adapter import (
    BiRefNetBackgroundRemover,
    BiRefNetConfig,
)
from wearwise_ai.models.background_removal.huggingface_runner import (
    HuggingFaceBiRefNetRunner,
    HuggingFaceBiRefNetRunnerConfig,
)
from wearwise_ai.storage.supabase_artifact_store import (
    SupabaseArtifactStore,
    SupabaseArtifactStoreConfig,
)
from wearwise_ai.workflows.background_removal_job import BackgroundRemovalJobWorkflow


def main() -> None:
    load_environment_file()

    parser = argparse.ArgumentParser(description="Run one WearWise background-removal AI job.")
    parser.add_argument("--job-id", default=getenv("WEARWISE_AI_JOB_ID"), required=False)
    parser.add_argument("--worker-id", default=getenv("WEARWISE_AI_WORKER_ID", "local-ai-worker"))
    parser.add_argument("--request-id", default=getenv("WEARWISE_REQUEST_ID"))
    parser.add_argument(
        "--api-base-url",
        default=getenv("WEARWISE_API_INTERNAL_BASE_URL", "http://127.0.0.1:8000"),
    )
    parser.add_argument(
        "--api-timeout-seconds",
        type=float,
        default=float(getenv("WEARWISE_API_INTERNAL_TIMEOUT_SECONDS", "60")),
    )
    parser.add_argument(
        "--api-max-attempts",
        type=int,
        default=int(getenv("WEARWISE_API_INTERNAL_MAX_ATTEMPTS", "3")),
    )
    parser.add_argument(
        "--api-retry-backoff-seconds",
        type=float,
        default=float(getenv("WEARWISE_API_INTERNAL_RETRY_BACKOFF_SECONDS", "0.5")),
    )
    parser.add_argument("--internal-token", default=getenv("WEARWISE_AI_INTERNAL_TOKEN"))
    parser.add_argument("--supabase-url", default=getenv("SUPABASE_URL"))
    parser.add_argument("--supabase-service-role-key", default=getenv("SUPABASE_SERVICE_ROLE_KEY"))
    parser.add_argument(
        "--supabase-bucket",
        default=getenv("WEARWISE_MEDIA_STORAGE_BUCKET", "wearwise-media"),
    )
    parser.add_argument(
        "--birefnet-model-id",
        default=getenv("WEARWISE_BIREFNET_MODEL_ID", "ZhengPeng7/BiRefNet"),
    )
    parser.add_argument("--birefnet-revision", default=getenv("WEARWISE_BIREFNET_MODEL_REVISION"))
    parser.add_argument("--device", default=getenv("WEARWISE_BIREFNET_DEVICE", "cpu"))
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        default=getenv("WEARWISE_BIREFNET_LOCAL_FILES_ONLY") == "1",
    )
    args = parser.parse_args()

    if not args.job_id:
        raise SystemExit("--job-id or WEARWISE_AI_JOB_ID is required.")
    if not args.internal_token:
        raise SystemExit("--internal-token or WEARWISE_AI_INTERNAL_TOKEN is required.")
    if not args.supabase_url:
        raise SystemExit("--supabase-url or SUPABASE_URL is required.")
    if not args.supabase_service_role_key:
        raise SystemExit("--supabase-service-role-key or SUPABASE_SERVICE_ROLE_KEY is required.")

    runner_config = HuggingFaceBiRefNetRunnerConfig(
        model_id=args.birefnet_model_id,
        revision=args.birefnet_revision,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    remover_config = BiRefNetConfig(
        model_name=args.birefnet_model_id,
        model_version=args.birefnet_revision or "unrevisioned",
    )
    workflow = BackgroundRemovalJobWorkflow(
        internal_api=LaravelInternalApiClient(
            InternalApiConfig(
                base_url=args.api_base_url,
                bearer_token=args.internal_token,
                timeout_seconds=args.api_timeout_seconds,
                max_attempts=args.api_max_attempts,
                retry_backoff_seconds=args.api_retry_backoff_seconds,
            )
        ),
        preprocessing_service=PreprocessingService(),
        background_removal_service=BackgroundRemovalService(
            remover=BiRefNetBackgroundRemover(
                config=remover_config,
                runner=HuggingFaceBiRefNetRunner(runner_config=runner_config),
            ),
            artifact_store=SupabaseArtifactStore(
                SupabaseArtifactStoreConfig(
                    base_url=args.supabase_url,
                    service_role_key=args.supabase_service_role_key,
                    bucket=args.supabase_bucket,
                )
            ),
        ),
    )
    outcome = workflow.run(
        job_id=args.job_id,
        worker_id=args.worker_id,
        request_id=args.request_id or str(uuid4()),
    )
    print(
        "background removal complete: "
        f"job_id={outcome.job_id} asset_id={outcome.asset_id} "
        f"preprocessing_status={outcome.preprocessing_status} "
        f"result_status={outcome.result.status if outcome.result else 'none'}"
    )


if __name__ == "__main__":
    main()
