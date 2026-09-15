from __future__ import annotations

import argparse
from os import getenv
from time import sleep
from uuid import uuid4

from wearwise_ai.application.attribute_extraction_service import AttributeExtractionService
from wearwise_ai.application.background_removal_service import BackgroundRemovalService
from wearwise_ai.application.clothing_classification_service import (
    CLOTHING_CLASSIFICATION_MODEL_NAME,
    ConfidenceRoutedClassifier,
    ClothingClassificationService,
)
from wearwise_ai.application.colour_extraction_service import ColourExtractionService
from wearwise_ai.application.embedding_generation_service import (
    EMBEDDING_MODEL_NAME,
    EmbeddingGenerationService,
)
from wearwise_ai.application.outfit_visual_generation_service import OutfitVisualGenerationService
from wearwise_ai.application.preprocessing_service import PreprocessingService
from wearwise_ai.application.wardrobe_item_visual_generation_service import (
    WardrobeItemVisualGenerationService,
)
from wearwise_ai.core.env import load_environment_file
from wearwise_ai.infrastructure.internal_api_client import (
    InternalApiConfig,
    InternalApiError,
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
from wearwise_ai.models.classification import (
    SigLIPCategoryClassifier,
    TransformersSigLIPZeroShotCategoryRunner,
)
from wearwise_ai.models.embedding.huggingface_runner import (
    HuggingFaceSigLIPRunnerConfig,
    TransformersSigLIPImageEmbedder,
)
from wearwise_ai.models.embedding.siglip_adapter import SigLIPConfig, SigLIPEmbeddingGenerator
from wearwise_ai.models.openai_garment_analysis import (
    GarmentAnalysisAttributeExtractor,
    GarmentAnalysisCategoryClassifier,
    InMemoryGarmentAnalysisCache,
    OpenAIGarmentAnalyser,
    OpenAIGarmentAnalysisConfig,
)
from wearwise_ai.models.openai_image import (
    OpenAIOutfitVisualConfig,
    OpenAIOutfitVisualGenerator,
    OpenAIWardrobeItemVisualConfig,
    OpenAIWardrobeItemVisualGenerator,
)
from wearwise_ai.storage.supabase_artifact_store import (
    SupabaseArtifactStore,
    SupabaseArtifactStoreConfig,
)
from wearwise_ai.workflows.attribute_extraction_job import AttributeExtractionJobWorkflow
from wearwise_ai.workflows.background_removal_job import BackgroundRemovalJobWorkflow
from wearwise_ai.workflows.clothing_classification_job import ClothingClassificationJobWorkflow
from wearwise_ai.workflows.colour_extraction_job import ColourExtractionJobWorkflow
from wearwise_ai.workflows.embedding_generation_job import EmbeddingGenerationJobWorkflow
from wearwise_ai.workflows.outfit_visual_generation_job import OutfitVisualGenerationJobWorkflow
from wearwise_ai.workflows.wardrobe_item_visual_generation_job import (
    WardrobeItemVisualGenerationJobWorkflow,
)

SUPPORTED_JOB_TYPES = [
    "background_removal",
    "colour_extraction",
    "clothing_classification",
    "attribute_extraction",
    "embedding_generation",
    "wardrobe_item_visual_generation",
    "outfit_visual_generation",
]


def main() -> None:
    load_environment_file()

    parser = argparse.ArgumentParser(description="Run a WearWise AI worker loop.")
    parser.add_argument("--worker-id", default=getenv("WEARWISE_AI_WORKER_ID", "local-ai-worker"))
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
    parser.add_argument("--birefnet-device", default=getenv("WEARWISE_BIREFNET_DEVICE", "cpu"))
    parser.add_argument(
        "--siglip-model-id", default=getenv("WEARWISE_SIGLIP_MODEL_ID", EMBEDDING_MODEL_NAME)
    )
    parser.add_argument("--siglip-revision", default=getenv("WEARWISE_SIGLIP_MODEL_REVISION"))
    parser.add_argument("--siglip-device", default=getenv("WEARWISE_SIGLIP_DEVICE", "cpu"))
    parser.add_argument(
        "--openai-classification-model",
        default=getenv("WEARWISE_OPENAI_CLASSIFICATION_MODEL", CLOTHING_CLASSIFICATION_MODEL_NAME),
    )
    parser.add_argument(
        "--openai-classification-model-version",
        default=getenv("WEARWISE_OPENAI_CLASSIFICATION_MODEL_VERSION", "responses-v1"),
    )
    parser.add_argument(
        "--openai-classification-image-detail",
        default=getenv("WEARWISE_OPENAI_CLASSIFICATION_IMAGE_DETAIL", "low"),
        choices=("low", "high", "auto"),
    )
    parser.add_argument(
        "--openai-colour-model",
        default=getenv(
            "WEARWISE_OPENAI_COLOUR_MODEL",
            getenv("WEARWISE_OPENAI_CLASSIFICATION_MODEL", CLOTHING_CLASSIFICATION_MODEL_NAME),
        ),
    )
    parser.add_argument(
        "--openai-colour-model-version",
        default=getenv("WEARWISE_OPENAI_COLOUR_MODEL_VERSION", "responses-v1"),
    )
    parser.add_argument(
        "--openai-colour-image-detail",
        default=getenv(
            "WEARWISE_OPENAI_COLOUR_IMAGE_DETAIL",
            getenv("WEARWISE_OPENAI_CLASSIFICATION_IMAGE_DETAIL", "low"),
        ),
        choices=("low", "high", "auto"),
    )
    parser.add_argument(
        "--openai-attribute-model",
        default=getenv(
            "WEARWISE_OPENAI_ATTRIBUTE_MODEL",
            getenv("WEARWISE_OPENAI_CLASSIFICATION_MODEL", CLOTHING_CLASSIFICATION_MODEL_NAME),
        ),
    )
    parser.add_argument(
        "--openai-attribute-model-version",
        default=getenv("WEARWISE_OPENAI_ATTRIBUTE_MODEL_VERSION", "responses-v1"),
    )
    parser.add_argument(
        "--openai-attribute-image-detail",
        default=getenv(
            "WEARWISE_OPENAI_ATTRIBUTE_IMAGE_DETAIL",
            getenv("WEARWISE_OPENAI_CLASSIFICATION_IMAGE_DETAIL", "low"),
        ),
        choices=("low", "high", "auto"),
    )
    parser.add_argument(
        "--openai-outfit-visual-model",
        default=getenv("WEARWISE_OPENAI_OUTFIT_VISUAL_MODEL", "gpt-image-1-mini"),
    )
    parser.add_argument(
        "--openai-outfit-visual-model-version",
        default=getenv("WEARWISE_OPENAI_OUTFIT_VISUAL_MODEL_VERSION", "images-v1"),
    )
    parser.add_argument(
        "--openai-outfit-visual-size",
        default=getenv("WEARWISE_OPENAI_OUTFIT_VISUAL_SIZE", "1024x1024"),
    )
    parser.add_argument(
        "--openai-outfit-visual-quality",
        default=getenv("WEARWISE_OPENAI_OUTFIT_VISUAL_QUALITY", "low"),
    )
    parser.add_argument(
        "--openai-outfit-visual-output-format",
        default=getenv("WEARWISE_OPENAI_OUTFIT_VISUAL_OUTPUT_FORMAT", "webp"),
        choices=("webp", "png", "jpeg"),
    )
    parser.add_argument(
        "--openai-wardrobe-item-visual-model",
        default=getenv("WEARWISE_OPENAI_WARDROBE_ITEM_VISUAL_MODEL", "gpt-image-1-mini"),
    )
    parser.add_argument(
        "--openai-wardrobe-item-visual-model-version",
        default=getenv("WEARWISE_OPENAI_WARDROBE_ITEM_VISUAL_MODEL_VERSION", "images-v1"),
    )
    parser.add_argument(
        "--openai-wardrobe-item-visual-size",
        default=getenv("WEARWISE_OPENAI_WARDROBE_ITEM_VISUAL_SIZE", "1024x1024"),
    )
    parser.add_argument(
        "--openai-wardrobe-item-visual-quality",
        default=getenv("WEARWISE_OPENAI_WARDROBE_ITEM_VISUAL_QUALITY", "low"),
    )
    parser.add_argument(
        "--openai-wardrobe-item-visual-output-format",
        default=getenv("WEARWISE_OPENAI_WARDROBE_ITEM_VISUAL_OUTPUT_FORMAT", "webp"),
        choices=("webp", "png", "jpeg"),
    )
    parser.add_argument("--poll-interval-seconds", type=float, default=5.0)
    parser.add_argument("--max-jobs", type=int, default=0, help="0 means run until stopped.")
    parser.add_argument("--once", action="store_true", help="Exit after one poll, even when idle.")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        default=_env_bool("WEARWISE_AI_LOCAL_FILES_ONLY"),
    )
    args = parser.parse_args()

    if not args.internal_token:
        raise SystemExit("--internal-token or WEARWISE_AI_INTERNAL_TOKEN is required.")
    if not args.supabase_url:
        raise SystemExit("--supabase-url or SUPABASE_URL is required.")
    if not args.supabase_service_role_key:
        raise SystemExit("--supabase-service-role-key or SUPABASE_SERVICE_ROLE_KEY is required.")

    internal_api = LaravelInternalApiClient(
        InternalApiConfig(
            base_url=args.api_base_url,
            bearer_token=args.internal_token,
            timeout_seconds=args.api_timeout_seconds,
            max_attempts=args.api_max_attempts,
            retry_backoff_seconds=args.api_retry_backoff_seconds,
        )
    )
    artifact_store = SupabaseArtifactStore(
        SupabaseArtifactStoreConfig(
            base_url=args.supabase_url,
            service_role_key=args.supabase_service_role_key,
            bucket=args.supabase_bucket,
        )
    )
    workflows = _build_workflows(
        args=args,
        internal_api=internal_api,
        artifact_store=artifact_store,
    )

    processed = 0
    print(f"wearwise ai worker started: worker_id={args.worker_id}")
    while True:
        request_id = str(uuid4())
        try:
            response = internal_api.claim_next_job(
                request_id=request_id,
                worker_id=args.worker_id,
                job_types=SUPPORTED_JOB_TYPES,
            )
        except InternalApiError as exc:
            print(f"claim-next failed; retrying: {exc}")
            if args.once:
                raise
            sleep(args.poll_interval_seconds)
            continue
        job = response.get("data")
        if not isinstance(job, dict):
            if args.once:
                print("no queued ai job found")
                return
            sleep(args.poll_interval_seconds)
            continue

        job_id = str(job["id"])
        job_type = str(job["job_type"])
        print(f"claimed ai job: job_id={job_id} job_type={job_type}")
        try:
            workflows[job_type].run(
                job_id=job_id,
                worker_id=args.worker_id,
                request_id=request_id,
            )
        except Exception as exc:
            try:
                internal_api.fail_job(
                    job_id,
                    request_id=request_id,
                    payload={
                        "worker_id": args.worker_id,
                        "error_code": "AI_WORKER_UNHANDLED_ERROR",
                        "error_message": str(exc),
                        "status_message": f"{job_type} failed",
                        "retryable": True,
                    },
                )
            except InternalApiError as fail_exc:
                print(f"could not report failed ai job: job_id={job_id} error={fail_exc}")
            print(f"failed ai job: job_id={job_id} job_type={job_type} error={exc}")
        else:
            print(f"completed ai job: job_id={job_id} job_type={job_type}")

        processed += 1
        if args.once or (args.max_jobs > 0 and processed >= args.max_jobs):
            return


def _build_workflows(
    *,
    args: argparse.Namespace,
    internal_api: LaravelInternalApiClient,
    artifact_store: SupabaseArtifactStore,
) -> dict[str, object]:
    birefnet_runner_config = HuggingFaceBiRefNetRunnerConfig(
        model_id=args.birefnet_model_id,
        revision=args.birefnet_revision,
        device=args.birefnet_device,
        local_files_only=args.local_files_only,
    )
    siglip_runner_config = HuggingFaceSigLIPRunnerConfig(
        model_id=args.siglip_model_id,
        revision=args.siglip_revision,
        device=args.siglip_device,
        local_files_only=args.local_files_only,
    )

    siglip_embedding_runner = TransformersSigLIPImageEmbedder(
        runner_config=siglip_runner_config,
    )
    siglip_classifier_runner = TransformersSigLIPZeroShotCategoryRunner(
        runner_config=siglip_runner_config,
    )
    garment_analysis_cache = InMemoryGarmentAnalysisCache()
    garment_analyser = OpenAIGarmentAnalyser(
        config=OpenAIGarmentAnalysisConfig(
            model_name=args.openai_attribute_model,
            model_version=args.openai_attribute_model_version,
            image_detail=args.openai_attribute_image_detail,
        ),
        cache=garment_analysis_cache,
    )

    return {
        "background_removal": BackgroundRemovalJobWorkflow(
            internal_api=internal_api,
            preprocessing_service=PreprocessingService(),
            background_removal_service=BackgroundRemovalService(
                remover=BiRefNetBackgroundRemover(
                    config=BiRefNetConfig(
                        model_name=args.birefnet_model_id,
                        model_version=args.birefnet_revision or "unrevisioned",
                    ),
                    runner=HuggingFaceBiRefNetRunner(runner_config=birefnet_runner_config),
                ),
                artifact_store=artifact_store,
            ),
        ),
        "colour_extraction": ColourExtractionJobWorkflow(
            internal_api=internal_api,
            colour_extraction_service=ColourExtractionService(),
            artifact_reader=artifact_store,
        ),
        "clothing_classification": ClothingClassificationJobWorkflow(
            internal_api=internal_api,
            classification_service=ClothingClassificationService(
                classifier=ConfidenceRoutedClassifier(
                    primary=SigLIPCategoryClassifier(runner=siglip_classifier_runner),
                    fallback=GarmentAnalysisCategoryClassifier(analyser=garment_analyser),
                ),
            ),
            artifact_reader=artifact_store,
        ),
        "attribute_extraction": AttributeExtractionJobWorkflow(
            internal_api=internal_api,
            attribute_extraction_service=AttributeExtractionService(
                extractor=GarmentAnalysisAttributeExtractor(analyser=garment_analyser),
            ),
            artifact_reader=artifact_store,
        ),
        "embedding_generation": EmbeddingGenerationJobWorkflow(
            internal_api=internal_api,
            embedding_generation_service=EmbeddingGenerationService(
                embedder=SigLIPEmbeddingGenerator(
                    config=SigLIPConfig(
                        model_name=args.siglip_model_id,
                        model_version=args.siglip_revision or "unrevisioned",
                    ),
                    runner=siglip_embedding_runner,
                ),
            ),
            artifact_reader=artifact_store,
        ),
        "wardrobe_item_visual_generation": WardrobeItemVisualGenerationJobWorkflow(
            internal_api=internal_api,
            generation_service=WardrobeItemVisualGenerationService(
                generator=OpenAIWardrobeItemVisualGenerator(
                    config=OpenAIWardrobeItemVisualConfig(
                        model_name=args.openai_wardrobe_item_visual_model,
                        model_version=args.openai_wardrobe_item_visual_model_version,
                        size=args.openai_wardrobe_item_visual_size,
                        quality=args.openai_wardrobe_item_visual_quality,
                        output_format=args.openai_wardrobe_item_visual_output_format,
                    )
                ),
                artifact_store=artifact_store,
            ),
            artifact_reader=artifact_store,
        ),
        "outfit_visual_generation": OutfitVisualGenerationJobWorkflow(
            internal_api=internal_api,
            generation_service=OutfitVisualGenerationService(
                generator=OpenAIOutfitVisualGenerator(
                    config=OpenAIOutfitVisualConfig(
                        model_name=args.openai_outfit_visual_model,
                        model_version=args.openai_outfit_visual_model_version,
                        size=args.openai_outfit_visual_size,
                        quality=args.openai_outfit_visual_quality,
                        output_format=args.openai_outfit_visual_output_format,
                    )
                ),
                artifact_store=artifact_store,
            ),
        ),
    }


def _env_bool(name: str) -> bool:
    return getenv(name, "").lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    main()
