from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Protocol

from wearwise_ai.application.background_removal_service import (
    BackgroundRemovalModelInfo,
    BackgroundRemovalOutput,
    BackgroundRemovalRequest,
)


DEFAULT_BIREFNET_MODULES = (
    "PIL",
    "einops",
    "kornia",
    "numpy",
    "timm",
    "torch",
    "torchvision",
    "transformers",
)


class ModelUnavailableError(RuntimeError):
    """Raised when a configured model adapter cannot run in this environment."""


@dataclass(frozen=True, slots=True)
class BiRefNetConfig:
    model_name: str = "BiRefNet"
    model_version: str = "unconfigured"
    model_family: str = "birefnet"
    output_content_type: str = "image/webp"
    required_modules: tuple[str, ...] = DEFAULT_BIREFNET_MODULES


@dataclass(frozen=True, slots=True)
class BiRefNetDependencyStatus:
    required_modules: tuple[str, ...]
    missing_modules: tuple[str, ...]

    @property
    def is_ready(self) -> bool:
        return len(self.missing_modules) == 0


class BiRefNetRunner(Protocol):
    def remove_background(
        self,
        request: BackgroundRemovalRequest,
        config: BiRefNetConfig,
    ) -> BackgroundRemovalOutput: ...


def check_birefnet_dependencies(
    required_modules: tuple[str, ...] = DEFAULT_BIREFNET_MODULES,
) -> BiRefNetDependencyStatus:
    missing = tuple(module_name for module_name in required_modules if find_spec(module_name) is None)
    return BiRefNetDependencyStatus(
        required_modules=required_modules,
        missing_modules=missing,
    )


class BiRefNetBackgroundRemover:
    """BackgroundRemover adapter for BiRefNet-style garment cutout inference.

    The default adapter intentionally performs dependency checks only. The heavy
    model runner is injected so tests, workers, and future GPU deployments can
    share the same application contract without importing ML packages at module
    import time.
    """

    def __init__(
        self,
        *,
        config: BiRefNetConfig | None = None,
        runner: BiRefNetRunner | None = None,
    ) -> None:
        self._config = config or BiRefNetConfig()
        self._runner = runner

    @property
    def model_info(self) -> BackgroundRemovalModelInfo:
        return BackgroundRemovalModelInfo(
            family=self._config.model_family,
            name=self._config.model_name,
            version=self._config.model_version,
        )

    def remove_background(self, request: BackgroundRemovalRequest) -> BackgroundRemovalOutput:
        if self._runner is None:
            status = check_birefnet_dependencies(self._config.required_modules)
            if not status.is_ready:
                missing = ", ".join(status.missing_modules)
                raise ModelUnavailableError(
                    "BiRefNet background removal dependencies are not installed: "
                    f"{missing}. Install the ML runtime extra and configure a model runner."
                )

            raise ModelUnavailableError(
                "BiRefNet dependencies are installed, but no model runner is configured."
            )

        output = self._runner.remove_background(request, self._config)

        if output.content_type != self._config.output_content_type:
            raise ValueError(
                "BiRefNet runner returned an unexpected content type: "
                f"{output.content_type!r}; expected {self._config.output_content_type!r}."
            )

        return output
