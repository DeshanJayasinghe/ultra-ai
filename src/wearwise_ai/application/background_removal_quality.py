from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BackgroundRemovalQualityRule:
    min_foreground_coverage: float = 0.08
    max_foreground_coverage: float = 0.82
    max_edge_foreground_ratio: float = 0.42
    min_confidence: float = 0.20


@dataclass(frozen=True, slots=True)
class BackgroundRemovalQuality:
    status: str
    reasons: tuple[str, ...]

    @property
    def needs_review(self) -> bool:
        return self.status == "needs_review"


class BackgroundRemovalQualityAssessor:
    def __init__(self, *, rule: BackgroundRemovalQualityRule | None = None) -> None:
        self._rule = rule or BackgroundRemovalQualityRule()

    def assess(
        self,
        *,
        confidence: float,
        metrics: dict[str, int | float | str] | None,
    ) -> BackgroundRemovalQuality:
        metrics = metrics or {}
        reasons: list[str] = []

        if confidence < self._rule.min_confidence:
            reasons.append("low_confidence")

        foreground_coverage = _optional_float(metrics.get("foreground_coverage_ratio"))
        if foreground_coverage is not None:
            if foreground_coverage < self._rule.min_foreground_coverage:
                reasons.append("foreground_too_small")
            if foreground_coverage > self._rule.max_foreground_coverage:
                reasons.append("foreground_too_large")

        edge_foreground_ratio = _optional_float(metrics.get("edge_foreground_ratio"))
        if (
            edge_foreground_ratio is not None
            and edge_foreground_ratio > self._rule.max_edge_foreground_ratio
        ):
            reasons.append("foreground_touches_edges")

        return BackgroundRemovalQuality(
            status="needs_review" if reasons else "accepted",
            reasons=tuple(reasons),
        )


def _optional_float(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None
