from __future__ import annotations

import unittest

from wearwise_ai.application.background_removal_quality import (
    BackgroundRemovalQualityAssessor,
)


class BackgroundRemovalQualityTests(unittest.TestCase):
    def test_accepts_balanced_mask_metrics(self) -> None:
        quality = BackgroundRemovalQualityAssessor().assess(
            confidence=0.72,
            metrics={
                "foreground_coverage_ratio": 0.45,
                "edge_foreground_ratio": 0.08,
            },
        )

        self.assertEqual(quality.status, "accepted")
        self.assertEqual(quality.reasons, ())

    def test_flags_tiny_foreground_for_review(self) -> None:
        quality = BackgroundRemovalQualityAssessor().assess(
            confidence=0.72,
            metrics={
                "foreground_coverage_ratio": 0.02,
                "edge_foreground_ratio": 0.01,
            },
        )

        self.assertEqual(quality.status, "needs_review")
        self.assertEqual(quality.reasons, ("foreground_too_small",))

    def test_flags_edge_touch_and_low_confidence(self) -> None:
        quality = BackgroundRemovalQualityAssessor().assess(
            confidence=0.12,
            metrics={
                "foreground_coverage_ratio": 0.52,
                "edge_foreground_ratio": 0.91,
            },
        )

        self.assertEqual(quality.status, "needs_review")
        self.assertEqual(quality.reasons, ("low_confidence", "foreground_touches_edges"))


if __name__ == "__main__":
    unittest.main()
