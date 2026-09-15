from __future__ import annotations

import unittest
from types import SimpleNamespace

from wearwise_ai.models.openai_usage import (
    estimate_text_cost_usd,
    image_usage_metrics,
    usage_metrics,
)


def _response(
    input_tokens: int,
    output_tokens: int,
    *,
    cached: int = 0,
    reasoning: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_tokens_details=SimpleNamespace(cached_tokens=cached),
            output_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
        )
    )


class UsageMetricsTests(unittest.TestCase):
    def test_extracts_token_counts(self) -> None:
        metrics = usage_metrics(
            _response(803, 323, cached=100, reasoning=64),
            model_name="gpt-5-nano",
        )

        self.assertEqual(metrics["openai_input_tokens"], 803)
        self.assertEqual(metrics["openai_output_tokens"], 323)
        self.assertEqual(metrics["openai_cached_input_tokens"], 100)
        self.assertEqual(metrics["openai_reasoning_tokens"], 64)
        self.assertEqual(metrics["openai_model"], "gpt-5-nano")

    def test_returns_empty_without_usage_block(self) -> None:
        self.assertEqual(usage_metrics(SimpleNamespace(), model_name="gpt-5"), {})

    def test_unknown_model_reports_tokens_without_cost(self) -> None:
        metrics = usage_metrics(_response(10, 10), model_name="mystery-model")

        self.assertEqual(metrics["openai_input_tokens"], 10)
        self.assertNotIn("openai_estimated_cost_usd", metrics)


class PricingTests(unittest.TestCase):
    def test_prices_dated_snapshot_as_base_model(self) -> None:
        dated = estimate_text_cost_usd(
            model_name="gpt-5-2025-08-07",
            input_tokens=803,
            cached_input_tokens=0,
            output_tokens=323,
        )
        base = estimate_text_cost_usd(
            model_name="gpt-5",
            input_tokens=803,
            cached_input_tokens=0,
            output_tokens=323,
        )

        self.assertEqual(dated, base)

    def test_nano_snapshot_does_not_price_as_gpt_5(self) -> None:
        """The longest matching prefix wins, so nano never inherits gpt-5 rates."""
        nano = estimate_text_cost_usd(
            model_name="gpt-5-nano-2025-01-01",
            input_tokens=803,
            cached_input_tokens=0,
            output_tokens=323,
        )

        self.assertEqual(
            nano,
            estimate_text_cost_usd(
                model_name="gpt-5-nano",
                input_tokens=803,
                cached_input_tokens=0,
                output_tokens=323,
            ),
        )

    def test_cached_input_is_discounted(self) -> None:
        uncached = estimate_text_cost_usd(
            model_name="gpt-5",
            input_tokens=1000,
            cached_input_tokens=0,
            output_tokens=0,
        )
        cached = estimate_text_cost_usd(
            model_name="gpt-5",
            input_tokens=1000,
            cached_input_tokens=1000,
            output_tokens=0,
        )

        assert uncached is not None and cached is not None
        self.assertLess(cached, uncached)

    def test_nano_is_materially_cheaper_than_gpt_5(self) -> None:
        gpt5 = estimate_text_cost_usd(
            model_name="gpt-5",
            input_tokens=803,
            cached_input_tokens=0,
            output_tokens=323,
        )
        nano = estimate_text_cost_usd(
            model_name="gpt-5-nano",
            input_tokens=803,
            cached_input_tokens=0,
            output_tokens=323,
        )

        assert gpt5 is not None and nano is not None
        self.assertGreater(gpt5 / nano, 20)


class ImageUsageTests(unittest.TestCase):
    def test_image_mini_is_cheaper_than_full(self) -> None:
        full = image_usage_metrics(_response(200, 1568), model_name="gpt-image-1")
        mini = image_usage_metrics(_response(200, 1568), model_name="gpt-image-1-mini")

        self.assertGreater(
            full["openai_estimated_cost_usd"],
            mini["openai_estimated_cost_usd"],
        )

    def test_reports_model_without_usage_block(self) -> None:
        metrics = image_usage_metrics(SimpleNamespace(), model_name="gpt-image-1-mini")

        self.assertEqual(metrics["openai_model"], "gpt-image-1-mini")


if __name__ == "__main__":
    unittest.main()
