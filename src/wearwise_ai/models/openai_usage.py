from __future__ import annotations

from typing import Any

# USD per 1M tokens, as published by OpenAI. Used only to make spend visible in
# worker logs and job metrics; billing remains authoritative.
TEXT_PRICES_PER_MILLION: dict[str, tuple[float, float, float]] = {
    # model: (input, cached_input, output)
    "gpt-5": (1.25, 0.125, 10.00),
    "gpt-5.1": (1.25, 0.125, 10.00),
    "gpt-5-mini": (0.25, 0.025, 2.00),
    "gpt-5-nano": (0.05, 0.005, 0.40),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-4o-mini": (0.15, 0.075, 0.60),
}
IMAGE_PRICES_PER_MILLION: dict[str, tuple[float, float]] = {
    # model: (image_input, image_output)
    "gpt-image-1": (10.00, 40.00),
    "gpt-image-1-mini": (2.50, 8.00),
}


def usage_metrics(response: Any, *, model_name: str) -> dict[str, int | float | str]:
    """Extract token usage from an OpenAI response and price it.

    Returns an empty dict when the response carries no usage block, so callers
    can merge it into resource metrics unconditionally.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}

    input_tokens = _int(getattr(usage, "input_tokens", 0))
    output_tokens = _int(getattr(usage, "output_tokens", 0))
    cached_tokens = _int(
        getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", 0)
    )
    reasoning_tokens = _int(
        getattr(getattr(usage, "output_tokens_details", None), "reasoning_tokens", 0)
    )

    metrics: dict[str, int | float | str] = {
        "openai_model": model_name,
        "openai_input_tokens": input_tokens,
        "openai_cached_input_tokens": cached_tokens,
        "openai_output_tokens": output_tokens,
        "openai_reasoning_tokens": reasoning_tokens,
    }

    cost = estimate_text_cost_usd(
        model_name=model_name,
        input_tokens=input_tokens,
        cached_input_tokens=cached_tokens,
        output_tokens=output_tokens,
    )
    if cost is not None:
        metrics["openai_estimated_cost_usd"] = cost

    return metrics


def estimate_text_cost_usd(
    *,
    model_name: str,
    input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
) -> float | None:
    prices = _lookup(TEXT_PRICES_PER_MILLION, model_name)
    if prices is None:
        return None

    input_price, cached_price, output_price = prices
    uncached = max(0, input_tokens - cached_input_tokens)
    total = (
        (uncached / 1_000_000) * input_price
        + (cached_input_tokens / 1_000_000) * cached_price
        + (output_tokens / 1_000_000) * output_price
    )
    return round(total, 6)


def image_usage_metrics(response: Any, *, model_name: str) -> dict[str, int | float | str]:
    """Extract token usage from an OpenAI Images response and price it."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"openai_model": model_name}

    input_tokens = _int(getattr(usage, "input_tokens", 0))
    output_tokens = _int(getattr(usage, "output_tokens", 0))

    metrics: dict[str, int | float | str] = {
        "openai_model": model_name,
        "openai_input_tokens": input_tokens,
        "openai_output_tokens": output_tokens,
    }

    prices = _lookup(IMAGE_PRICES_PER_MILLION, model_name)
    if prices is not None:
        image_input_price, image_output_price = prices
        metrics["openai_estimated_cost_usd"] = round(
            (input_tokens / 1_000_000) * image_input_price
            + (output_tokens / 1_000_000) * image_output_price,
            6,
        )

    return metrics


def log_usage(metrics: dict[str, int | float | str], *, job_type: str) -> None:
    """Emit one greppable line per paid OpenAI call.

    Makes spend auditable from the worker log: a quiet log means no spend,
    which plain `claim-next` polling noise cannot otherwise distinguish.
    """
    if not metrics:
        return

    cost = metrics.get("openai_estimated_cost_usd")
    cost_text = f"${cost:.6f}" if isinstance(cost, int | float) else "unpriced"
    print(
        "openai_call"
        f" job_type={job_type}"
        f" model={metrics.get('openai_model', 'unknown')}"
        f" input_tokens={metrics.get('openai_input_tokens', 0)}"
        f" cached_input_tokens={metrics.get('openai_cached_input_tokens', 0)}"
        f" output_tokens={metrics.get('openai_output_tokens', 0)}"
        f" reasoning_tokens={metrics.get('openai_reasoning_tokens', 0)}"
        f" estimated_cost={cost_text}",
        flush=True,
    )


def _lookup(table: dict[str, Any], model_name: str) -> Any | None:
    if model_name in table:
        return table[model_name]

    # Dated snapshots such as "gpt-5-2025-08-07" price as their base model.
    # Match the longest known prefix so "gpt-5-nano-..." never falls back to "gpt-5".
    candidates = [key for key in table if model_name.startswith(key)]
    if not candidates:
        return None

    return table[max(candidates, key=len)]


def _int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int | float):
        return int(value)
    return 0
