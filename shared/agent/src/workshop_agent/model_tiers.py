"""Model tier mapping. One place to update when new models ship."""

from __future__ import annotations

import os
from typing import Literal

from .helpers import infer_provider, is_tier
from .types import ModelSpec, Provider

ModelTier = Literal["small", "medium", "large"]

ANTHROPIC_MODEL_TIERS: dict[ModelTier, ModelSpec] = {
    "small": ModelSpec(provider=Provider.ANTHROPIC, model="claude-haiku-4-5"),
    "medium": ModelSpec(provider=Provider.ANTHROPIC, model="claude-sonnet-4-6"),
    "large": ModelSpec(provider=Provider.ANTHROPIC, model="claude-opus-4-6"),
}

OPENAI_MODEL_TIERS: dict[ModelTier, ModelSpec] = {
    "small": ModelSpec(provider=Provider.OPENAI, model="gpt-4o-mini"),
    "medium": ModelSpec(provider=Provider.OPENAI, model="gpt-4o"),
    "large": ModelSpec(provider=Provider.OPENAI, model="gpt-4o"),
}

# Back-compat alias for imports/tests.
MODEL_TIERS = ANTHROPIC_MODEL_TIERS


def _active_tiers() -> dict[ModelTier, ModelSpec]:
    """Pick tier models from whichever provider API key is configured."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return ANTHROPIC_MODEL_TIERS
    if os.environ.get("OPENAI_API_KEY"):
        return OPENAI_MODEL_TIERS
    return ANTHROPIC_MODEL_TIERS


def resolve_model_spec(model: str | None = None, provider: str | None = None) -> ModelSpec:
    model_name = model or "medium"
    if is_tier(model_name):
        return _active_tiers()[model_name].model_copy()  # type: ignore[arg-type]
    return ModelSpec(
        provider=Provider(provider) if provider else Provider(infer_provider(model_name)),
        model=model_name,
    )
