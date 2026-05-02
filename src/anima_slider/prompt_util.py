from __future__ import annotations

import re
from typing import Literal

import yaml
from pydantic import BaseModel, root_validator


ACTION_TYPES = Literal["erase", "enhance"]


class PromptSettings(BaseModel):
    target: str
    positive: str | None = None
    unconditional: str = ""
    neutral: str | None = None
    guidance_scale: float = 1.0
    action: ACTION_TYPES = "enhance"
    width: int = 1024
    height: int = 1024
    batch_size: int = 1

    @root_validator(pre=True)
    def fill_prompts(cls, values):
        if "target" not in values:
            raise ValueError("target must be specified")
        if "positive" not in values:
            values["positive"] = values["target"]
        if "unconditional" not in values:
            values["unconditional"] = ""
        if "neutral" not in values:
            values["neutral"] = values["target"]
        return values


def load_prompts_from_yaml(path: str) -> list[PromptSettings]:
    with open(path, "r", encoding="utf-8") as handle:
        prompts = yaml.safe_load(handle)
    if not prompts:
        raise ValueError("prompts file is empty")
    return [PromptSettings(**prompt) for prompt in prompts]


def validate_prompts(prompts: list[PromptSettings], allow_unsafe_age_terms: bool = False) -> list[str]:
    errors = []
    unsafe_age_terms = {
        "child",
        "teen",
        "minor",
        "schoolgirl",
        "schoolboy",
        "young girl",
        "young boy",
        "student",
        "school uniform",
    }
    for index, prompt in enumerate(prompts, start=1):
        if prompt.width <= 0 or prompt.height <= 0:
            errors.append(f"prompt {index}: width/height must be positive")
        if prompt.width * prompt.height > 2_100_000:
            errors.append(
                f"prompt {index}: {prompt.width}x{prompt.height} exceeds Anima preview guidance near 1MP"
            )
        joined = " ".join([prompt.target, prompt.positive or "", prompt.unconditional, prompt.neutral or ""]).lower()
        matched = sorted(
            term
            for term in unsafe_age_terms
            if re.search(rf"(?<![a-z0-9_]){re.escape(term)}(?![a-z0-9_])", joined)
        )
        if matched and not allow_unsafe_age_terms:
            errors.append(f"prompt {index}: unsafe age term(s) found: {', '.join(matched)}")
    return errors
