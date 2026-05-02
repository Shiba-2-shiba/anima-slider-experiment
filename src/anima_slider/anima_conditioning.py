from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import torch

from .prompt_util import PromptSettings


ConditionRole = Literal["target", "positive", "unconditional", "neutral"]
CONDITION_ROLES: tuple[ConditionRole, ...] = ("target", "positive", "unconditional", "neutral")


@dataclass
class AnimaCond:
    cond: torch.Tensor
    pooled: torch.Tensor | None
    extra: dict[str, torch.Tensor]


@dataclass
class AnimaPromptConds:
    prompt_index: int
    action: str
    guidance_scale: float
    width: int
    height: int
    batch_size: int
    texts: dict[ConditionRole, str]
    conds: dict[ConditionRole, AnimaCond]


def _as_tensor_dict(raw: dict[str, Any]) -> dict[str, torch.Tensor]:
    tensors = {}
    for key, value in raw.items():
        if torch.is_tensor(value):
            tensors[key] = value.detach().cpu()
    return tensors


def condition_from_comfy_dict(encoded: dict[str, Any]) -> AnimaCond:
    if "cond" not in encoded:
        raise ValueError("encoded conditioning is missing 'cond'")
    cond = encoded["cond"]
    if not torch.is_tensor(cond):
        raise TypeError("encoded conditioning 'cond' must be a torch.Tensor")

    pooled = encoded.get("pooled_output")
    if pooled is not None and not torch.is_tensor(pooled):
        raise TypeError("encoded conditioning 'pooled_output' must be a torch.Tensor when present")

    extra = _as_tensor_dict({key: value for key, value in encoded.items() if key not in {"cond", "pooled_output"}})
    return AnimaCond(cond=cond.detach().cpu(), pooled=pooled.detach().cpu() if pooled is not None else None, extra=extra)


def encode_text_with_comfy_clip(clip: Any, text: str) -> AnimaCond:
    tokens = clip.tokenize(text)
    encoded = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
    return condition_from_comfy_dict(encoded)


def prompt_texts(prompt: PromptSettings) -> dict[ConditionRole, str]:
    return {
        "target": prompt.target,
        "positive": prompt.positive or prompt.target,
        "unconditional": prompt.unconditional,
        "neutral": prompt.neutral or prompt.target,
    }


def encode_prompt_condition_set(
    clip: Any,
    prompt: PromptSettings,
    prompt_index: int,
    roles: Iterable[ConditionRole] = CONDITION_ROLES,
) -> AnimaPromptConds:
    texts = prompt_texts(prompt)
    selected_roles = tuple(roles)
    conds = {role: encode_text_with_comfy_clip(clip, texts[role]) for role in selected_roles}
    return AnimaPromptConds(
        prompt_index=prompt_index,
        action=prompt.action,
        guidance_scale=prompt.guidance_scale,
        width=prompt.width,
        height=prompt.height,
        batch_size=prompt.batch_size,
        texts={role: texts[role] for role in selected_roles},
        conds=conds,
    )


def encode_prompt_conditions(
    clip: Any,
    prompts: list[PromptSettings],
    roles: Iterable[ConditionRole] = CONDITION_ROLES,
    limit: int | None = None,
) -> list[AnimaPromptConds]:
    selected = prompts[:limit] if limit is not None else prompts
    return [
        encode_prompt_condition_set(clip, prompt, prompt_index=index, roles=roles)
        for index, prompt in enumerate(selected)
    ]


def tensor_summary(tensor: torch.Tensor | None) -> dict[str, Any] | None:
    if tensor is None:
        return None
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}


def condition_summary(cond: AnimaCond) -> dict[str, Any]:
    return {
        "cond": tensor_summary(cond.cond),
        "pooled": tensor_summary(cond.pooled),
        "extra": {key: tensor_summary(value) for key, value in sorted(cond.extra.items())},
    }


def cache_manifest(records: list[AnimaPromptConds]) -> dict[str, Any]:
    return {
        "prompt_count": len(records),
        "roles": sorted({role for record in records for role in record.conds}),
        "records": [
            {
                "prompt_index": record.prompt_index,
                "action": record.action,
                "guidance_scale": record.guidance_scale,
                "width": record.width,
                "height": record.height,
                "batch_size": record.batch_size,
                "conditions": {
                    role: condition_summary(cond)
                    for role, cond in sorted(record.conds.items())
                },
            }
            for record in records
        ],
    }


def save_condition_cache(path: str | Path, records: list[AnimaPromptConds], metadata: dict[str, Any] | None = None):
    payload = {
        "format": "anima_slider_condition_cache_v1",
        "metadata": metadata or {},
        "manifest": cache_manifest(records),
        "records": records,
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)


def load_condition_cache(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("format") != "anima_slider_condition_cache_v1":
        raise ValueError(f"Unsupported condition cache format: {payload.get('format')!r}")
    return payload
