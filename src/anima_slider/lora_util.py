from __future__ import annotations

import fnmatch
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


RAW_MODEL_PREFIX = "model."
MODEL_WEIGHT_PREFIX = "diffusion_model."


@dataclass(frozen=True)
class LoraTarget:
    raw_weight_key: str
    comfy_weight_key: str
    lora_key: str
    shape: tuple[int, ...]

    @property
    def out_dim(self) -> int:
        return self.shape[0]

    @property
    def in_dim(self) -> int:
        total = 1
        for dim in self.shape[1:]:
            total *= dim
        return total


@dataclass(frozen=True)
class TargetInspection:
    targets: list[LoraTarget]
    total_keys: int
    weight_keys: int
    include_matched_keys: int
    excluded_keys: int
    skipped_non_2d_keys: int
    skipped_non_diffusion_model_keys: int
    sample_include_misses: list[str]
    sample_excluded: list[str]
    sample_non_2d: list[str]
    sample_non_diffusion_model: list[str]


def normalize_weight_key(raw_key: str) -> str:
    key = raw_key
    if key.startswith(RAW_MODEL_PREFIX):
        key = key[len(RAW_MODEL_PREFIX):]
    return key


def lora_key_for_weight(raw_key: str) -> str:
    key = normalize_weight_key(raw_key)
    if not key.endswith(".weight"):
        raise ValueError(f"Expected a .weight key, got: {raw_key}")
    return key[:-len(".weight")]


def _matches_any(values: list[str], patterns: list[str]) -> bool:
    return any(
        fnmatch.fnmatchcase(value, pattern)
        for value in values
        for pattern in patterns
    )


def inspect_safetensors_targets(
    checkpoint_path: str,
    include_patterns: list[str],
    exclude_patterns: list[str],
) -> list[LoraTarget]:
    return inspect_safetensors_targets_with_stats(
        checkpoint_path,
        include_patterns,
        exclude_patterns,
    ).targets


def _append_sample(samples: list[str], value: str, limit: int):
    if len(samples) < limit:
        samples.append(value)


def inspect_safetensors_targets_with_stats(
    checkpoint_path: str,
    include_patterns: list[str],
    exclude_patterns: list[str],
    sample_limit: int = 20,
) -> TargetInspection:
    targets = []
    total_keys = 0
    weight_keys = 0
    include_matched_keys = 0
    excluded_keys = 0
    skipped_non_2d_keys = 0
    skipped_non_diffusion_model_keys = 0
    sample_include_misses: list[str] = []
    sample_excluded: list[str] = []
    sample_non_2d: list[str] = []
    sample_non_diffusion_model: list[str] = []

    with safe_open(checkpoint_path, framework="pt", device="cpu") as handle:
        for raw_key in handle.keys():
            total_keys += 1
            if not raw_key.endswith(".weight"):
                continue
            weight_keys += 1
            comfy_key = normalize_weight_key(raw_key)
            lora_key = lora_key_for_weight(raw_key)
            raw_key_without_weight = raw_key[:-len(".weight")]
            comfy_key_without_weight = comfy_key[:-len(".weight")]
            match_values = [
                raw_key,
                raw_key_without_weight,
                comfy_key,
                comfy_key_without_weight,
                lora_key,
            ]

            if not _matches_any(match_values, include_patterns):
                _append_sample(sample_include_misses, raw_key, sample_limit)
                continue
            include_matched_keys += 1
            if _matches_any(match_values, exclude_patterns):
                excluded_keys += 1
                _append_sample(sample_excluded, raw_key, sample_limit)
                continue

            shape = tuple(handle.get_tensor(raw_key).shape)
            if len(shape) != 2:
                skipped_non_2d_keys += 1
                _append_sample(sample_non_2d, f"{raw_key} shape={shape}", sample_limit)
                continue
            if not comfy_key.startswith(MODEL_WEIGHT_PREFIX):
                skipped_non_diffusion_model_keys += 1
                _append_sample(sample_non_diffusion_model, raw_key, sample_limit)
                continue
            targets.append(
                LoraTarget(
                    raw_weight_key=raw_key,
                    comfy_weight_key=comfy_key,
                    lora_key=lora_key,
                    shape=shape,
                )
            )
    return TargetInspection(
        targets=targets,
        total_keys=total_keys,
        weight_keys=weight_keys,
        include_matched_keys=include_matched_keys,
        excluded_keys=excluded_keys,
        skipped_non_2d_keys=skipped_non_2d_keys,
        skipped_non_diffusion_model_keys=skipped_non_diffusion_model_keys,
        sample_include_misses=sample_include_misses,
        sample_excluded=sample_excluded,
        sample_non_2d=sample_non_2d,
        sample_non_diffusion_model=sample_non_diffusion_model,
    )


def classify_target(target: LoraTarget) -> str:
    key = target.lora_key
    if ".self_attn." in key:
        return "self_attn"
    if ".cross_attn." in key:
        return "cross_attn"
    if ".mlp." in key:
        return "mlp"
    return "other"


def summarize_targets(targets: list[LoraTarget]) -> dict[str, Counter]:
    return {
        "by_module": Counter(classify_target(target) for target in targets),
        "by_shape": Counter("x".join(str(dim) for dim in target.shape) for target in targets),
    }


def target_to_record(target: LoraTarget) -> dict[str, object]:
    return {
        "raw_weight_key": target.raw_weight_key,
        "comfy_weight_key": target.comfy_weight_key,
        "lora_key": target.lora_key,
        "module": classify_target(target),
        "shape": list(target.shape),
        "out_dim": target.out_dim,
        "in_dim": target.in_dim,
    }


def build_target_report(inspection: TargetInspection, rank: int) -> dict[str, object]:
    summary = summarize_targets(inspection.targets)
    return {
        "counts": {
            "total_keys": inspection.total_keys,
            "weight_keys": inspection.weight_keys,
            "include_matched_keys": inspection.include_matched_keys,
            "excluded_keys": inspection.excluded_keys,
            "skipped_non_2d_keys": inspection.skipped_non_2d_keys,
            "skipped_non_diffusion_model_keys": inspection.skipped_non_diffusion_model_keys,
            "accepted_targets": len(inspection.targets),
            "estimated_lora_parameters": estimate_lora_parameters(inspection.targets, rank),
        },
        "by_module": dict(sorted(summary["by_module"].items())),
        "by_shape": dict(sorted(summary["by_shape"].items())),
        "samples": {
            "include_misses": inspection.sample_include_misses,
            "excluded": inspection.sample_excluded,
            "non_2d": inspection.sample_non_2d,
            "non_diffusion_model": inspection.sample_non_diffusion_model,
        },
        "targets": [target_to_record(target) for target in inspection.targets],
    }


def save_target_report_json(output_path: str, inspection: TargetInspection, rank: int):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(build_target_report(inspection, rank), handle, indent=2)


def save_target_report_csv(output_path: str, targets: list[LoraTarget]):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["raw_weight_key", "comfy_weight_key", "lora_key", "module", "shape", "out_dim", "in_dim"]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for target in targets:
            record = target_to_record(target)
            record["shape"] = "x".join(str(dim) for dim in target.shape)
            writer.writerow(record)


def estimate_lora_parameters(targets: list[LoraTarget], rank: int) -> int:
    return sum((target.out_dim + target.in_dim) * rank for target in targets)


def create_zero_lora_state_dict(
    targets: list[LoraTarget],
    rank: int,
    alpha: float,
) -> dict[str, torch.Tensor]:
    state = {}
    for target in targets:
        down = torch.zeros((rank, target.in_dim), dtype=torch.float32)
        up = torch.zeros((target.out_dim, rank), dtype=torch.float32)
        state[f"{target.lora_key}.lora_down.weight"] = down
        state[f"{target.lora_key}.lora_up.weight"] = up
        state[f"{target.lora_key}.alpha"] = torch.tensor(alpha, dtype=torch.float32)
    return state


def save_zero_lora(
    output_path: str,
    targets: list[LoraTarget],
    rank: int,
    alpha: float,
    metadata: dict[str, str] | None = None,
):
    save_file(create_zero_lora_state_dict(targets, rank, alpha), output_path, metadata=metadata or {})
