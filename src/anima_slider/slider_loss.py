from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SliderOutputs:
    target: torch.Tensor
    positive: torch.Tensor
    unconditional: torch.Tensor
    neutral: torch.Tensor


def desired_slider_output(
    positive: torch.Tensor,
    unconditional: torch.Tensor,
    neutral: torch.Tensor,
    guidance_scale: float,
    action: str,
) -> torch.Tensor:
    direction = positive - unconditional
    if action == "enhance":
        return neutral + guidance_scale * direction
    if action == "erase":
        return neutral - guidance_scale * direction
    raise ValueError(f"Unsupported slider action: {action!r}")


def normalize_like(tensor: torch.Tensor, reference: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    tensor_flat = tensor.float().reshape(tensor.shape[0], -1)
    reference_flat = reference.float().reshape(reference.shape[0], -1)
    tensor_norm = tensor_flat.norm(dim=1).clamp_min(eps)
    reference_norm = reference_flat.norm(dim=1).clamp_min(eps)
    scale = (reference_norm / tensor_norm).reshape([tensor.shape[0]] + [1] * (tensor.ndim - 1))
    return tensor * scale.to(device=tensor.device, dtype=tensor.dtype)


def flow_slider_teacher(
    target: torch.Tensor,
    positive: torch.Tensor,
    unconditional: torch.Tensor,
    eta: float,
    action: str,
    normalize_to: torch.Tensor | None = None,
) -> torch.Tensor:
    direction = positive - unconditional
    if action == "enhance":
        teacher = target + eta * direction
    elif action == "erase":
        teacher = target - eta * direction
    else:
        raise ValueError(f"Unsupported slider action: {action!r}")
    if normalize_to is not None:
        teacher = normalize_like(teacher, normalize_to)
    return teacher


def slider_mse_loss(outputs: SliderOutputs, guidance_scale: float, action: str) -> torch.Tensor:
    desired = desired_slider_output(
        outputs.positive,
        outputs.unconditional,
        outputs.neutral,
        guidance_scale=guidance_scale,
        action=action,
    )
    return F.mse_loss(outputs.target, desired)


def tensor_stats(tensor: torch.Tensor) -> dict[str, float | list[int] | str]:
    detached = tensor.detach().float()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "mean": float(detached.mean().item()),
        "std": float(detached.std().item()),
        "min": float(detached.min().item()),
        "max": float(detached.max().item()),
    }
