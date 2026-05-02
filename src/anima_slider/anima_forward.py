from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

from .anima_conditioning import AnimaCond


def add_comfyui_to_path(comfyui_path: str | Path):
    root = Path(comfyui_path).resolve()
    if not (root / "comfy" / "sd.py").exists():
        raise FileNotFoundError(f"ComfyUI path does not look valid: {root}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)


def load_comfy_diffusion_model(comfyui_path: str | Path, diffusion_model_path: str | Path):
    add_comfyui_to_path(comfyui_path)

    import comfy.model_management  # type: ignore
    import comfy.sd  # type: ignore

    patcher = comfy.sd.load_diffusion_model(str(Path(diffusion_model_path).resolve()))
    comfy.model_management.load_models_gpu([patcher], force_full_load=True)
    return patcher


def latent_shape_for_resolution(model_patcher: Any, width: int, height: int, batch_size: int = 1) -> tuple[int, ...]:
    latent_format = model_patcher.get_model_object("latent_format")
    channels = latent_format.latent_channels
    latent_h = height // latent_format.spacial_downscale_ratio
    latent_w = width // latent_format.spacial_downscale_ratio
    if latent_format.latent_dimensions == 3:
        return (batch_size, channels, 1, latent_h, latent_w)
    return (batch_size, channels, latent_h, latent_w)


def make_random_latent(
    model_patcher: Any,
    width: int,
    height: int,
    batch_size: int,
    seed: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    shape = latent_shape_for_resolution(model_patcher, width, height, batch_size=batch_size)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    latent = torch.randn(shape, generator=generator, dtype=torch.float32)
    return latent.to(device or model_patcher.load_device)


def sigma_for_fraction(model_patcher: Any, fraction: float, batch_size: int = 1, device: torch.device | str | None = None) -> torch.Tensor:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be between 0 and 1")
    model_sampling = model_patcher.get_model_object("model_sampling")
    sigmas = model_sampling.sigmas
    index = round((len(sigmas) - 1) * fraction)
    sigma = sigmas[index].detach().float().reshape(1).repeat(batch_size)
    return sigma.to(device or model_patcher.load_device)


def simple_sigmas_from_model_sampling(model_sampling: Any, steps: int, device: torch.device | str | None = None) -> torch.Tensor:
    if steps <= 0:
        raise ValueError("steps must be positive")
    source = model_sampling.sigmas.detach().float().cpu()
    stride = len(source) / steps
    sigmas = [float(source[-(1 + int(index * stride))]) for index in range(steps)]
    sigmas.append(0.0)
    return torch.tensor(sigmas, dtype=torch.float32, device=device)


def sigmas_for_steps(
    model_patcher: Any,
    steps: int,
    scheduler_name: str = "simple",
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if scheduler_name != "simple":
        raise ValueError(f"Unsupported scheduler for flow trainer: {scheduler_name!r}")
    model_sampling = model_patcher.get_model_object("model_sampling")
    return simple_sigmas_from_model_sampling(model_sampling, steps=steps, device=device or model_patcher.load_device)


def validate_training_step_index(sigmas: torch.Tensor, step_index: int) -> int:
    last_model_step = len(sigmas) - 2
    if step_index < 0 or step_index > last_model_step:
        raise ValueError(f"step_index must be between 0 and {last_model_step}, got {step_index}")
    if float(sigmas[step_index].detach().cpu().item()) == 0.0:
        raise ValueError("step_index must not select sigma 0")
    return step_index


def reshape_sigma_for_latent(sigma: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
    if sigma.nelement() == 1:
        return sigma.reshape(())
    return sigma.reshape(sigma.shape[:1] + (1,) * (latent.ndim - 1))


def scale_noise_for_sigma(model_patcher: Any, noise: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    model_sampling = model_patcher.get_model_object("model_sampling")
    latent_image = torch.zeros_like(noise)
    sigma = sigma.to(device=noise.device, dtype=torch.float32)
    return model_sampling.noise_scaling(sigma, noise, latent_image, max_denoise=True)


def euler_step_from_denoised(
    latent: torch.Tensor,
    sigma: torch.Tensor,
    next_sigma: torch.Tensor,
    denoised: torch.Tensor,
) -> torch.Tensor:
    sigma = reshape_sigma_for_latent(sigma.to(device=latent.device, dtype=latent.dtype), latent)
    next_sigma = reshape_sigma_for_latent(next_sigma.to(device=latent.device, dtype=latent.dtype), latent)
    derivative = (latent - denoised) / sigma.clamp_min(torch.finfo(latent.dtype).eps)
    return latent + derivative * (next_sigma - sigma)


def condition_to_model_kwargs(cond: AnimaCond, device: torch.device | str, dtype: torch.dtype | None = None) -> dict[str, torch.Tensor]:
    cross_attn_dtype = dtype or cond.cond.dtype
    kwargs = {
        "c_crossattn": cond.cond.to(device=device, dtype=cross_attn_dtype),
    }
    for key, value in cond.extra.items():
        if key == "t5xxl_ids" and value.ndim == 1:
            value = value.unsqueeze(0)
        elif key == "t5xxl_weights" and value.ndim == 1:
            value = value.unsqueeze(0).unsqueeze(-1)

        if value.dtype in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool}:
            kwargs[key] = value.to(device=device)
        else:
            kwargs[key] = value.to(device=device, dtype=cross_attn_dtype)
    return kwargs


def apply_model_with_condition(
    model_patcher: Any,
    latent: torch.Tensor,
    sigma: torch.Tensor,
    cond: AnimaCond,
) -> torch.Tensor:
    model = model_patcher.model
    dtype = model.get_dtype_inference()
    kwargs = condition_to_model_kwargs(cond, device=latent.device, dtype=dtype)
    return model.apply_model(latent, sigma.to(latent.device), **kwargs)
