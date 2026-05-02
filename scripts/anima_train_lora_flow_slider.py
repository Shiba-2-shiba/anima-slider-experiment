from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from anima_slider import anima_conditioning
from anima_slider import anima_forward
from anima_slider import config_util
from anima_slider import lora_network
from anima_slider import slider_loss


def freeze_parameters(model: torch.nn.Module):
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def parse_indices(raw: str | None, fallback: int) -> list[int]:
    if raw is None:
        return [fallback]
    indices = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not indices:
        raise ValueError("No prompt indices were provided")
    return indices


def resolve_model_override(raw_model: str | None, current_model_path: str) -> str | None:
    if raw_model is None:
        return None
    candidate = Path(raw_model)
    if candidate.is_absolute() or len(candidate.parts) > 1:
        return raw_model
    return str(Path(current_model_path).parent / candidate)


def collect_cli_overrides(args, config: config_util.RootConfig) -> dict[str, object]:
    return {
        "model": resolve_model_override(args.model, config.model.diffusion_model_path),
        "text_encoder": args.text_encoder,
        "vae": args.vae,
        "comfyui": args.comfyui,
    }


def validate_anima_resolution(width: int, height: int, step: int = 16):
    if width % step != 0 or height % step != 0:
        raise ValueError(f"Anima training resolution must be divisible by {step}, got {width}x{height}")


def resolve_step_bounds(num_inference_steps: int, min_step_index: int | None, max_step_index: int | None) -> tuple[int, int]:
    if num_inference_steps < 3:
        raise ValueError("num_inference_steps must be at least 3")
    minimum = 1 if min_step_index is None else min_step_index
    maximum = (num_inference_steps - 2) if max_step_index is None else max_step_index
    if minimum < 0:
        raise ValueError("min_step_index must be non-negative")
    if maximum < minimum:
        raise ValueError("max_step_index must be greater than or equal to min_step_index")
    if maximum > num_inference_steps - 2:
        raise ValueError("max_step_index must leave at least one following sigma")
    return minimum, maximum


def time_shift(mu: float, sigma: float, t: torch.Tensor) -> torch.Tensor:
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def flux_shift_mu(image_seq_len: int, base_shift: float = 0.5, max_shift: float = 1.15) -> float:
    slope = (max_shift - base_shift) / (4096 - 256)
    intercept = base_shift - slope * 256
    return slope * image_seq_len + intercept


def nearest_step_index_for_sigma(sigmas: torch.Tensor, sampled_sigma: torch.Tensor, minimum: int, maximum: int) -> int:
    candidates = sigmas[minimum : maximum + 1].detach().float().cpu()
    sigma_value = sampled_sigma.detach().float().cpu().reshape(()).item()
    offset = int((candidates - sigma_value).abs().argmin().item())
    return minimum + offset


def choose_training_step_index(
    step: int,
    seed: int,
    minimum: int,
    maximum: int,
    mode: str,
    sigmas: torch.Tensor | None = None,
    sigmoid_scale: float = 1.0,
    discrete_flow_shift: float = 1.0,
    image_seq_len: int | None = None,
) -> int:
    if mode == "mid":
        return (minimum + maximum) // 2
    generator = torch.Generator(device="cpu").manual_seed(seed + step)
    if mode == "uniform":
        return int(torch.randint(minimum, maximum + 1, (1,), generator=generator).item())
    if mode == "early_late":
        midpoint = (minimum + maximum) // 2
        use_early = bool(torch.randint(0, 2, (1,), generator=generator).item())
        if use_early:
            return int(torch.randint(minimum, midpoint + 1, (1,), generator=generator).item())
        return int(torch.randint(midpoint, maximum + 1, (1,), generator=generator).item())
    if mode in {"sigmoid", "shift", "flux_shift"}:
        if sigmas is None:
            raise ValueError(f"{mode} timestep sampling requires sigmas")
        sampled_sigma = torch.sigmoid(sigmoid_scale * torch.randn((1,), generator=generator))
        if mode == "shift":
            shift = float(discrete_flow_shift)
            sampled_sigma = (sampled_sigma * shift) / (1 + (shift - 1) * sampled_sigma)
        elif mode == "flux_shift":
            mu = flux_shift_mu(image_seq_len or 1024)
            sampled_sigma = time_shift(mu, 1.0, sampled_sigma)
        return nearest_step_index_for_sigma(sigmas, sampled_sigma, minimum, maximum)
    raise ValueError(f"Unsupported timestep_sampling: {mode!r}")


def compute_loss_weight_for_sigma(sigma: torch.Tensor, weighting_scheme: str) -> torch.Tensor:
    sigma = sigma.detach().float()
    if weighting_scheme == "sigma_sqrt":
        return sigma.clamp_min(1e-4) ** -2.0
    if weighting_scheme == "cosmap":
        denominator = 1 - 2 * sigma + 2 * sigma**2
        return 2 / (math.pi * denominator)
    if weighting_scheme == "none" or weighting_scheme is None:
        return torch.ones_like(sigma)
    raise ValueError(f"Unsupported loss_weighting_scheme: {weighting_scheme!r}")


def forward_role(patcher, latent, sigma, cond):
    return anima_forward.apply_model_with_condition(patcher, latent, sigma, cond)


@torch.no_grad()
def target_trajectory_latent(patcher, noise, sigmas, step_index: int, cond):
    latent = anima_forward.scale_noise_for_sigma(patcher, noise, sigmas[0].reshape(1).to(noise.device))
    with lora_network.lora_enabled(patcher.model, False):
        for index in range(step_index):
            sigma = sigmas[index].reshape(1).repeat(noise.shape[0]).to(noise.device)
            next_sigma = sigmas[index + 1].reshape(1).repeat(noise.shape[0]).to(noise.device)
            denoised = forward_role(patcher, latent, sigma, cond)
            latent = anima_forward.euler_step_from_denoised(latent, sigma, next_sigma, denoised)
    return latent.detach()


@torch.no_grad()
def compute_flow_teacher_parts(patcher, latent, sigma, record):
    with lora_network.lora_enabled(patcher.model, False):
        target_base = forward_role(patcher, latent, sigma, record.conds["target"]).detach()
        positive_base = forward_role(patcher, latent, sigma, record.conds["positive"]).detach()
        unconditional_base = forward_role(patcher, latent, sigma, record.conds["unconditional"]).detach()
    return {
        "target_base": target_base,
        "positive_base": positive_base,
        "unconditional_base": unconditional_base,
    }


def teacher_from_parts(teacher_parts: dict[str, torch.Tensor], eta: float, action: str) -> torch.Tensor:
    return slider_loss.flow_slider_teacher(
        teacher_parts["target_base"],
        teacher_parts["positive_base"],
        teacher_parts["unconditional_base"],
        eta=eta,
        action=action,
        normalize_to=teacher_parts["positive_base"],
    )


@torch.no_grad()
def compute_flow_teacher(patcher, latent, sigma, record, eta: float, action: str | None = None):
    teacher_parts = compute_flow_teacher_parts(patcher, latent, sigma, record)
    teacher_parts["teacher"] = teacher_from_parts(teacher_parts, eta=eta, action=action or record.action).detach()
    return teacher_parts


def tensor_norm(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().norm().cpu().item())


def branch_loss_for_teacher(
    patcher,
    latent,
    sigma,
    target_cond,
    teacher: torch.Tensor,
    loss_weight: torch.Tensor,
    train: bool,
    lora_multiplier: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    context = torch.enable_grad() if train else torch.inference_mode()
    with context, lora_network.lora_enabled(patcher.model, True), lora_network.lora_multiplier(patcher.model, lora_multiplier):
        model_pred = forward_role(patcher, latent, sigma, target_cond)
        raw_loss = F.mse_loss(model_pred.float(), teacher.float())
        loss = raw_loss * loss_weight.to(device=raw_loss.device)
    return loss, raw_loss, model_pred


def build_flow_loss_info(
    step_index: int,
    sigmas: torch.Tensor,
    loss_weight: torch.Tensor,
    raw_loss: torch.Tensor,
    model_pred: torch.Tensor,
    teacher_parts: dict[str, torch.Tensor],
    teacher: torch.Tensor,
) -> dict:
    return {
        "step_index": step_index,
        "sigma": float(sigmas[step_index].detach().cpu().item()),
        "loss_weight": float(loss_weight.detach().cpu().item()),
        "raw_loss": float(raw_loss.detach().cpu().item()),
        "teacher_norm": tensor_norm(teacher),
        "model_pred_norm": tensor_norm(model_pred),
        "target_base_norm": tensor_norm(teacher_parts["target_base"]),
        "positive_base_norm": tensor_norm(teacher_parts["positive_base"]),
        "unconditional_base_norm": tensor_norm(teacher_parts["unconditional_base"]),
    }


def flow_loss_for_record(
    patcher,
    record,
    width: int,
    height: int,
    seed: int,
    sigmas: torch.Tensor,
    step_index: int,
    eta: float,
    loss_weighting_scheme: str,
    train: bool,
    direction_loss: str = "enhance_only",
) -> tuple[torch.Tensor, dict]:
    device = patcher.load_device
    noise = anima_forward.make_random_latent(
        patcher,
        width=width,
        height=height,
        batch_size=record.batch_size,
        seed=seed,
        device=device,
    )
    step_index = anima_forward.validate_training_step_index(sigmas, step_index)
    latent = target_trajectory_latent(patcher, noise, sigmas, step_index=step_index, cond=record.conds["target"])
    sigma = sigmas[step_index].reshape(1).repeat(record.batch_size).to(device)
    teacher_parts = compute_flow_teacher_parts(patcher, latent, sigma, record)
    loss_weight = compute_loss_weight_for_sigma(sigma, loss_weighting_scheme).mean().to(device)

    if direction_loss == "enhance_only":
        teacher = teacher_from_parts(teacher_parts, eta=eta, action=record.action).detach()
        loss, raw_loss, model_pred = branch_loss_for_teacher(
            patcher,
            latent,
            sigma,
            record.conds["target"],
            teacher,
            loss_weight,
            train=train,
            lora_multiplier=1.0,
        )
        return loss, build_flow_loss_info(
            step_index,
            sigmas,
            loss_weight,
            raw_loss,
            model_pred,
            teacher_parts,
            teacher,
        )

    if direction_loss == "bidirectional":
        enhance_teacher = teacher_from_parts(teacher_parts, eta=eta, action="enhance").detach()
        erase_teacher = teacher_from_parts(teacher_parts, eta=eta, action="erase").detach()
        enhance_loss, enhance_raw_loss, enhance_model_pred = branch_loss_for_teacher(
            patcher,
            latent,
            sigma,
            record.conds["target"],
            enhance_teacher,
            loss_weight,
            train=train,
            lora_multiplier=1.0,
        )
        erase_loss, erase_raw_loss, erase_model_pred = branch_loss_for_teacher(
            patcher,
            latent,
            sigma,
            record.conds["target"],
            erase_teacher,
            loss_weight,
            train=train,
            lora_multiplier=-1.0,
        )
        loss = (enhance_loss + erase_loss) * 0.5
        raw_loss = (enhance_raw_loss + erase_raw_loss) * 0.5
        info = build_flow_loss_info(
            step_index,
            sigmas,
            loss_weight,
            raw_loss,
            enhance_model_pred,
            teacher_parts,
            enhance_teacher,
        )
        info.update(
            {
                "direction_loss": direction_loss,
                "enhance_raw_loss": float(enhance_raw_loss.detach().cpu().item()),
                "erase_raw_loss": float(erase_raw_loss.detach().cpu().item()),
                "enhance_loss": float(enhance_loss.detach().cpu().item()),
                "erase_loss": float(erase_loss.detach().cpu().item()),
                "enhance_teacher_norm": tensor_norm(enhance_teacher),
                "erase_teacher_norm": tensor_norm(erase_teacher),
                "enhance_model_pred_norm": tensor_norm(enhance_model_pred),
                "erase_model_pred_norm": tensor_norm(erase_model_pred),
            }
        )
        return loss, info

    raise ValueError(f"Unsupported direction_loss: {direction_loss!r}")


def branch_eval_summary(details: list[dict], branch: str) -> dict:
    loss_key = f"{branch}_loss"
    raw_key = f"{branch}_raw_loss"
    values = [detail[loss_key] for detail in details if loss_key in detail]
    raw_values = [detail[raw_key] for detail in details if raw_key in detail]
    return {
        "losses": values,
        "mean_loss": sum(values) / len(values) if values else None,
        "raw_losses": raw_values,
        "mean_raw_loss": sum(raw_values) / len(raw_values) if raw_values else None,
    }


def evaluate_records(
    patcher,
    records,
    width: int,
    height: int,
    seed: int,
    sigmas: torch.Tensor,
    step_indices: list[int],
    eta: float,
    loss_weighting_scheme: str,
    direction_loss: str = "enhance_only",
):
    losses = []
    details = []
    for record in records:
        for step_index in step_indices:
            loss, info = flow_loss_for_record(
                patcher,
                record,
                width=width,
                height=height,
                seed=seed + record.prompt_index + step_index,
                sigmas=sigmas,
                step_index=step_index,
                eta=eta,
                loss_weighting_scheme=loss_weighting_scheme,
                train=False,
                direction_loss=direction_loss,
            )
            value = float(loss.detach().cpu().item())
            losses.append(value)
            details.append({"prompt_index": record.prompt_index, "loss": value, **info})
    result = {
        "losses": losses,
        "mean_loss": sum(losses) / len(losses) if losses else None,
        "details": details,
    }
    if direction_loss == "bidirectional":
        result["branches"] = {
            "enhance": branch_eval_summary(details, "enhance"),
            "erase": branch_eval_summary(details, "erase"),
        }
    return result


def train(args):
    config = config_util.load_config_from_yaml(args.config_file)
    config = config_util.apply_config_overrides(config, collect_cli_overrides(args, config))
    cache = anima_conditioning.load_condition_cache(args.cache)
    all_records = cache["records"]
    prompt_indices = parse_indices(args.prompt_indices, args.prompt_index)
    eval_indices = parse_indices(args.eval_prompt_indices, prompt_indices[0]) if args.eval_prompt_indices else prompt_indices
    train_records = [all_records[index] for index in prompt_indices]
    eval_records = [all_records[index] for index in eval_indices]

    min_step_index, max_step_index = resolve_step_bounds(args.num_inference_steps, args.min_step_index, args.max_step_index)
    eval_step_indices = parse_indices(args.eval_step_indices, (min_step_index + max_step_index) // 2)

    patcher = anima_forward.load_comfy_diffusion_model(config.model.comfyui_path, config.model.diffusion_model_path)
    freeze_parameters(patcher.model)
    injected = lora_network.inject_lora_linear_modules(
        patcher.model,
        config.network.include_patterns,
        config.network.exclude_patterns,
        rank=args.rank or config.network.rank,
        alpha=args.alpha or config.network.alpha,
    )
    params = lora_network.lora_parameters(patcher.model)
    if not params:
        raise RuntimeError("No LoRA parameters were injected")

    optimizer = torch.optim.AdamW(params, lr=args.lr or config.train.lr)
    width = args.width or train_records[0].width
    height = args.height or train_records[0].height
    validate_anima_resolution(width, height)
    device = patcher.load_device
    image_seq_len = (height // 16) * (width // 16)
    sigmas = anima_forward.sigmas_for_steps(
        patcher,
        steps=args.num_inference_steps,
        scheduler_name=args.scheduler_name,
        device=device,
    )
    for index in eval_step_indices:
        anima_forward.validate_training_step_index(sigmas, index)

    losses = []
    step_records = []
    initial_eval = None
    final_eval = None

    try:
        patcher.pre_run()
        initial_eval = evaluate_records(
            patcher,
            eval_records,
            width=width,
            height=height,
            seed=args.eval_seed,
            sigmas=sigmas,
            step_indices=eval_step_indices,
            eta=args.eta,
            loss_weighting_scheme=args.loss_weighting_scheme,
            direction_loss=args.direction_loss,
        )
        for step in range(args.steps):
            record = train_records[step % len(train_records)]
            step_seed = args.seed + step if args.vary_seed else args.seed
            step_index = choose_training_step_index(
                step=step,
                seed=args.seed,
                minimum=min_step_index,
                maximum=max_step_index,
                mode=args.timestep_sampling,
                sigmas=sigmas,
                sigmoid_scale=args.sigmoid_scale,
                discrete_flow_shift=args.discrete_flow_shift,
                image_seq_len=image_seq_len,
            )

            optimizer.zero_grad(set_to_none=True)
            loss, info = flow_loss_for_record(
                patcher,
                record,
                width=width,
                height=height,
                seed=step_seed,
                sigmas=sigmas,
                step_index=step_index,
                eta=args.eta,
                loss_weighting_scheme=args.loss_weighting_scheme,
                train=True,
                direction_loss=args.direction_loss,
            )
            loss.backward()
            optimizer.step()

            loss_value = float(loss.detach().cpu().item())
            losses.append(loss_value)
            step_record = {
                "step": step + 1,
                "prompt_index": record.prompt_index,
                "seed": step_seed,
                "loss": loss_value,
                **info,
            }
            step_records.append(step_record)
            print(json.dumps(step_record), flush=True)
        final_eval = evaluate_records(
            patcher,
            eval_records,
            width=width,
            height=height,
            seed=args.eval_seed,
            sigmas=sigmas,
            step_indices=eval_step_indices,
            eta=args.eta,
            loss_weighting_scheme=args.loss_weighting_scheme,
            direction_loss=args.direction_loss,
        )
    finally:
        patcher.cleanup()

    output_lora = Path(args.output_lora)
    output_lora.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        lora_network.lora_state_dict_from_model(patcher.model),
        str(output_lora),
        metadata={
            "format": "pt",
            "base_model": config.model.diffusion_model_path,
            "trainer_type": "flow_slider",
            "ss_timestep_sampling": args.timestep_sampling,
            "ss_sigmoid_scale": str(args.sigmoid_scale),
            "ss_discrete_flow_shift": str(args.discrete_flow_shift),
            "ss_weighting_scheme": args.loss_weighting_scheme,
            "ss_direction_loss": args.direction_loss,
            "note": "Experimental Anima/Cosmos RFlow FLUX-style slider LoRA.",
        },
    )

    report = {
        "trainer_type": "flow_slider",
        "device": str(device),
        "base_model": config.model.diffusion_model_path,
        "text_encoder": config.model.text_encoder_path,
        "vae": config.model.vae_path,
        "prompt_indices": prompt_indices,
        "eval_prompt_indices": eval_indices,
        "width": width,
        "height": height,
        "latent_shape": list(anima_forward.latent_shape_for_resolution(patcher, width, height, train_records[0].batch_size)),
        "steps": args.steps,
        "lr": args.lr or config.train.lr,
        "rank": args.rank or config.network.rank,
        "alpha": args.alpha or config.network.alpha,
        "injected_targets": len(injected),
        "num_inference_steps": args.num_inference_steps,
        "scheduler_name": args.scheduler_name,
        "timestep_sampling": args.timestep_sampling,
        "sigmoid_scale": args.sigmoid_scale,
        "discrete_flow_shift": args.discrete_flow_shift,
        "loss_weighting_scheme": args.loss_weighting_scheme,
        "direction_loss": args.direction_loss,
        "image_seq_len": image_seq_len,
        "min_step_index": min_step_index,
        "max_step_index": max_step_index,
        "eval_step_indices": eval_step_indices,
        "eta": args.eta,
        "sigmas": [float(value) for value in sigmas.detach().cpu().tolist()],
        "losses": losses,
        "step_records": step_records,
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "initial_eval": initial_eval,
        "final_eval": final_eval,
        "output_lora": str(output_lora),
    }
    if args.output_report:
        output_report = Path(args.output_report)
        output_report.parent.mkdir(parents=True, exist_ok=True)
        output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an experimental Anima LoRA with a FLUX-style flow slider loss.")
    parser.add_argument("--config_file", default="configs/config-anima-slider.yaml")
    parser.add_argument(
        "--model",
        "--diffusion_model",
        dest="model",
        default=None,
        help=(
            "Override model.diffusion_model_path. A bare filename is resolved next to the config's "
            "current diffusion model path."
        ),
    )
    parser.add_argument("--text_encoder", default=None, help="Override model.text_encoder_path")
    parser.add_argument("--vae", default=None, help="Override model.vae_path")
    parser.add_argument("--comfyui", default=None, help="Override model.comfyui_path")
    parser.add_argument("--cache", default="cache/anima-age-conditioning.pt")
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--prompt_indices", default=None, help="Comma-separated prompt indices to cycle during training.")
    parser.add_argument("--eval_prompt_indices", default=None, help="Comma-separated prompt indices for fixed before/after eval.")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--seed", type=int, default=961218314523996)
    parser.add_argument("--eval_seed", type=int, default=961218314523996)
    parser.add_argument("--vary_seed", action="store_true")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--scheduler_name", default="simple")
    parser.add_argument(
        "--timestep_sampling",
        choices=["uniform", "mid", "early_late", "sigmoid", "shift", "flux_shift"],
        default="uniform",
    )
    parser.add_argument("--sigmoid_scale", type=float, default=1.0)
    parser.add_argument("--discrete_flow_shift", type=float, default=1.0)
    parser.add_argument("--loss_weighting_scheme", choices=["none", "sigma_sqrt", "cosmap"], default="none")
    parser.add_argument(
        "--direction_loss",
        choices=["enhance_only", "bidirectional"],
        default="enhance_only",
        help="Use the current enhance-only objective or train +LoRA enhance and -LoRA erase branches.",
    )
    parser.add_argument("--min_step_index", type=int, default=None)
    parser.add_argument("--max_step_index", type=int, default=None)
    parser.add_argument("--eval_step_indices", default=None)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--output_lora", default="models/flow_smoke/anima_flow_smoke.safetensors")
    parser.add_argument("--output_report", default="reports/flow-train-smoke.json")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
