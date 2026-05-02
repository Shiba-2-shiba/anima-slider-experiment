from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
from safetensors.torch import save_file

from anima_slider import anima_conditioning
from anima_slider import anima_forward
from anima_slider import config_util
from anima_slider import lora_network
from anima_slider import slider_loss


def freeze_parameters(model: torch.nn.Module):
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def forward_role(patcher, latent, sigma, cond):
    return anima_forward.apply_model_with_condition(patcher, latent, sigma, cond)


def compute_base_outputs(patcher, latent, sigma, record):
    with torch.inference_mode(), lora_network.lora_enabled(patcher.model, False):
        return {
            "positive": forward_role(patcher, latent, sigma, record.conds["positive"]).detach(),
            "unconditional": forward_role(patcher, latent, sigma, record.conds["unconditional"]).detach(),
            "neutral": forward_role(patcher, latent, sigma, record.conds["neutral"]).detach(),
        }


def parse_indices(raw: str | None, fallback: int) -> list[int]:
    if raw is None:
        return [fallback]
    indices = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not indices:
        raise ValueError("No prompt indices were provided")
    return indices


def record_loss(patcher, record, width: int, height: int, seed: int, sigma_fraction: float) -> float:
    device = patcher.load_device
    latent = anima_forward.make_random_latent(
        patcher,
        width=width,
        height=height,
        batch_size=record.batch_size,
        seed=seed,
        device=device,
    )
    sigma = anima_forward.sigma_for_fraction(
        patcher,
        fraction=sigma_fraction,
        batch_size=record.batch_size,
        device=device,
    )
    base = compute_base_outputs(patcher, latent, sigma, record)
    with torch.inference_mode(), lora_network.lora_enabled(patcher.model, True):
        target = forward_role(patcher, latent, sigma, record.conds["target"])
        loss = slider_loss.slider_mse_loss(
            slider_loss.SliderOutputs(
                target=target,
                positive=base["positive"],
                unconditional=base["unconditional"],
                neutral=base["neutral"],
            ),
            guidance_scale=record.guidance_scale,
            action=record.action,
        )
    return float(loss.detach().cpu().item())


def evaluate_records(patcher, records, width: int, height: int, seed: int, sigma_fraction: float) -> dict:
    losses = [
        record_loss(patcher, record, width=width, height=height, seed=seed + record.prompt_index, sigma_fraction=sigma_fraction)
        for record in records
    ]
    return {
        "losses": losses,
        "mean_loss": sum(losses) / len(losses) if losses else None,
    }


def train(args):
    config = config_util.load_config_from_yaml(args.config_file)
    cache = anima_conditioning.load_condition_cache(args.cache)
    all_records = cache["records"]
    prompt_indices = parse_indices(args.prompt_indices, args.prompt_index)
    eval_indices = parse_indices(args.eval_prompt_indices, prompt_indices[0]) if args.eval_prompt_indices else prompt_indices
    train_records = [all_records[index] for index in prompt_indices]
    eval_records = [all_records[index] for index in eval_indices]

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
    device = patcher.load_device
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
            sigma_fraction=args.sigma_fraction,
        )
        for step in range(args.steps):
            record = train_records[step % len(train_records)]
            step_seed = args.seed + step if args.vary_seed else args.seed
            latent = anima_forward.make_random_latent(
                patcher,
                width=width,
                height=height,
                batch_size=record.batch_size,
                seed=step_seed,
                device=device,
            )
            sigma = anima_forward.sigma_for_fraction(
                patcher,
                fraction=args.sigma_fraction,
                batch_size=record.batch_size,
                device=device,
            )
            base = compute_base_outputs(patcher, latent, sigma, record)

            optimizer.zero_grad(set_to_none=True)
            with torch.enable_grad(), lora_network.lora_enabled(patcher.model, True):
                target = forward_role(patcher, latent, sigma, record.conds["target"])
                loss = slider_loss.slider_mse_loss(
                    slider_loss.SliderOutputs(
                        target=target,
                        positive=base["positive"],
                        unconditional=base["unconditional"],
                        neutral=base["neutral"],
                    ),
                    guidance_scale=record.guidance_scale,
                    action=record.action,
                )
                loss.backward()
                optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            step_record = {"step": step + 1, "prompt_index": record.prompt_index, "seed": step_seed, "loss": losses[-1]}
            step_records.append(step_record)
            print(json.dumps(step_record), flush=True)
        final_eval = evaluate_records(
            patcher,
            eval_records,
            width=width,
            height=height,
            seed=args.eval_seed,
            sigma_fraction=args.sigma_fraction,
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
            "note": "Experimental one-step Anima slider LoRA smoke output.",
        },
    )

    report = {
        "device": str(device),
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an experimental Anima LoRA with the direct one-step slider loss.")
    parser.add_argument("--config_file", default="configs/config-anima-slider.yaml")
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
    parser.add_argument("--sigma_fraction", type=float, default=0.5)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--output_lora", default="models/one_step_smoke/anima_one_step_smoke.safetensors")
    parser.add_argument("--output_report", default="reports/one-step-train-smoke.json")
    train(parser.parse_args())
