from __future__ import annotations

import argparse
import json

import torch

from anima_slider import anima_conditioning
from anima_slider import anima_forward
from anima_slider import config_util
from anima_slider import slider_loss


def main(args):
    config = config_util.load_config_from_yaml(args.config_file)
    cache = anima_conditioning.load_condition_cache(args.cache)
    records = cache["records"]
    record = records[args.prompt_index]

    patcher = anima_forward.load_comfy_diffusion_model(
        config.model.comfyui_path,
        config.model.diffusion_model_path,
    )
    device = patcher.load_device
    width = args.width or record.width
    height = args.height or record.height
    latent = anima_forward.make_random_latent(
        patcher,
        width=width,
        height=height,
        batch_size=record.batch_size,
        seed=args.seed,
        device=device,
    )
    sigma = anima_forward.sigma_for_fraction(
        patcher,
        fraction=args.sigma_fraction,
        batch_size=record.batch_size,
        device=device,
    )

    try:
        patcher.pre_run()
        roles = tuple(role.strip() for role in args.roles.split(",") if role.strip())
        invalid = sorted(set(roles) - set(anima_conditioning.CONDITION_ROLES))
        if invalid:
            raise ValueError(f"Invalid role(s): {', '.join(invalid)}")
        with torch.inference_mode():
            outputs = {
                role: anima_forward.apply_model_with_condition(patcher, latent, sigma, record.conds[role])
                for role in roles
            }
            loss = None
            if set(outputs) == set(anima_conditioning.CONDITION_ROLES):
                loss = slider_loss.slider_mse_loss(
                    slider_loss.SliderOutputs(**outputs),
                    guidance_scale=record.guidance_scale,
                    action=record.action,
                )
    finally:
        patcher.cleanup()

    report = {
        "prompt_index": record.prompt_index,
        "action": record.action,
        "guidance_scale": record.guidance_scale,
        "device": str(device),
        "width": width,
        "height": height,
        "latent_shape": list(latent.shape),
        "sigma": float(sigma[0].detach().cpu().item()),
        "loss": float(loss.detach().cpu().item()) if loss is not None else None,
        "outputs": {role: slider_loss.tensor_stats(output.detach().cpu()) for role, output in outputs.items()},
    }
    print(json.dumps(report, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one direct Anima slider-loss step against a cached conditioning set.")
    parser.add_argument("--config_file", default="configs/config-anima-slider.yaml")
    parser.add_argument("--cache", default="cache/anima-age-conditioning.pt")
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=961218314523996)
    parser.add_argument("--sigma_fraction", type=float, default=0.5)
    parser.add_argument("--roles", default="target,positive,unconditional,neutral")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--output", default=None)
    main(parser.parse_args())
