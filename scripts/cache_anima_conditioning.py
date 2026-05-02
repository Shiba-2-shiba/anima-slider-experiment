from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

from anima_slider import anima_conditioning
from anima_slider import config_util
from anima_slider import prompt_util


def parse_roles(raw: str) -> tuple[anima_conditioning.ConditionRole, ...]:
    roles = tuple(part.strip() for part in raw.split(",") if part.strip())
    invalid = sorted(set(roles) - set(anima_conditioning.CONDITION_ROLES))
    if invalid:
        raise ValueError(f"Invalid role(s): {', '.join(invalid)}")
    return roles


def load_comfy_clip(comfyui_path: str, text_encoder_path: str, device: str):
    root = Path(comfyui_path).resolve()
    if not (root / "comfy" / "sd.py").exists():
        raise FileNotFoundError(f"ComfyUI path does not look valid: {root}")
    sys.path.insert(0, str(root))

    import comfy.sd  # type: ignore

    model_options = {}
    if device == "cpu":
        model_options["load_device"] = torch.device("cpu")
        model_options["offload_device"] = torch.device("cpu")

    return comfy.sd.load_clip(
        ckpt_paths=[str(Path(text_encoder_path).resolve())],
        embedding_directory=None,
        clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION,
        model_options=model_options,
    )


def dry_run_manifest(prompts: list[prompt_util.PromptSettings], roles: tuple[anima_conditioning.ConditionRole, ...], limit: int | None):
    selected = prompts[:limit] if limit is not None else prompts
    records = []
    for index, prompt in enumerate(selected):
        texts = anima_conditioning.prompt_texts(prompt)
        records.append(
            {
                "prompt_index": index,
                "action": prompt.action,
                "guidance_scale": prompt.guidance_scale,
                "width": prompt.width,
                "height": prompt.height,
                "batch_size": prompt.batch_size,
                "roles": {role: {"chars": len(texts[role])} for role in roles},
            }
        )
    return {"prompt_count": len(selected), "roles": list(roles), "records": records}


def main(args):
    config = config_util.load_config_from_yaml(args.config_file)
    config = config_util.apply_config_overrides(
        config,
        {
            "text_encoder": args.text_encoder,
            "comfyui": args.comfyui,
        },
    )
    prompts = prompt_util.load_prompts_from_yaml(args.prompts_file)

    errors = config_util.validate_config_paths(config)
    errors.extend(prompt_util.validate_prompts(prompts, allow_unsafe_age_terms=args.allow_unsafe_age_terms))
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)

    roles = parse_roles(args.roles)
    if args.dry_run:
        manifest = dry_run_manifest(prompts, roles, args.limit)
        print(json.dumps(manifest, indent=2))
        return

    clip = load_comfy_clip(config.model.comfyui_path, config.model.text_encoder_path, args.device)
    records = anima_conditioning.encode_prompt_conditions(clip, prompts, roles=roles, limit=args.limit)
    metadata = {
        "config_file": args.config_file,
        "prompts_file": args.prompts_file,
        "text_encoder_path": config.model.text_encoder_path,
        "comfyui_path": config.model.comfyui_path,
        "device": args.device,
    }
    anima_conditioning.save_condition_cache(args.output, records, metadata=metadata)
    print(json.dumps(anima_conditioning.cache_manifest(records), indent=2))
    print(f"Wrote conditioning cache: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache Anima prompt conditioning with the ComfyUI Anima text encoder.")
    parser.add_argument("--config_file", default="configs/config-anima-slider.yaml")
    parser.add_argument("--prompts_file", default="prompts/prompts-anima-age_slider.yaml")
    parser.add_argument("--output", default="cache/anima_conditioning.pt")
    parser.add_argument("--text_encoder", default=None, help="Override model.text_encoder_path")
    parser.add_argument("--comfyui", default=None, help="Override model.comfyui_path")
    parser.add_argument("--device", choices=["default", "cpu"], default="default")
    parser.add_argument("--roles", default="target,positive,unconditional,neutral")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--allow_unsafe_age_terms",
        action="store_true",
        help="Allow prompt terms such as child/young girl/young boy for explicit nonsexual age-direction experiments.",
    )
    parser.add_argument("--dry_run", action="store_true")
    main(parser.parse_args())
