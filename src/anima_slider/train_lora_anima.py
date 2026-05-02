from __future__ import annotations

import argparse
from pathlib import Path
import sys

from . import config_util
from . import lora_util
from . import prompt_util


DEBUG_SEPARATOR = "[anima-debug]"


def parse_set_overrides(raw_overrides: list[str] | None) -> dict[str, str]:
    overrides = {}
    for raw_override in raw_overrides or []:
        if "=" not in raw_override:
            raise ValueError(f"Invalid --set override {raw_override!r}. Expected section.key=value.")
        key, value = raw_override.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --set override {raw_override!r}. Empty key.")
        overrides[key] = value.strip()
    return overrides


def collect_cli_overrides(args) -> dict[str, object]:
    overrides = {
        "model": args.model,
        "text_encoder": args.text_encoder,
        "vae": args.vae,
        "comfyui": args.comfyui,
        "rank": args.rank,
        "alpha": args.alpha,
        "iterations": args.iterations,
        "lr": args.lr,
        "precision": args.precision,
        "name": args.name,
        "save_path": args.save_path,
        "preset": args.preset,
    }
    overrides.update(parse_set_overrides(args.set))
    return overrides


def print_summary(config, prompts, targets):
    print("Anima slider config")
    print(f"  diffusion_model: {config.model.diffusion_model_path}")
    print(f"  text_encoder:    {config.model.text_encoder_path}")
    print(f"  vae:             {config.model.vae_path}")
    print(f"  comfyui:         {config.model.comfyui_path}")
    print(f"  prompts:         {len(prompts)}")
    print(f"  lora_targets:    {len(targets)}")
    print(f"  preset:          {config.network.preset or 'custom'}")
    print(f"  rank/alpha:      {config.network.rank}/{config.network.alpha}")
    print(f"  train:           iterations={config.train.iterations}, lr={config.train.lr}, precision={config.train.precision}")
    print(f"  save:            {config.save.path}")
    print(f"  lora_params:     {lora_util.estimate_lora_parameters(targets, config.network.rank):,}")


def _format_bytes(size: int) -> str:
    return f"{size / 1024 / 1024:.2f} MiB"


def _print_path_debug(label: str, raw_path: str):
    path = Path(raw_path)
    if path.is_file():
        print(f"{DEBUG_SEPARATOR} path.{label}: exists=file size={_format_bytes(path.stat().st_size)} path={path}")
    elif path.is_dir():
        print(f"{DEBUG_SEPARATOR} path.{label}: exists=dir path={path}")
    else:
        print(f"{DEBUG_SEPARATOR} path.{label}: missing path={path}")


def _print_samples(label: str, values: list[str]):
    print(f"{DEBUG_SEPARATOR} {label}.count: {len(values)}")
    for value in values:
        print(f"{DEBUG_SEPARATOR} {label}.sample: {value}")


def print_debug_log(config, prompts, inspection, args):
    targets = inspection.targets
    target_summary = lora_util.summarize_targets(targets)

    print(f"{DEBUG_SEPARATOR} config_file: {args.config_file}")
    print(f"{DEBUG_SEPARATOR} prompts_file: {args.prompts_file}")
    print(f"{DEBUG_SEPARATOR} python: {sys.version.split()[0]}")
    try:
        import torch

        print(f"{DEBUG_SEPARATOR} torch: {torch.__version__}")
        print(f"{DEBUG_SEPARATOR} torch.cuda.available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"{DEBUG_SEPARATOR} torch.cuda.device_count: {torch.cuda.device_count()}")
            print(f"{DEBUG_SEPARATOR} torch.cuda.current_device: {torch.cuda.current_device()}")
            print(f"{DEBUG_SEPARATOR} torch.cuda.device_name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    except Exception as exc:
        print(f"{DEBUG_SEPARATOR} torch.inspect_error: {exc}")

    _print_path_debug("diffusion_model", config.model.diffusion_model_path)
    _print_path_debug("text_encoder", config.model.text_encoder_path)
    _print_path_debug("vae", config.model.vae_path)
    _print_path_debug("comfyui", config.model.comfyui_path)

    comfyui_path = Path(config.model.comfyui_path)
    for marker in [
        "comfy/ldm/anima/model.py",
        "comfy/text_encoders/anima.py",
        "comfy/supported_models.py",
        "comfy/sd.py",
    ]:
        marker_path = comfyui_path / marker
        print(f"{DEBUG_SEPARATOR} comfyui_marker.{marker}: {marker_path.exists()}")

    print(f"{DEBUG_SEPARATOR} network.target: {config.network.target}")
    print(f"{DEBUG_SEPARATOR} network.rank: {config.network.rank}")
    print(f"{DEBUG_SEPARATOR} network.alpha: {config.network.alpha}")
    for pattern in config.network.include_patterns:
        print(f"{DEBUG_SEPARATOR} include_pattern: {pattern}")
    for pattern in config.network.exclude_patterns:
        print(f"{DEBUG_SEPARATOR} exclude_pattern: {pattern}")

    print(f"{DEBUG_SEPARATOR} checkpoint.total_keys: {inspection.total_keys}")
    print(f"{DEBUG_SEPARATOR} checkpoint.weight_keys: {inspection.weight_keys}")
    print(f"{DEBUG_SEPARATOR} checkpoint.include_matched_keys: {inspection.include_matched_keys}")
    print(f"{DEBUG_SEPARATOR} checkpoint.excluded_keys: {inspection.excluded_keys}")
    print(f"{DEBUG_SEPARATOR} checkpoint.skipped_non_2d_keys: {inspection.skipped_non_2d_keys}")
    print(f"{DEBUG_SEPARATOR} checkpoint.skipped_non_diffusion_model_keys: {inspection.skipped_non_diffusion_model_keys}")
    print(f"{DEBUG_SEPARATOR} lora.accepted_targets: {len(targets)}")
    print(f"{DEBUG_SEPARATOR} lora.estimated_params: {lora_util.estimate_lora_parameters(targets, config.network.rank)}")
    for module, count in target_summary["by_module"].most_common():
        print(f"{DEBUG_SEPARATOR} lora.module_count.{module}: {count}")
    for shape, count in target_summary["by_shape"].most_common(10):
        print(f"{DEBUG_SEPARATOR} lora.shape_count.{shape}: {count}")

    _print_samples("checkpoint.include_miss", inspection.sample_include_misses)
    _print_samples("checkpoint.excluded", inspection.sample_excluded)
    _print_samples("checkpoint.non_2d", inspection.sample_non_2d)
    _print_samples("checkpoint.non_diffusion_model", inspection.sample_non_diffusion_model)

    resolution_counts = {}
    action_counts = {}
    guidance_values = []
    for prompt in prompts:
        resolution_counts[f"{prompt.width}x{prompt.height}"] = resolution_counts.get(f"{prompt.width}x{prompt.height}", 0) + 1
        action_counts[prompt.action] = action_counts.get(prompt.action, 0) + 1
        guidance_values.append(prompt.guidance_scale)
    print(f"{DEBUG_SEPARATOR} prompts.count: {len(prompts)}")
    print(f"{DEBUG_SEPARATOR} prompts.resolutions: {resolution_counts}")
    print(f"{DEBUG_SEPARATOR} prompts.actions: {action_counts}")
    print(f"{DEBUG_SEPARATOR} prompts.guidance_values: {sorted(set(guidance_values))}")
    for index, prompt in enumerate(prompts[:5], start=1):
        print(
            f"{DEBUG_SEPARATOR} prompt.sample.{index}: "
            f"action={prompt.action} size={prompt.width}x{prompt.height} guidance={prompt.guidance_scale} "
            f"target={prompt.target[:160]}"
        )


def main(args):
    config = config_util.load_config_from_yaml(args.config_file)
    config = config_util.apply_config_overrides(config, collect_cli_overrides(args))
    config = config_util.finalize_training_config(config, append_name_suffix=not args.no_name_suffix)

    prompts_file = args.prompts_file
    prompts = prompt_util.load_prompts_from_yaml(prompts_file)

    errors = []
    errors.extend(config_util.validate_config_paths(config))
    errors.extend(prompt_util.validate_prompts(prompts))

    inspection = lora_util.inspect_safetensors_targets_with_stats(
        config.model.diffusion_model_path,
        config.network.include_patterns,
        config.network.exclude_patterns,
    )
    targets = inspection.targets
    if not targets:
        errors.append("No LoRA targets matched the configured include/exclude patterns")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)

    print_summary(config, prompts, targets)
    if args.debug_log or config.logging.verbose:
        print_debug_log(config, prompts, inspection, args)

    if args.inspect_targets:
        for target in targets[: args.inspect_limit]:
            print(f"  {target.raw_weight_key} -> {target.lora_key} shape={target.shape}")
        if len(targets) > args.inspect_limit:
            print(f"  ... {len(targets) - args.inspect_limit} more")

    if args.target_report_json:
        lora_util.save_target_report_json(args.target_report_json, inspection, config.network.rank)
        print(f"Wrote target report JSON: {args.target_report_json}")
    if args.target_report_csv:
        lora_util.save_target_report_csv(args.target_report_csv, targets)
        print(f"Wrote target report CSV: {args.target_report_csv}")

    if args.write_init_lora:
        output_dir = Path(config.save.path)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{config.save.name}_init.safetensors"
        lora_util.save_zero_lora(
            str(output_path),
            targets,
            rank=config.network.rank,
            alpha=config.network.alpha,
            metadata={
                "format": "pt",
                "base_model": config.model.diffusion_model_path,
                "text_encoder": config.model.text_encoder_path,
                "note": "Zero initialized Anima diffusion-model-only LoRA skeleton; not trained.",
            },
        )
        print(f"Wrote zero initialized LoRA skeleton: {output_path}")

    if args.dry_run or args.write_init_lora:
        return

    raise SystemExit(
        "Anima training loop is not enabled yet. "
        "This separate path currently validates config/prompts and builds ComfyUI-compatible "
        "diffusion-model-only LoRA targets without changing the SDXL trainer."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anima/Cosmos DiT slider LoRA trainer path.")
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--prompts_file", required=True)
    parser.add_argument("--model", default=None, help="Override model.diffusion_model_path")
    parser.add_argument("--text_encoder", default=None, help="Override model.text_encoder_path")
    parser.add_argument("--vae", default=None, help="Override model.vae_path")
    parser.add_argument("--comfyui", default=None, help="Override model.comfyui_path")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--precision", default=None)
    parser.add_argument("--preset", default=None, help="Override network.preset, e.g. attn_only or attn_mlp")
    parser.add_argument("--name", default=None)
    parser.add_argument("--save_path", "--output_dir", dest="save_path", default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--no_name_suffix", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--debug_log", action="store_true", help="Print detailed diagnostics for future trainer implementation")
    parser.add_argument("--inspect_targets", action="store_true")
    parser.add_argument("--inspect_limit", type=int, default=25)
    parser.add_argument("--target_report_json", default=None)
    parser.add_argument("--target_report_csv", default=None)
    parser.add_argument("--write_init_lora", action="store_true")
    main(parser.parse_args())
