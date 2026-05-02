from __future__ import annotations

import argparse
from pathlib import Path

from . import config_util
from . import lora_util


def main(args):
    config = config_util.load_config_from_yaml(args.config_file)
    config = config_util.apply_config_overrides(config, {"preset": args.preset, "rank": args.rank})
    errors = config_util.validate_config_paths(config)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(2)

    inspection = lora_util.inspect_safetensors_targets_with_stats(
        config.model.diffusion_model_path,
        config.network.include_patterns,
        config.network.exclude_patterns,
    )
    targets = inspection.targets
    print(f"accepted_targets: {len(targets)}")
    print(f"estimated_lora_parameters: {lora_util.estimate_lora_parameters(targets, config.network.rank):,}")

    if args.json:
        lora_util.save_target_report_json(args.json, inspection, config.network.rank)
        print(f"wrote_json: {Path(args.json)}")
    if args.csv:
        lora_util.save_target_report_csv(args.csv, targets)
        print(f"wrote_csv: {Path(args.csv)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect Anima LoRA target keys and write JSON/CSV reports.")
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--preset", default=None, help="Override network.preset, e.g. attn_only or attn_mlp")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--json", default=None, help="Output target report JSON path")
    parser.add_argument("--csv", default=None, help="Output target table CSV path")
    main(parser.parse_args())
