from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from anima_slider import config_util


def path_status(path: str | Path) -> dict[str, object]:
    resolved = Path(path)
    return {
        "path": str(resolved),
        "exists": resolved.exists(),
        "is_dir": resolved.is_dir(),
        "is_file": resolved.is_file(),
    }


def collect_status(config_file: str) -> dict[str, object]:
    config = config_util.load_config_from_yaml(config_file)
    paths = {
        "diffusion_model": config.model.diffusion_model_path,
        "text_encoder": config.model.text_encoder_path,
        "vae": config.model.vae_path,
        "comfyui": config.model.comfyui_path,
        "workflows": REPO_ROOT / "workflows",
        "models_output": REPO_ROOT / "models",
        "cache": REPO_ROOT / "cache",
        "reports": REPO_ROOT / "reports",
    }
    required = ["diffusion_model", "text_encoder", "vae", "comfyui", "workflows"]
    status = {name: path_status(path) for name, path in paths.items()}
    errors = config_util.validate_config_paths(config)
    for name in required:
        if not status[name]["exists"]:
            errors.append(f"{name} is missing: {status[name]['path']}")
    if not (Path(config.model.comfyui_path) / "comfy" / "sd.py").exists():
        errors.append(f"ComfyUI checkout is incomplete: {config.model.comfyui_path}")
    return {
        "repo_root": str(REPO_ROOT),
        "config_file": str(Path(config_file).resolve()),
        "ok": not errors,
        "paths": status,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check local-only assets needed for Anima slider training.")
    parser.add_argument("--config_file", default="configs/config-anima-slider.yaml")
    parser.add_argument("--json", action="store_true", help="Print machine-readable status.")
    args = parser.parse_args()

    status = collect_status(args.config_file)
    if args.json:
        print(json.dumps(status, indent=2, ensure_ascii=False))
    else:
        print(f"repo_root: {status['repo_root']}")
        for name, item in status["paths"].items():
            marker = "OK" if item["exists"] else "MISSING"
            print(f"{marker:7} {name}: {item['path']}")
        if status["errors"]:
            print("\nErrors:")
            for error in status["errors"]:
                print(f"- {error}")
    return 0 if status["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
