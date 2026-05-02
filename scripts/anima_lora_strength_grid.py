from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_STRENGTHS = [-6.0, -3.0, -1.0, 0.0, 1.0, 3.0, 6.0]


def normalize_argv(argv: list[str]) -> list[str]:
    normalized = []
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--strengths" and index + 1 < len(argv):
            value = argv[index + 1]
            if value.startswith("-"):
                normalized.append(f"--strengths={value}")
                index += 2
                continue
        normalized.append(arg)
        index += 1
    return normalized


def parse_strengths(raw_strengths: str | None) -> list[float]:
    if raw_strengths is None:
        return DEFAULT_STRENGTHS
    return [float(value.strip()) for value in raw_strengths.split(",") if value.strip()]


def main(args):
    strengths = parse_strengths(args.strengths)
    script = Path(__file__).with_name("anima_comfy_infer.py")
    for strength in strengths:
        label = str(strength).replace("-", "neg").replace(".", "p")
        command = [
            sys.executable,
            str(script),
            "--workflow",
            args.workflow,
            "--url",
            args.url,
            "--strength",
            str(strength),
            "--seed",
            str(args.seed),
            "--filename_prefix",
            f"{args.filename_prefix}/strength_{label}",
        ]
        if args.lora_name:
            command.extend(["--lora_name", args.lora_name])
        if args.positive:
            command.extend(["--positive", args.positive])
        if args.negative:
            command.extend(["--negative", args.negative])
        if args.steps is not None:
            command.extend(["--steps", str(args.steps)])
        if args.cfg is not None:
            command.extend(["--cfg", str(args.cfg)])
        if args.width is not None:
            command.extend(["--width", str(args.width)])
        if args.height is not None:
            command.extend(["--height", str(args.height)])
        if args.dry_run:
            command.append("--dry_run")

        print(f"running_strength: {strength}")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an Anima LoRA strength grid through ComfyUI.")
    parser.add_argument("--workflow", default=str(Path("workflows") / "anima_lora_model_only.json"))
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--strengths", default=None, help="Comma-separated strengths. Default: -6,-3,-1,0,1,3,6")
    parser.add_argument("--lora_name", default=None)
    parser.add_argument("--positive", default=None)
    parser.add_argument("--negative", default=None)
    parser.add_argument("--seed", type=int, default=961218314523996)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--cfg", type=float, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--filename_prefix", default="Anima/strength_grid")
    parser.add_argument("--dry_run", action="store_true")
    main(parser.parse_args(normalize_argv(sys.argv[1:])))
