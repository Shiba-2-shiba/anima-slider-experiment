from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from anima_slider import comfy_workflow


def post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def run_workflow(url: str, workflow: dict, timeout_seconds: int) -> dict:
    submit = post_json(f"{url.rstrip('/')}/prompt", {"prompt": workflow})
    prompt_id = submit["prompt_id"]
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        history = get_json(f"{url.rstrip('/')}/history/{prompt_id}")
        if prompt_id in history:
            return history[prompt_id]
        time.sleep(1)
    raise TimeoutError(f"ComfyUI prompt did not finish within {timeout_seconds}s: {prompt_id}")


def main(args):
    workflow = comfy_workflow.load_api_workflow(args.workflow)
    comfy_workflow.configure_anima_prompt(
        workflow,
        positive=args.positive,
        negative=args.negative,
        seed=args.seed,
        steps=args.steps,
        cfg=args.cfg,
        width=args.width,
        height=args.height,
        batch_size=args.batch_size,
        filename_prefix=args.filename_prefix,
        strength_model=args.strength,
    )
    if args.lora_name is not None:
        comfy_workflow.set_input(workflow, "60", "lora_name", args.lora_name)
    if args.save_patched_workflow:
        comfy_workflow.save_api_workflow(args.save_patched_workflow, workflow)

    if args.dry_run:
        print(json.dumps(workflow, indent=2))
        return

    result = run_workflow(args.url, workflow, args.timeout)
    status = result.get("status", {})
    print(json.dumps({"status": status, "outputs": result.get("outputs", {})}, indent=2))
    if status.get("status_str") != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an Anima ComfyUI API workflow.")
    parser.add_argument("--workflow", default=str(Path("workflows") / "anima_base.json"))
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--positive", default=None)
    parser.add_argument("--negative", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--cfg", type=float, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--strength", type=float, default=None)
    parser.add_argument("--lora_name", default=None)
    parser.add_argument("--filename_prefix", default=None)
    parser.add_argument("--save_patched_workflow", default=None)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--dry_run", action="store_true")
    main(parser.parse_args())
