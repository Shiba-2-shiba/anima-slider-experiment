from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


Workflow = dict[str, dict[str, Any]]


def load_api_workflow(path: str | Path) -> Workflow:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if "prompt" in data and isinstance(data["prompt"], dict):
        data = data["prompt"]
    if not isinstance(data, dict):
        raise ValueError(f"Expected API workflow dict: {path}")
    return data


def save_api_workflow(path: str | Path, workflow: Workflow):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(workflow, handle, indent=2)


def clone_workflow(workflow: Workflow) -> Workflow:
    return deepcopy(workflow)


def require_node(workflow: Workflow, node_id: str) -> dict[str, Any]:
    try:
        return workflow[str(node_id)]
    except KeyError as exc:
        raise KeyError(f"Workflow node not found: {node_id}") from exc


def set_input(workflow: Workflow, node_id: str, input_name: str, value: Any):
    node = require_node(workflow, node_id)
    inputs = node.setdefault("inputs", {})
    if input_name not in inputs:
        raise KeyError(f"Workflow node {node_id} has no input named {input_name!r}")
    inputs[input_name] = value


def set_if_present(workflow: Workflow, node_id: str, input_name: str, value: Any):
    node = require_node(workflow, node_id)
    inputs = node.setdefault("inputs", {})
    if input_name in inputs:
        inputs[input_name] = value


def configure_anima_prompt(
    workflow: Workflow,
    *,
    positive: str | None = None,
    negative: str | None = None,
    seed: int | None = None,
    steps: int | None = None,
    cfg: float | None = None,
    width: int | None = None,
    height: int | None = None,
    batch_size: int | None = None,
    filename_prefix: str | None = None,
    strength_model: float | None = None,
):
    if positive is not None:
        set_input(workflow, "11", "text", positive)
    if negative is not None:
        set_input(workflow, "12", "text", negative)
    if seed is not None:
        set_input(workflow, "19", "seed", seed)
        set_if_present(workflow, "19", "control_after_generate", "fixed")
    if steps is not None:
        set_input(workflow, "19", "steps", steps)
    if cfg is not None:
        set_input(workflow, "19", "cfg", cfg)
    if width is not None:
        set_input(workflow, "28", "width", width)
    if height is not None:
        set_input(workflow, "28", "height", height)
    if batch_size is not None:
        set_input(workflow, "28", "batch_size", batch_size)
    if filename_prefix is not None:
        set_input(workflow, "46", "filename_prefix", filename_prefix)
    if strength_model is not None and "60" in workflow:
        set_input(workflow, "60", "strength_model", strength_model)


def with_lora_strength(workflow: Workflow, strength: float) -> Workflow:
    output = clone_workflow(workflow)
    configure_anima_prompt(output, strength_model=strength)
    return output
