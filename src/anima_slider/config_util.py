from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import BaseModel


PRECISION_TYPES = {"fp32", "float32", "fp16", "float16", "bf16", "bfloat16"}
REPO_ROOT_TOKEN = "{repo_root}"

NETWORK_PRESETS = {
    "attn_only": {
        "include_patterns": [
            "model.diffusion_model.blocks.*.self_attn.*_proj",
            "model.diffusion_model.blocks.*.cross_attn.*_proj",
        ],
        "exclude_patterns": [
            "*llm_adapter*",
            "*qwen*",
            "*t_embedder*",
            "*x_embedder*",
            "*final_layer*",
            "*adaln_modulation*",
        ],
    },
    "attn_mlp": {
        "include_patterns": [
            "model.diffusion_model.blocks.*.self_attn.*_proj",
            "model.diffusion_model.blocks.*.cross_attn.*_proj",
            "model.diffusion_model.blocks.*.mlp.layer1",
            "model.diffusion_model.blocks.*.mlp.layer2",
        ],
        "exclude_patterns": [
            "*llm_adapter*",
            "*qwen*",
            "*t_embedder*",
            "*x_embedder*",
            "*final_layer*",
            "*adaln_modulation*",
        ],
    },
}

OVERRIDE_ALIASES = {
    "model": "model.diffusion_model_path",
    "diffusion_model": "model.diffusion_model_path",
    "text_encoder": "model.text_encoder_path",
    "vae": "model.vae_path",
    "comfyui": "model.comfyui_path",
    "rank": "network.rank",
    "alpha": "network.alpha",
    "iterations": "train.iterations",
    "lr": "train.lr",
    "precision": "train.precision",
    "name": "save.name",
    "save_path": "save.path",
    "output_dir": "save.path",
    "preset": "network.preset",
}


class ModelConfig(BaseModel):
    diffusion_model_path: str
    text_encoder_path: str
    vae_path: str
    comfyui_path: str


class NetworkConfig(BaseModel):
    rank: int = 16
    alpha: float = 16.0
    target: str = "diffusion_model_only"
    preset: str | None = None
    include_patterns: list[str] = []
    exclude_patterns: list[str] = []


class TrainConfig(BaseModel):
    precision: str = "bfloat16"
    iterations: int = 500
    lr: float = 2e-5
    optimizer: str = "AdamW"
    sampler: str = "er_sde"
    scheduler: str = "normal"
    steps: int = 30
    cfg: float = 4.5


class SaveConfig(BaseModel):
    name: str = "anima_slider"
    path: str = "./models"
    per_steps: int = 100
    precision: str = "bfloat16"


class LoggingConfig(BaseModel):
    verbose: bool = False


class RootConfig(BaseModel):
    model: ModelConfig
    network: NetworkConfig
    train: TrainConfig = TrainConfig()
    save: SaveConfig = SaveConfig()
    logging: LoggingConfig = LoggingConfig()


def _model_to_dict(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _parse_override_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def _set_nested_value(data: dict[str, Any], key: str, value: Any):
    resolved_key = OVERRIDE_ALIASES.get(key, key)
    path = resolved_key.split(".")
    current: Any = data
    for segment in path[:-1]:
        if not isinstance(current, dict) or segment not in current:
            raise ValueError(f"Unknown config override: {key}")
        current = current[segment]

    leaf = path[-1]
    if not isinstance(current, dict) or leaf not in current:
        raise ValueError(f"Unknown config override: {key}")
    current[leaf] = _parse_override_value(value)


def load_config_from_yaml(config_path: str) -> RootConfig:
    with open(config_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    config = apply_network_preset(RootConfig(**data))
    return resolve_model_paths(config, default_repo_root_for_config(config_path))


def apply_config_overrides(config: RootConfig, overrides: Mapping[str, Any]) -> RootConfig:
    data = _model_to_dict(config)
    for key, value in overrides.items():
        if value is None:
            continue
        _set_nested_value(data, key, value)
    return apply_network_preset(RootConfig(**data))


def apply_network_preset(config: RootConfig) -> RootConfig:
    preset_name = config.network.preset
    if not preset_name:
        return config
    if preset_name not in NETWORK_PRESETS:
        return config
    preset = NETWORK_PRESETS[preset_name]
    data = _model_to_dict(config)
    data["network"]["include_patterns"] = list(preset["include_patterns"])
    data["network"]["exclude_patterns"] = list(preset["exclude_patterns"])
    return RootConfig(**data)


def default_repo_root_for_config(config_path: str | Path) -> Path:
    path = Path(config_path).resolve()
    if path.parent.name == "configs":
        return path.parent.parent
    return path.parent


def resolve_repo_path(raw_path: str, repo_root: str | Path) -> str:
    root = Path(repo_root).resolve()
    expanded = raw_path.replace(REPO_ROOT_TOKEN, str(root))
    path = Path(expanded).expanduser()
    if path.is_absolute():
        return str(path)
    return str((root / path).resolve())


def resolve_model_paths(config: RootConfig, repo_root: str | Path) -> RootConfig:
    data = _model_to_dict(config)
    for key in ["diffusion_model_path", "text_encoder_path", "vae_path", "comfyui_path"]:
        data["model"][key] = resolve_repo_path(data["model"][key], repo_root)
    return RootConfig(**data)


def finalize_training_config(config: RootConfig, append_name_suffix: bool = True) -> RootConfig:
    if append_name_suffix:
        config.save.name += f"_alpha{config.network.alpha}_rank{config.network.rank}"
    config.save.path = str(Path(config.save.path) / config.save.name)
    return config


def validate_config_paths(config: RootConfig) -> list[str]:
    errors = []
    for label, raw_path in [
        ("diffusion_model_path", config.model.diffusion_model_path),
        ("text_encoder_path", config.model.text_encoder_path),
        ("vae_path", config.model.vae_path),
        ("comfyui_path", config.model.comfyui_path),
    ]:
        if not Path(raw_path).exists():
            errors.append(f"{label} does not exist: {raw_path}")

    if config.train.precision not in PRECISION_TYPES:
        errors.append(f"Invalid train.precision: {config.train.precision}")
    if config.save.precision not in PRECISION_TYPES:
        errors.append(f"Invalid save.precision: {config.save.precision}")
    if config.network.target != "diffusion_model_only":
        errors.append("Only network.target=diffusion_model_only is supported for the first Anima trainer path")
    if config.network.preset and config.network.preset not in NETWORK_PRESETS:
        errors.append(
            f"Invalid network.preset: {config.network.preset}. "
            f"Expected one of: {', '.join(sorted(NETWORK_PRESETS))}"
        )
    if not config.network.include_patterns:
        errors.append("network.include_patterns is empty")
    return errors
