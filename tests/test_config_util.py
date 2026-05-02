import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from anima_slider import config_util


class AnimaSliderConfigUtilTests(unittest.TestCase):
    def test_attn_only_preset_replaces_include_and_exclude_patterns(self):
        config = config_util.RootConfig(
            model=config_util.ModelConfig(
                diffusion_model_path="model.safetensors",
                text_encoder_path="qwen.safetensors",
                vae_path="vae.safetensors",
                comfyui_path="ComfyUI",
            ),
            network=config_util.NetworkConfig(
                preset="attn_only",
                include_patterns=["unused"],
                exclude_patterns=[],
            ),
        )

        config = config_util.apply_network_preset(config)

        self.assertEqual(
            config.network.include_patterns,
            [
                "model.diffusion_model.blocks.*.self_attn.*_proj",
                "model.diffusion_model.blocks.*.cross_attn.*_proj",
            ],
        )
        self.assertIn("*llm_adapter*", config.network.exclude_patterns)

    def test_repo_root_token_resolves_from_configs_directory(self):
        root = REPO_ROOT / "tmp-config-fixture-token"
        config_dir = root / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "config.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "model:",
                    '  diffusion_model_path: "{repo_root}/学習用モデル/Diffusion_model/model.safetensors"',
                    '  text_encoder_path: "{repo_root}/学習用モデル/Textencorder/qwen.safetensors"',
                    '  vae_path: "{repo_root}/学習用モデル/VAE/vae.safetensors"',
                    '  comfyui_path: "{repo_root}/ComfyUI"',
                    "network:",
                    '  preset: "attn_only"',
                ]
            ),
            encoding="utf-8",
        )

        config = config_util.load_config_from_yaml(str(config_path))

        self.assertEqual(
            config.model.diffusion_model_path,
            str(root / "学習用モデル" / "Diffusion_model" / "model.safetensors"),
        )
        self.assertEqual(config.model.comfyui_path, str(root / "ComfyUI"))

    def test_relative_model_paths_resolve_from_repo_root(self):
        root = REPO_ROOT / "tmp-config-fixture-relative"
        config_dir = root / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "config.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "model:",
                    '  diffusion_model_path: "学習用モデル/Diffusion_model/model.safetensors"',
                    '  text_encoder_path: "学習用モデル/Textencorder/qwen.safetensors"',
                    '  vae_path: "学習用モデル/VAE/vae.safetensors"',
                    '  comfyui_path: "ComfyUI"',
                    "network:",
                    '  preset: "attn_only"',
                ]
            ),
            encoding="utf-8",
        )

        config = config_util.load_config_from_yaml(str(config_path))

        self.assertEqual(
            config.model.text_encoder_path,
            str(root / "学習用モデル" / "Textencorder" / "qwen.safetensors"),
        )
        self.assertEqual(config.model.comfyui_path, str(root / "ComfyUI"))


if __name__ == "__main__":
    unittest.main()
