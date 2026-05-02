import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from anima_slider import config_util
from anima_slider import lora_util
from anima_slider import lora_network


class FakeSafeOpen:
    def __init__(self, tensors):
        self.tensors = tensors

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def keys(self):
        return self.tensors.keys()

    def get_tensor(self, key):
        return self.tensors[key]


class AnimaSliderLoraUtilTests(unittest.TestCase):
    def test_include_patterns_match_weight_keys_without_requiring_weight_suffix(self):
        tensors = {
            "model.diffusion_model.blocks.0.self_attn.q_proj.weight": torch.zeros((4, 3)),
            "model.diffusion_model.blocks.0.cross_attn.output_proj.weight": torch.zeros((5, 4)),
            "model.diffusion_model.blocks.0.mlp.layer1.weight": torch.zeros((6, 5)),
            "model.diffusion_model.llm_adapter.linear.weight": torch.zeros((7, 6)),
            "model.diffusion_model.blocks.0.norm.weight": torch.zeros((8,)),
        }
        with patch.object(lora_util, "safe_open", return_value=FakeSafeOpen(tensors)):
            targets = lora_util.inspect_safetensors_targets(
                "unused.safetensors",
                include_patterns=[
                    "model.diffusion_model.blocks.*.self_attn.*_proj",
                    "model.diffusion_model.blocks.*.cross_attn.*_proj",
                    "model.diffusion_model.blocks.*.mlp.layer*",
                    "model.diffusion_model.llm_adapter.*",
                ],
                exclude_patterns=["model.diffusion_model.llm_adapter.*"],
            )

        self.assertEqual(
            [target.lora_key for target in targets],
            [
                "diffusion_model.blocks.0.self_attn.q_proj",
                "diffusion_model.blocks.0.cross_attn.output_proj",
                "diffusion_model.blocks.0.mlp.layer1",
            ],
        )

    def test_target_report_contains_counts_and_rows(self):
        inspection = lora_util.TargetInspection(
            targets=[
                lora_util.LoraTarget(
                    raw_weight_key="model.diffusion_model.blocks.0.self_attn.q_proj.weight",
                    comfy_weight_key="diffusion_model.blocks.0.self_attn.q_proj.weight",
                    lora_key="diffusion_model.blocks.0.self_attn.q_proj",
                    shape=(4, 3),
                )
            ],
            total_keys=2,
            weight_keys=1,
            include_matched_keys=1,
            excluded_keys=0,
            skipped_non_2d_keys=0,
            skipped_non_diffusion_model_keys=0,
            sample_include_misses=[],
            sample_excluded=[],
            sample_non_2d=[],
            sample_non_diffusion_model=[],
        )

        report = lora_util.build_target_report(inspection, rank=2)

        self.assertEqual(report["counts"]["accepted_targets"], 1)
        self.assertEqual(report["counts"]["estimated_lora_parameters"], 14)
        self.assertEqual(report["by_module"], {"self_attn": 1})
        self.assertEqual(report["targets"][0]["shape"], [4, 3])

    def test_net_prefixed_anima_checkpoint_keys_normalize_to_diffusion_model_targets(self):
        tensors = {
            "net.blocks.0.self_attn.q_proj.weight": torch.zeros((4, 3)),
            "net.blocks.0.cross_attn.output_proj.weight": torch.zeros((5, 4)),
            "net.blocks.0.mlp.layer1.weight": torch.zeros((6, 5)),
            "net.final_layer.linear.weight": torch.zeros((7, 6)),
        }
        preset = config_util.NETWORK_PRESETS["attn_mlp"]
        with patch.object(lora_util, "safe_open", return_value=FakeSafeOpen(tensors)):
            targets = lora_util.inspect_safetensors_targets(
                "unused.safetensors",
                include_patterns=preset["include_patterns"],
                exclude_patterns=preset["exclude_patterns"],
            )

        self.assertEqual(
            sorted(target.lora_key for target in targets),
            [
                "diffusion_model.blocks.0.cross_attn.output_proj",
                "diffusion_model.blocks.0.mlp.layer1",
                "diffusion_model.blocks.0.self_attn.q_proj",
            ],
        )


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = torch.nn.Module()
        self.self_attn.q_proj = torch.nn.Linear(3, 4, bias=False)
        self.mlp = torch.nn.Module()
        self.mlp.layer1 = torch.nn.Linear(3, 4, bias=False)


class FakeDiffusion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion_model = torch.nn.Module()
        self.diffusion_model.blocks = torch.nn.ModuleList([Block()])


class AnimaSliderLoraNetworkTests(unittest.TestCase):
    def test_inject_lora_linear_modules_matches_checkpoint_style_patterns(self):
        model = FakeDiffusion()
        injected = lora_network.inject_lora_linear_modules(
            model,
            include_patterns=["model.diffusion_model.blocks.*.self_attn.*_proj"],
            exclude_patterns=[],
            rank=2,
            alpha=2.0,
        )

        self.assertEqual([item.lora_key for item in injected], ["diffusion_model.blocks.0.self_attn.q_proj"])
        self.assertIsInstance(model.diffusion_model.blocks[0].self_attn.q_proj, lora_network.LoRALinear)
        self.assertIsInstance(model.diffusion_model.blocks[0].mlp.layer1, torch.nn.Linear)

        state = lora_network.lora_state_dict_from_model(model)
        self.assertEqual(
            sorted(state),
            [
                "diffusion_model.blocks.0.self_attn.q_proj.alpha",
                "diffusion_model.blocks.0.self_attn.q_proj.lora_down.weight",
                "diffusion_model.blocks.0.self_attn.q_proj.lora_up.weight",
            ],
        )

    def test_lora_enabled_context_disables_delta(self):
        model = FakeDiffusion()
        lora_network.inject_lora_linear_modules(
            model,
            include_patterns=["model.diffusion_model.blocks.*.self_attn.*_proj"],
            exclude_patterns=[],
            rank=2,
            alpha=2.0,
        )
        wrapped = model.diffusion_model.blocks[0].self_attn.q_proj
        torch.nn.init.ones_(wrapped.lora_up.weight)
        x = torch.ones(1, 3)

        enabled = wrapped(x)
        with lora_network.lora_enabled(model, False):
            disabled = wrapped(x)

        self.assertFalse(torch.equal(enabled, disabled))
        self.assertTrue(wrapped.enabled)
        self.assertEqual(len(lora_network.lora_parameters(model)), 2)

    def test_lora_multiplier_scales_and_restores_delta(self):
        model = FakeDiffusion()
        lora_network.inject_lora_linear_modules(
            model,
            include_patterns=["model.diffusion_model.blocks.*.self_attn.*_proj"],
            exclude_patterns=[],
            rank=1,
            alpha=1.0,
        )
        wrapped = model.diffusion_model.blocks[0].self_attn.q_proj
        torch.nn.init.ones_(wrapped.lora_down.weight)
        torch.nn.init.ones_(wrapped.lora_up.weight)
        x = torch.ones(1, 3)

        with lora_network.lora_multiplier(model, 0.0):
            base = wrapped(x)
        positive = wrapped(x)
        with lora_network.lora_multiplier(model, -1.0):
            negative = wrapped(x)

        self.assertTrue(torch.allclose(positive - base, -(negative - base)))
        self.assertEqual(wrapped.multiplier, 1.0)

    def test_lora_multiplier_restores_nested_contexts(self):
        model = FakeDiffusion()
        lora_network.inject_lora_linear_modules(
            model,
            include_patterns=["model.diffusion_model.blocks.*.self_attn.*_proj"],
            exclude_patterns=[],
            rank=1,
            alpha=1.0,
        )
        wrapped = model.diffusion_model.blocks[0].self_attn.q_proj

        with lora_network.lora_multiplier(model, 2.0):
            self.assertEqual(wrapped.multiplier, 2.0)
            with lora_network.lora_multiplier(model, -1.0):
                self.assertEqual(wrapped.multiplier, -1.0)
            self.assertEqual(wrapped.multiplier, 2.0)

        self.assertEqual(wrapped.multiplier, 1.0)

    def test_inject_lora_linear_modules_applies_regex_rank_overrides(self):
        model = FakeDiffusion()
        injected = lora_network.inject_lora_linear_modules(
            model,
            include_patterns=[
                "model.diffusion_model.blocks.*.self_attn.*_proj",
                "model.diffusion_model.blocks.*.mlp.layer*",
            ],
            exclude_patterns=[],
            rank=2,
            alpha=2.0,
            reg_dims={r".*self_attn.*": 3, r".*mlp.*": 1},
        )

        self.assertEqual(
            [(item.lora_key, item.rank) for item in injected],
            [
                ("diffusion_model.blocks.0.self_attn.q_proj", 3),
                ("diffusion_model.blocks.0.mlp.layer1", 1),
            ],
        )
        self.assertEqual(model.diffusion_model.blocks[0].self_attn.q_proj.rank, 3)
        self.assertEqual(model.diffusion_model.blocks[0].mlp.layer1.rank, 1)


if __name__ == "__main__":
    unittest.main()
