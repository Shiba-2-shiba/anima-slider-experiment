from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import anima_train_lora_flow_slider as flow_trainer


class AnimaTrainLoraFlowSliderTests(unittest.TestCase):
    def test_parse_indices_uses_fallback_when_missing(self):
        self.assertEqual(flow_trainer.parse_indices(None, 7), [7])

    def test_parse_indices_ignores_empty_parts(self):
        self.assertEqual(flow_trainer.parse_indices("0, 2,,5", 7), [0, 2, 5])

    def test_resolve_step_bounds_default_leaves_previous_and_next_sigma(self):
        self.assertEqual(flow_trainer.resolve_step_bounds(8, None, None), (1, 6))

    def test_resolve_step_bounds_rejects_last_zero_sigma_slot(self):
        with self.assertRaises(ValueError):
            flow_trainer.resolve_step_bounds(8, None, 7)

    def test_validate_anima_resolution_requires_patch_multiple(self):
        flow_trainer.validate_anima_resolution(512, 768)

        with self.assertRaises(ValueError):
            flow_trainer.validate_anima_resolution(510, 768)

    def test_choose_training_step_index_mid_is_stable(self):
        self.assertEqual(
            flow_trainer.choose_training_step_index(step=0, seed=123, minimum=1, maximum=6, mode="mid"),
            3,
        )

    def test_choose_training_step_index_uniform_stays_in_bounds(self):
        values = [
            flow_trainer.choose_training_step_index(step=step, seed=123, minimum=1, maximum=6, mode="uniform")
            for step in range(20)
        ]

        self.assertTrue(all(1 <= value <= 6 for value in values))

    def test_choose_training_step_index_sigmoid_stays_in_bounds(self):
        sigmas = torch.tensor([1.0, 0.9, 0.75, 0.5, 0.25, 0.0])
        values = [
            flow_trainer.choose_training_step_index(
                step=step,
                seed=123,
                minimum=1,
                maximum=4,
                mode="sigmoid",
                sigmas=sigmas,
                sigmoid_scale=1.0,
            )
            for step in range(20)
        ]

        self.assertTrue(all(1 <= value <= 4 for value in values))

    def test_choose_training_step_index_shift_stays_in_bounds(self):
        sigmas = torch.tensor([1.0, 0.9, 0.75, 0.5, 0.25, 0.0])
        values = [
            flow_trainer.choose_training_step_index(
                step=step,
                seed=123,
                minimum=1,
                maximum=4,
                mode="shift",
                sigmas=sigmas,
                sigmoid_scale=1.0,
                discrete_flow_shift=3.0,
            )
            for step in range(20)
        ]

        self.assertTrue(all(1 <= value <= 4 for value in values))

    def test_choose_training_step_index_flux_shift_stays_in_bounds(self):
        sigmas = torch.tensor([1.0, 0.9, 0.75, 0.5, 0.25, 0.0])
        values = [
            flow_trainer.choose_training_step_index(
                step=step,
                seed=123,
                minimum=1,
                maximum=4,
                mode="flux_shift",
                sigmas=sigmas,
                image_seq_len=1024,
            )
            for step in range(20)
        ]

        self.assertTrue(all(1 <= value <= 4 for value in values))

    def test_compute_loss_weight_for_sigma(self):
        sigma = torch.tensor([0.5])

        self.assertTrue(torch.allclose(flow_trainer.compute_loss_weight_for_sigma(sigma, "none"), torch.tensor([1.0])))
        self.assertTrue(torch.allclose(flow_trainer.compute_loss_weight_for_sigma(sigma, "sigma_sqrt"), torch.tensor([4.0])))
        self.assertGreater(flow_trainer.compute_loss_weight_for_sigma(sigma, "cosmap").item(), 1.0)

    def test_parser_defaults_to_enhance_only_direction_loss(self):
        args = flow_trainer.build_parser().parse_args([])

        self.assertEqual(args.direction_loss, "enhance_only")

    def test_parser_accepts_model_override_aliases(self):
        parser = flow_trainer.build_parser()

        self.assertEqual(parser.parse_args(["--model", "other.safetensors"]).model, "other.safetensors")
        self.assertEqual(parser.parse_args(["--diffusion_model", "other.safetensors"]).model, "other.safetensors")

    def test_bare_model_override_resolves_next_to_config_model(self):
        resolved = flow_trainer.resolve_model_override(
            "other.safetensors",
            "C:/Models/anima/base.safetensors",
        )

        self.assertEqual(resolved, "C:\\Models\\anima\\other.safetensors")

    def test_path_model_override_is_left_as_given(self):
        resolved = flow_trainer.resolve_model_override(
            "D:/Models/other.safetensors",
            "C:/Models/anima/base.safetensors",
        )

        self.assertEqual(resolved, "D:/Models/other.safetensors")

    def test_branch_eval_summary_averages_branch_losses(self):
        details = [
            {"enhance_loss": 2.0, "enhance_raw_loss": 1.0},
            {"enhance_loss": 4.0, "enhance_raw_loss": 3.0},
        ]

        summary = flow_trainer.branch_eval_summary(details, "enhance")

        self.assertEqual(summary["losses"], [2.0, 4.0])
        self.assertEqual(summary["mean_loss"], 3.0)
        self.assertEqual(summary["raw_losses"], [1.0, 3.0])
        self.assertEqual(summary["mean_raw_loss"], 2.0)


if __name__ == "__main__":
    unittest.main()
