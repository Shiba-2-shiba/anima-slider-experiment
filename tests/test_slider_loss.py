from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from anima_slider import slider_loss


class SliderLossTests(unittest.TestCase):
    def test_enhance_desired_output(self):
        positive = torch.tensor([3.0])
        unconditional = torch.tensor([1.0])
        neutral = torch.tensor([10.0])

        desired = slider_loss.desired_slider_output(positive, unconditional, neutral, guidance_scale=2.0, action="enhance")

        self.assertTrue(torch.equal(desired, torch.tensor([14.0])))

    def test_erase_desired_output(self):
        positive = torch.tensor([3.0])
        unconditional = torch.tensor([1.0])
        neutral = torch.tensor([10.0])

        desired = slider_loss.desired_slider_output(positive, unconditional, neutral, guidance_scale=2.0, action="erase")

        self.assertTrue(torch.equal(desired, torch.tensor([6.0])))

    def test_slider_mse_loss(self):
        outputs = slider_loss.SliderOutputs(
            target=torch.tensor([0.0]),
            positive=torch.tensor([3.0]),
            unconditional=torch.tensor([1.0]),
            neutral=torch.tensor([0.0]),
        )

        loss = slider_loss.slider_mse_loss(outputs, guidance_scale=1.5, action="enhance")

        self.assertEqual(loss.item(), 9.0)

    def test_normalize_like_matches_reference_norm_per_batch(self):
        tensor = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
        reference = torch.tensor([[6.0, 8.0], [5.0, 0.0]])

        normalized = slider_loss.normalize_like(tensor, reference)

        self.assertTrue(torch.allclose(normalized.norm(dim=1), reference.norm(dim=1)))
        self.assertTrue(torch.allclose(normalized[0], torch.tensor([6.0, 8.0])))
        self.assertTrue(torch.allclose(normalized[1], torch.tensor([0.0, 5.0])))

    def test_flow_slider_teacher_enhances_direction(self):
        target = torch.tensor([[10.0, 0.0]])
        positive = torch.tensor([[3.0, 4.0]])
        unconditional = torch.tensor([[1.0, 1.0]])

        teacher = slider_loss.flow_slider_teacher(
            target,
            positive,
            unconditional,
            eta=2.0,
            action="enhance",
        )

        self.assertTrue(torch.equal(teacher, torch.tensor([[14.0, 6.0]])))

    def test_flow_slider_teacher_erases_direction(self):
        target = torch.tensor([[10.0, 0.0]])
        positive = torch.tensor([[3.0, 4.0]])
        unconditional = torch.tensor([[1.0, 1.0]])

        teacher = slider_loss.flow_slider_teacher(
            target,
            positive,
            unconditional,
            eta=2.0,
            action="erase",
        )

        self.assertTrue(torch.equal(teacher, torch.tensor([[6.0, -6.0]])))

    def test_flow_slider_teacher_can_normalize_to_positive_prediction(self):
        target = torch.tensor([[10.0, 0.0]])
        positive = torch.tensor([[3.0, 4.0]])
        unconditional = torch.tensor([[1.0, 1.0]])

        teacher = slider_loss.flow_slider_teacher(
            target,
            positive,
            unconditional,
            eta=2.0,
            action="enhance",
            normalize_to=positive,
        )

        self.assertTrue(torch.allclose(teacher.norm(dim=1), positive.norm(dim=1)))


if __name__ == "__main__":
    unittest.main()
