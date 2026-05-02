from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from anima_slider import anima_conditioning
from anima_slider import anima_forward


class FakeLatentFormat:
    latent_channels = 16
    latent_dimensions = 3
    spacial_downscale_ratio = 8


class FakePatcher:
    load_device = torch.device("cpu")

    def get_model_object(self, name):
        if name == "latent_format":
            return FakeLatentFormat()
        if name == "model_sampling":
            return FakeModelSampling()
        raise KeyError(name)


class FakeModelSampling:
    sigmas = torch.linspace(0.001, 1.0, 1000)

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        return sigma.reshape([sigma.shape[0]] + [1] * (noise.ndim - 1)) * noise + latent_image


class AnimaForwardTests(unittest.TestCase):
    def test_latent_shape_for_anima_resolution(self):
        shape = anima_forward.latent_shape_for_resolution(FakePatcher(), width=1024, height=896, batch_size=2)

        self.assertEqual(shape, (2, 16, 1, 112, 128))

    def test_condition_to_model_kwargs_moves_extra_tensors(self):
        cond = anima_conditioning.AnimaCond(
            cond=torch.ones(1, 2, 4),
            pooled=None,
            extra={
                "t5xxl_ids": torch.tensor([1, 2], dtype=torch.int32),
                "t5xxl_weights": torch.ones(2),
            },
        )

        kwargs = anima_forward.condition_to_model_kwargs(cond, device=torch.device("cpu"), dtype=torch.float16)

        self.assertEqual(kwargs["c_crossattn"].dtype, torch.float16)
        self.assertEqual(kwargs["t5xxl_ids"].dtype, torch.int32)
        self.assertEqual(kwargs["t5xxl_ids"].shape, (1, 2))
        self.assertEqual(kwargs["t5xxl_weights"].dtype, torch.float16)
        self.assertEqual(kwargs["t5xxl_weights"].shape, (1, 2, 1))

    def test_simple_sigmas_match_comfy_shape(self):
        sigmas = anima_forward.sigmas_for_steps(FakePatcher(), steps=4, scheduler_name="simple", device=torch.device("cpu"))

        self.assertEqual(sigmas.shape, (5,))
        self.assertGreater(sigmas[0].item(), sigmas[1].item())
        self.assertEqual(sigmas[-1].item(), 0.0)

    def test_validate_training_step_index_rejects_zero_sigma(self):
        sigmas = torch.tensor([1.0, 0.5, 0.0])

        with self.assertRaises(ValueError):
            anima_forward.validate_training_step_index(sigmas, 2)

    def test_scale_noise_for_sigma_uses_model_sampling(self):
        noise = torch.ones(2, 1, 1, 2, 2)
        scaled = anima_forward.scale_noise_for_sigma(FakePatcher(), noise, torch.tensor([0.25, 0.5]))

        self.assertTrue(torch.equal(scaled[0], torch.full((1, 1, 2, 2), 0.25)))
        self.assertTrue(torch.equal(scaled[1], torch.full((1, 1, 2, 2), 0.5)))

    def test_euler_step_from_denoised(self):
        latent = torch.tensor([2.0])
        denoised = torch.tensor([1.0])

        stepped = anima_forward.euler_step_from_denoised(latent, torch.tensor([1.0]), torch.tensor([0.5]), denoised)

        self.assertTrue(torch.allclose(stepped, torch.tensor([1.5])))


if __name__ == "__main__":
    unittest.main()
