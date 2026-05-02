from __future__ import annotations

import unittest
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from anima_slider import anima_conditioning
from anima_slider.prompt_util import PromptSettings


class FakeClip:
    def tokenize(self, text):
        return {"text": text}

    def encode_from_tokens(self, tokens, return_pooled=False, return_dict=False):
        self.last_return_pooled = return_pooled
        self.last_return_dict = return_dict
        length = len(tokens["text"])
        return {
            "cond": torch.ones(1, length, 4),
            "pooled_output": torch.full((1, 4), length, dtype=torch.float32),
            "t5xxl_ids": torch.arange(length, dtype=torch.int32),
            "t5xxl_weights": torch.ones(length),
        }


class AnimaConditioningTests(unittest.TestCase):
    def test_prompt_texts_fills_defaults(self):
        prompt = PromptSettings(target="adult portrait")

        texts = anima_conditioning.prompt_texts(prompt)

        self.assertEqual(texts["target"], "adult portrait")
        self.assertEqual(texts["positive"], "adult portrait")
        self.assertEqual(texts["neutral"], "adult portrait")
        self.assertEqual(texts["unconditional"], "")

    def test_encode_prompt_condition_set_uses_comfy_clip_contract(self):
        clip = FakeClip()
        prompt = PromptSettings(
            target="adult target",
            positive="older adult",
            unconditional="younger adult",
            neutral="adult neutral",
            guidance_scale=3.5,
            action="enhance",
            width=896,
            height=1152,
        )

        record = anima_conditioning.encode_prompt_condition_set(clip, prompt, prompt_index=7, roles=("target", "positive"))

        self.assertTrue(clip.last_return_pooled)
        self.assertTrue(clip.last_return_dict)
        self.assertEqual(record.prompt_index, 7)
        self.assertEqual(record.texts["positive"], "older adult")
        self.assertEqual(set(record.conds), {"target", "positive"})
        self.assertEqual(record.conds["target"].cond.shape, (1, len("adult target"), 4))
        self.assertIn("t5xxl_ids", record.conds["target"].extra)
        self.assertIn("t5xxl_weights", record.conds["target"].extra)

    def test_cache_manifest_records_shapes(self):
        clip = FakeClip()
        prompt = PromptSettings(target="adult")
        record = anima_conditioning.encode_prompt_condition_set(clip, prompt, prompt_index=0, roles=("target",))

        manifest = anima_conditioning.cache_manifest([record])

        self.assertEqual(manifest["prompt_count"], 1)
        self.assertEqual(manifest["roles"], ["target"])
        self.assertEqual(manifest["records"][0]["conditions"]["target"]["cond"]["shape"], [1, 5, 4])
        self.assertEqual(manifest["records"][0]["conditions"]["target"]["extra"]["t5xxl_ids"]["shape"], [5])


if __name__ == "__main__":
    unittest.main()
