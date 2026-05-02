from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from anima_slider import prompt_util


class PromptUtilTests(unittest.TestCase):
    def test_validate_prompts_rejects_child_terms_by_default(self):
        prompts = [
            prompt_util.PromptSettings(
                target="safe adult portrait",
                positive="safe older adult portrait",
                unconditional="safe young girl child portrait",
                neutral="safe adult portrait",
            )
        ]

        errors = prompt_util.validate_prompts(prompts)

        self.assertIn("unsafe age term", errors[0])

    def test_validate_prompts_can_allow_explicit_age_term_experiments(self):
        prompts = [
            prompt_util.PromptSettings(
                target="safe adult portrait",
                positive="safe older adult portrait",
                unconditional="safe age appropriate nonsexual young girl child portrait",
                neutral="safe adult portrait",
            )
        ]

        errors = prompt_util.validate_prompts(prompts, allow_unsafe_age_terms=True)

        self.assertEqual(errors, [])

    def test_breast_size_slider_prompts_are_adult_nonsexual_and_valid(self):
        prompts = prompt_util.load_prompts_from_yaml(str(REPO_ROOT / "prompts" / "prompts-anima-breast_size_slider.yaml"))

        errors = prompt_util.validate_prompts(prompts)

        self.assertEqual(errors, [])
        self.assertEqual(len(prompts), 8)
        for prompt in prompts:
            joined = " ".join([prompt.target, prompt.positive or "", prompt.unconditional, prompt.neutral or ""])
            self.assertIn("adult woman", joined)
            self.assertIn("nonsexual", joined)
            self.assertIn("fully clothed", joined)
            self.assertIn("large breasts", prompt.positive or "")
            self.assertIn("small breasts", prompt.unconditional)


if __name__ == "__main__":
    unittest.main()
