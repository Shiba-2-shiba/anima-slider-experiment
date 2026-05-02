import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from anima_slider import comfy_workflow


class ComfyWorkflowTests(unittest.TestCase):
    def test_configure_anima_prompt_updates_expected_nodes(self):
        workflow = comfy_workflow.load_api_workflow(REPO_ROOT / "workflows" / "anima_lora_model_only.json")
        comfy_workflow.configure_anima_prompt(
            workflow,
            positive="positive prompt",
            negative="negative prompt",
            seed=123,
            steps=12,
            cfg=3.5,
            width=896,
            height=1152,
            filename_prefix="Anima/test",
            strength_model=-3,
        )

        self.assertEqual(workflow["11"]["inputs"]["text"], "positive prompt")
        self.assertEqual(workflow["12"]["inputs"]["text"], "negative prompt")
        self.assertEqual(workflow["19"]["inputs"]["seed"], 123)
        self.assertEqual(workflow["19"]["inputs"]["control_after_generate"], "fixed")
        self.assertEqual(workflow["19"]["inputs"]["steps"], 12)
        self.assertEqual(workflow["19"]["inputs"]["cfg"], 3.5)
        self.assertEqual(workflow["28"]["inputs"]["width"], 896)
        self.assertEqual(workflow["28"]["inputs"]["height"], 1152)
        self.assertEqual(workflow["46"]["inputs"]["filename_prefix"], "Anima/test")
        self.assertEqual(workflow["60"]["inputs"]["strength_model"], -3)


if __name__ == "__main__":
    unittest.main()
