import os
import tempfile
import unittest

from src.config_parser import load_config
from src.dataset_loader import load_dataset


class TestConfigAndDataset(unittest.TestCase):
    def test_load_config_valid_json(self):
        content = """{
            "models": ["model-a", "model-b", "model-c"],
            "dataset_path": "datasets/sample_prompts.jsonl",
            "max_new_tokens": 30
        }"""
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            cfg = load_config(temp_path)
            self.assertEqual(len(cfg.models), 3)
            self.assertEqual(cfg.max_new_tokens, 30)
            self.assertEqual(cfg.dataset_path, "datasets/sample_prompts.jsonl")
        finally:
            os.remove(temp_path)

    def test_load_config_rejects_fewer_than_three_models(self):
        content = """{
            "models": ["model-a", "model-b"],
            "dataset_path": "dummy.jsonl"
        }"""
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            with self.assertRaises(ValueError):
                load_config(temp_path)
        finally:
            os.remove(temp_path)

    def test_load_dataset_jsonl(self):
        content = """{"id": "p1", "prompt": "Hello world"}
{"id": "p2", "prompt": "Explain gravity"}"""
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            prompts = load_dataset(temp_path)
            self.assertEqual(len(prompts), 2)
            self.assertEqual(prompts[0]["id"], "p1")
            self.assertEqual(prompts[0]["prompt"], "Hello world")
        finally:
            os.remove(temp_path)


if __name__ == "__main__":
    unittest.main()
