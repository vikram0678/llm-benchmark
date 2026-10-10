import os
import tempfile
import unittest

import pandas as pd

from src.profiler import compute_type_token_ratio, profile_inference
from src.reporter import generate_visualizations, save_results_csv


class TestProfilerAndReporter(unittest.TestCase):
    def test_compute_type_token_ratio(self):
        rep_text = "test test test test"
        self.assertEqual(compute_type_token_ratio(rep_text), 0.25)

        rich_text = "The quick brown fox"
        self.assertEqual(compute_type_token_ratio(rich_text), 1.0)

        self.assertEqual(compute_type_token_ratio(""), 0.0)

    def test_profile_inference_mock(self):
        def mock_inference():
            return "This is a benchmark test response.", 6

        metrics = profile_inference(mock_inference, min_quality_tokens=3)
        self.assertEqual(metrics["token_count"], 6)
        self.assertGreaterEqual(metrics["latency_sec"], 0.0)
        self.assertGreaterEqual(metrics["throughput_tps"], 0.0)
        self.assertGreater(metrics["peak_ram_mb"], 0.0)
        self.assertGreater(metrics["quality_ttr"], 0.0)
        self.assertFalse(metrics["is_degenerate"])

    def test_reporter_persistence_and_charts(self):
        records = [
            {
                "model_name": "model-alpha",
                "prompt_id": "p1",
                "latency_sec": 0.5,
                "throughput_tps": 20.0,
                "peak_ram_mb": 150.0,
                "peak_vram_mb": 0.0,
                "quality_ttr": 0.8,
                "token_count": 10,
                "is_degenerate": False,
                "prompt_text": "Hi",
                "output_text": "Hello there",
            },
            {
                "model_name": "model-beta",
                "prompt_id": "p1",
                "latency_sec": 0.8,
                "throughput_tps": 12.5,
                "peak_ram_mb": 220.0,
                "peak_vram_mb": 0.0,
                "quality_ttr": 0.65,
                "token_count": 10,
                "is_degenerate": False,
                "prompt_text": "Hi",
                "output_text": "Hi friend",
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = save_results_csv(records, tmpdir)
            self.assertTrue(os.path.exists(csv_path))

            df = pd.read_csv(csv_path)
            self.assertEqual(len(df), 2)
            self.assertIn("latency_sec", df.columns)
            self.assertIn("throughput_tps", df.columns)

            charts = generate_visualizations(df, tmpdir)
            self.assertEqual(len(charts), 3)
            for chart in charts:
                self.assertTrue(os.path.exists(chart))


if __name__ == "__main__":
    unittest.main()
