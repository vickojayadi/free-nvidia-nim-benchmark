import csv
import tempfile
import unittest
from pathlib import Path

import nim_benchmark as nb


class BenchmarkSummaryTests(unittest.TestCase):
    def test_summarize_model_calculates_requested_cli_metrics(self):
        rows = [
            {"model": "m/a", "status": "OK", "total_s": 2.0, "ttft_s": 0.5, "tokens_per_s": 100.0},
            {"model": "m/a", "status": "OK", "total_s": 4.0, "ttft_s": 0.7, "tokens_per_s": 120.0},
            {"model": "m/a", "status": "HTTP 429", "total_s": 0.1},
        ]
        summary = nb.summarize_model("m/a", rows)
        self.assertEqual(summary["uptime_pct"], 66.7)
        self.assertEqual(summary["avg_time_s"], 3.0)
        self.assertEqual(summary["ttft_s"], 0.6)
        self.assertEqual(summary["best_time_s"], 2.0)
        self.assertEqual(summary["throughput_tps"], 110.0)

    def test_relative_scores_reward_uptime_low_latency_and_throughput(self):
        summaries = [
            {"model": "fast", "uptime_pct": 100.0, "avg_time_s": 2.0, "ttft_s": 0.4, "throughput_tps": 200.0},
            {"model": "slow", "uptime_pct": 100.0, "avg_time_s": 5.0, "ttft_s": 1.0, "throughput_tps": 80.0},
            {"model": "flaky", "uptime_pct": 50.0, "avg_time_s": 1.5, "ttft_s": 0.3, "throughput_tps": 250.0},
        ]
        scored = {row["model"]: row for row in nb.apply_relative_scores(summaries)}
        self.assertGreater(scored["fast"]["score"], scored["slow"]["score"])
        self.assertGreater(scored["fast"]["score"], scored["flaky"]["score"])
        self.assertTrue(all(0.0 <= row["score"] <= 100.0 for row in scored.values()))

    def test_table_columns_match_requested_order(self):
        row = {
            "model": "m/a",
            "score": 88.8,
            "uptime_pct": 100.0,
            "intelligence": 42,
            "intelligence_estimated": False,
            "avg_time_s": 2.5,
            "ttft_s": 0.6,
            "best_time_s": 2.2,
            "throughput_tps": 150.0,
        }
        table = nb.render_table([row])
        header = table.splitlines()[0].lower()
        expected = ["model", "score", "uptime", "intelligence", "avg time", "ttft", "best time", "throughput"]
        positions = [header.index(name) for name in expected]
        self.assertEqual(positions, sorted(positions))

    def test_save_csv_accepts_rows_with_different_fields(self):
        rows = [
            {"model": "error-model", "status": "HTTP 429", "total_s": 0.2},
            {
                "model": "ok-model",
                "status": "OK",
                "total_s": 2.0,
                "ttft_s": 0.4,
                "generation_s": 1.5,
                "tokens_per_s": 100.0,
                "generated_chars": 500,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runs.csv"
            nb.save_csv(str(path), rows)
            with path.open(newline="", encoding="utf-8") as handle:
                saved = list(csv.DictReader(handle))

        self.assertEqual(len(saved), 2)
        self.assertEqual(saved[1]["tokens_per_s"], "100.0")
        self.assertEqual(saved[1]["generated_chars"], "500")

    def test_stream_throughput_requires_incremental_streaming(self):
        self.assertIsNone(
            nb.calculate_stream_throughput(
                completion_tokens=192,
                first_generated_at=10.0,
                last_generated_at=10.0,
                generated_events=1,
            )
        )
        self.assertAlmostEqual(
            nb.calculate_stream_throughput(
                completion_tokens=101,
                first_generated_at=10.0,
                last_generated_at=11.0,
                generated_events=20,
            ),
            100.0,
        )


if __name__ == "__main__":
    unittest.main()
