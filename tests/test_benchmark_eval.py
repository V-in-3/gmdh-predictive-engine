import csv
import os
import tempfile
import unittest

from scripts.benchmark_eval import evaluate_benchmark_model


class BenchmarkEvaluationTest(unittest.TestCase):
    def test_evaluate_benchmark_model_returns_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "benchmark.csv")
            with open(input_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["TransactionAmt", "TransactionDT", "proxy_flag", "isFraud"])
                writer.writeheader()
                writer.writerow({"TransactionAmt": 100, "TransactionDT": 5, "proxy_flag": 0, "isFraud": 0})
                writer.writerow({"TransactionAmt": 200, "TransactionDT": 10, "proxy_flag": 1, "isFraud": 1})
                writer.writerow({"TransactionAmt": 300, "TransactionDT": 6, "proxy_flag": 0, "isFraud": 0})
                writer.writerow({"TransactionAmt": 400, "TransactionDT": 11, "proxy_flag": 1, "isFraud": 1})

            output_path = os.path.join(tmpdir, "normalized.csv")
            model_path = os.path.join(tmpdir, "model.json")
            metrics_path = os.path.join(tmpdir, "metrics.json")

            result = evaluate_benchmark_model(
                input_path=input_path,
                output_path=output_path,
                model_output_path=model_path,
                metrics_output_path=metrics_path,
            )

            self.assertTrue(os.path.exists(output_path))
            self.assertTrue(os.path.exists(model_path))
            self.assertTrue(os.path.exists(metrics_path))
            self.assertIn("metrics", result)
            self.assertIn("auc_roc", result["metrics"])
            self.assertGreaterEqual(result["metrics"]["auc_roc"], 0.0)


if __name__ == "__main__":
    unittest.main()
