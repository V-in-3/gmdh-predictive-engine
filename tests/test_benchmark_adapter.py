import unittest

from scripts.benchmark_eval import normalize_benchmark_dataset


class BenchmarkAdapterTest(unittest.TestCase):
    def test_normalize_benchmark_dataset_creates_required_columns(self):
        rows = [
            {
                "TransactionAmt": 100.0,
                "TransactionDT": 1,
                "DeviceType": "desktop",
                "proxy_flag": 0,
                "merchant_name": "gift card",
                "isFraud": 0,
            },
            {
                "TransactionAmt": 150.0,
                "TransactionDT": 2,
                "DeviceType": "mobile",
                "proxy_flag": 1,
                "merchant_name": "electronics",
                "isFraud": 1,
            },
            {
                "TransactionAmt": 200.0,
                "TransactionDT": 3,
                "DeviceType": "mobile",
                "proxy_flag": 0,
                "merchant_name": "office",
                "isFraud": 0,
            },
        ]

        normalized = normalize_benchmark_dataset(rows, source_type="ieee_cis")
        self.assertTrue(normalized)
        for row in normalized:
            self.assertIn("semantic_risk", row)
            self.assertIn("velocity_1h", row)
            self.assertIn("proxy_score", row)
            self.assertIn("amount_deviation", row)
            self.assertIn("is_fraud", row)
            self.assertIn(row["is_fraud"], (0, 1))
            self.assertGreaterEqual(row["proxy_score"], 0)
            self.assertLessEqual(row["proxy_score"], 1)
            self.assertGreaterEqual(row["semantic_risk"], 0)
            self.assertLessEqual(row["semantic_risk"], 1)


if __name__ == "__main__":
    unittest.main()
