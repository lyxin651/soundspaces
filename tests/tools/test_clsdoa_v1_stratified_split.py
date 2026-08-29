import unittest

from tools.clsdoa_v1.stratified_source_split import (
    DEFAULT_SALT,
    _quota_counts,
    audit_split,
    build_split,
)


class StratifiedSourceSplitTests(unittest.TestCase):
    def test_quota_for_pilot_pool_has_hard_minima(self):
        for count in range(30, 40):
            train, val, test = _quota_counts(count)
            self.assertEqual(train + val + test, count)
            self.assertGreaterEqual(train, 20)
            self.assertGreaterEqual(val, 4)
            self.assertGreaterEqual(test, 4)

    def test_split_is_per_class_sorted_and_deterministic(self):
        rows = []
        for canonical_class in ("alpha", "beta"):
            for index in range(30):
                rows.append({
                    "canonical_class": canonical_class,
                    "base_clip_id": "{}:{}".format(canonical_class, index),
                    "identity_key": "{}:{}".format(canonical_class, index),
                    "manual_decision": "ACCEPT",
                    "license_status": "DATASET_LEVEL_VERIFIED",
                    "provenance_source": "test",
                })
        first, quotas = build_split(rows, salt=DEFAULT_SALT)
        second, _ = build_split(list(reversed(rows)), salt=DEFAULT_SALT)
        self.assertEqual(first, second)
        report = audit_split(first, quotas, salt=DEFAULT_SALT)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["split_overlap"], {
            "train_vs_val": [], "train_vs_test": [], "val_vs_test": [],
        })
        for item in report["per_class"].values():
            self.assertEqual((item["train"], item["val"], item["test"]), (20, 5, 5))

    def test_split_rejects_non_accepted_or_duplicate_identity(self):
        row = {
            "canonical_class": "alpha", "base_clip_id": "a",
            "identity_key": "same", "manual_decision": "REJECT",
            "license_status": "DATASET_LEVEL_VERIFIED", "provenance_source": "test",
        }
        with self.assertRaises(ValueError):
            build_split([row])

        row["manual_decision"] = "ACCEPT"
        with self.assertRaises(ValueError):
            build_split([row, dict(row, base_clip_id="b")])


if __name__ == "__main__":
    unittest.main()
