"""
Automated Test Suite for AI Topic 01 Lab:
AI Training Batch Stream Validator & Memory Profiler
"""

import unittest
from main import (
    FLAG_VALID,
    FLAG_CONTAINS_NAN,
    FLAG_DIM_MISMATCH,
    FLAG_ZERO_WEIGHT,
    validate_sample,
    profile_object_memory,
    format_batch_summary
)

class TestAIPythonFundamentals(unittest.TestCase):

    def test_bitwise_constants(self):
        """Verify bitwise flags are powers of two."""
        self.assertEqual(FLAG_VALID, 1)
        self.assertEqual(FLAG_CONTAINS_NAN, 2)
        self.assertEqual(FLAG_DIM_MISMATCH, 4)
        self.assertEqual(FLAG_ZERO_WEIGHT, 8)

    def test_validate_valid_sample(self):
        """Test clean sample passes with FLAG_VALID."""
        raw = {
            "sample_id": "sample_001",
            "features": [0.12345, 1.67891, -0.45678, 2.0],
            "label": "1",
            "loss_weight": "1.0"
        }
        cleaned, status = validate_sample(raw, expected_dim=4)
        self.assertEqual(cleaned["sample_id"], "SAMPLE_001")
        self.assertEqual(cleaned["label"], 1)
        self.assertEqual(cleaned["loss_weight"], 1.0)
        self.assertEqual(len(cleaned["features"]), 4)
        self.assertEqual(status, FLAG_VALID)

    def test_validate_nan_sample(self):
        """Test sample containing NaN triggers FLAG_CONTAINS_NAN."""
        raw = {
            "sample_id": "sample_002",
            "features": [0.1, "nan", 0.3, 0.4],
            "label": 0,
            "loss_weight": 1.0
        }
        cleaned, status = validate_sample(raw, expected_dim=4)
        self.assertTrue(bool(status & FLAG_CONTAINS_NAN))
        self.assertFalse(bool(status & FLAG_VALID))

    def test_validate_dimension_mismatch(self):
        """Test sample with wrong feature length triggers FLAG_DIM_MISMATCH."""
        raw = {
            "sample_id": "sample_003",
            "features": [0.1, 0.2],
            "label": 1,
            "loss_weight": 1.0
        }
        cleaned, status = validate_sample(raw, expected_dim=4)
        self.assertTrue(bool(status & FLAG_DIM_MISMATCH))
        self.assertFalse(bool(status & FLAG_VALID))

    def test_validate_zero_weight(self):
        """Test sample with zero or negative loss weight triggers FLAG_ZERO_WEIGHT."""
        raw = {
            "sample_id": "sample_004",
            "features": [0.1, 0.2, 0.3, 0.4],
            "label": 0,
            "loss_weight": 0.0
        }
        cleaned, status = validate_sample(raw, expected_dim=4)
        self.assertTrue(bool(status & FLAG_ZERO_WEIGHT))
        self.assertFalse(bool(status & FLAG_VALID))

    def test_profile_memory(self):
        """Test memory profiler on mutable and immutable objects."""
        sample_tensor_data = [0.1, 0.2, 0.3]
        prof_mut = profile_object_memory(sample_tensor_data, "Features List")
        self.assertEqual(prof_mut["type"], "list")
        self.assertTrue(prof_mut["is_mutable"])
        self.assertTrue(prof_mut["memory_id_hex"].startswith("0x"))

        sample_name = "BATCH_ID_001"
        prof_imm = profile_object_memory(sample_name, "Batch Name")
        self.assertEqual(prof_imm["type"], "str")
        self.assertFalse(prof_imm["is_mutable"])

    def test_format_batch_summary(self):
        """Test batch telemetry string formatting."""
        sample_cleaned = [{
            "sample_id": "SAMPLE_001",
            "features": [0.1, 0.2, 0.3, 0.4],
            "label": 1,
            "loss_weight": 1.0
        }]
        sample_status = [FLAG_VALID]
        output = format_batch_summary(sample_cleaned, sample_status)
        self.assertIsInstance(output, str)
        self.assertIn("SAMPLE_001", output)


if __name__ == "__main__":
    unittest.main()
