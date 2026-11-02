#!/usr/bin/env python3
"""Regression tests for cleaning model-generated MBPP Python code."""

import unittest

from run_mbpp_evalplus import (
    extract_code,
    remove_top_level_asserts,
    validate_python_candidate,
)


class TestMBPPCodeCleaning(unittest.TestCase):
    def test_remove_top_level_assert_and_keep_function(self):
        raw = "def f(x):\n    return x + 1\n\nassert f(2) == 7\n"
        cleaned, count = remove_top_level_asserts(raw)
        self.assertEqual(count, 1)
        self.assertEqual(cleaned, "def f(x):\n    return x + 1")
        self.assertEqual(validate_python_candidate(cleaned, "f")[:2], (True, True))

    def test_keep_assert_inside_function(self):
        raw = "def f(x):\n    assert x > 0\n    return x\n"
        cleaned, count = remove_top_level_asserts(raw)
        self.assertEqual(count, 0)
        self.assertEqual(cleaned, raw)

    def test_remove_multiline_top_level_assert(self):
        raw = (
            "import math\n\n"
            "def f(x):\n    return math.sqrt(x)\n"
            "assert (\n    f(4) == 3\n)\n"
        )
        cleaned, count = remove_top_level_asserts(raw)
        self.assertEqual(count, 1)
        self.assertIn("import math", cleaned)
        self.assertNotIn("assert", cleaned)
        self.assertEqual(validate_python_candidate(cleaned, "f")[:2], (True, True))

    def test_leave_invalid_python_for_validity_check(self):
        raw = "def f(:\n assert f(1) == 2\n"
        cleaned, count = remove_top_level_asserts(raw)
        self.assertEqual(count, 0)
        self.assertEqual(cleaned, raw)
        self.assertFalse(validate_python_candidate(cleaned, "f")[0])

    def test_multiple_asserts_and_other_statements(self):
        raw = "x = 3\nassert x == 3\nassert x > 4\ndef f():\n    return x\n"
        cleaned, count = remove_top_level_asserts(raw)
        self.assertEqual(count, 2)
        self.assertIn("x = 3", cleaned)
        self.assertIn("def f():", cleaned)
        self.assertNotIn("assert ", cleaned)

    def test_fenced_model_response_cleaned_for_oracle(self):
        raw = (
            "```python\n"
            "def maxAverageOfPath(matrix):\n    return 5.2\n"
            "assert maxAverageOfPath([1]) == 0\n"
            "```"
        )
        extracted, note = extract_code(raw, "maxAverageOfPath")
        cleaned, count = remove_top_level_asserts(extracted)
        self.assertIn("python fence", note)
        self.assertEqual(count, 1)
        self.assertEqual(validate_python_candidate(cleaned, "maxAverageOfPath")[:2], (True, True))
        namespace = {}
        exec(cleaned, namespace)
        self.assertEqual(namespace["maxAverageOfPath"]([1]), 5.2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
