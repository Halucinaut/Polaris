import tempfile
import unittest
from pathlib import Path

from polaris.eval_paths import prepare_output_path
from scripts.smoke.eval_utils import select_eval_datasets


class EvalGsm8kCliTests(unittest.TestCase):
    def test_output_path_parent_is_created_before_predictions_are_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "new_eval" / "results.json"
            resolved = prepare_output_path(str(output_path))
            self.assertEqual(resolved, output_path)
            self.assertTrue(output_path.parent.is_dir())
            prediction_path = resolved.parent / "test_predictions.jsonl"
            prediction_path.write_text('{"problem_id":"p","prediction":"x"}\n', encoding="utf-8")
            self.assertTrue(prediction_path.is_file())


class SelectEvalDatasetsTests(unittest.TestCase):
    def test_skip_review_false_returns_test_then_review(self):
        result = select_eval_datasets("test.jsonl", "review.jsonl", skip_review=False)
        self.assertEqual(result, [("test", "test.jsonl"), ("review", "review.jsonl")])

    def test_skip_review_true_returns_test_only(self):
        result = select_eval_datasets("test.jsonl", "review.jsonl", skip_review=True)
        self.assertEqual(result, [("test", "test.jsonl")])

    def test_default_order_is_test_first(self):
        result = select_eval_datasets("a.jsonl", "b.jsonl", skip_review=False)
        self.assertEqual(result[0], ("test", "a.jsonl"))
        self.assertEqual(result[1][0], "review")

    def test_preserves_path_strings_verbatim(self):
        result = select_eval_datasets("/abs/path/t.jsonl", "/abs/path/r.jsonl", skip_review=False)
        self.assertEqual(result[0][1], "/abs/path/t.jsonl")
        self.assertEqual(result[1][1], "/abs/path/r.jsonl")


if __name__ == "__main__":
    unittest.main()
