import json
import tempfile
import unittest
from pathlib import Path

from scripts.train_sft import load_sft_dataset


class LoadSftDatasetTests(unittest.TestCase):
    def test_loads_standard_jsonl(self):
        records = [{"id": "one"}, {"id": "two"}]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dataset.jsonl"
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            self.assertEqual(load_sft_dataset(path), records)

    def test_loads_pretty_printed_json_record_stream(self):
        records = [{"id": "one", "target": "first"}, {"id": "two", "target": "second"}]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dataset.jsonl"
            path.write_text(
                "\n".join(json.dumps(record, indent=2) for record in records),
                encoding="utf-8",
            )

            self.assertEqual(load_sft_dataset(path), records)


if __name__ == "__main__":
    unittest.main()
