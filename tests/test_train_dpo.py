import tempfile
import unittest
from pathlib import Path

from scripts.train_dpo import build_microbatch_groups, load_dpo_dataset, resolve_adapter_file


class TrainDpoDataTests(unittest.TestCase):
    def test_loader_accepts_pretty_printed_record_stream(self):
        records = [{"id": "a"}, {"id": "b"}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.jsonl"
            path.write_text(
                '{\n  "id": "a"\n}\n\n{\n  "id": "b"\n}\n', encoding="utf-8"
            )
            self.assertEqual(load_dpo_dataset(path), records)

    def test_one_epoch_consumes_each_record_once_with_partial_final_update(self):
        records = [{"id": index} for index in range(9)]
        groups = build_microbatch_groups(
            records, batch_size=2, gradient_accumulation_steps=2,
            num_epochs=1, max_steps=None,
        )

        self.assertEqual([[len(batch) for batch in group] for group in groups], [[2, 2], [2, 2], [1]])
        consumed = [record["id"] for group in groups for batch in group for record in batch]
        self.assertEqual(consumed, list(range(9)))

    def test_max_steps_reuses_complete_updates_only_after_epoch_end(self):
        records = [{"id": index} for index in range(5)]
        groups = build_microbatch_groups(
            records, batch_size=2, gradient_accumulation_steps=2,
            num_epochs=1, max_steps=4,
        )
        self.assertEqual([[record["id"] for batch in group for record in batch] for group in groups], [
            [0, 1, 2, 3], [4], [0, 1, 2, 3], [4],
        ])

    def test_resolve_adapter_file_accepts_directory_and_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = root / "adapters.safetensors"
            adapter.touch()
            self.assertEqual(resolve_adapter_file(root), adapter)
            self.assertEqual(resolve_adapter_file(adapter), adapter)


if __name__ == "__main__":
    unittest.main()
