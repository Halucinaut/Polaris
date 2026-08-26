"""Tests for train_sft.py init adapter resolution (no MLX required)."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_sft import (
    build_sft_checkpoint_provenance,
    resolve_init_adapter_file,
    resolve_init_adapter_path,
)


class TestResolveInitAdapterPath(unittest.TestCase):
    def test_cli_overrides_config(self):
        config = {"sft": {"init_adapter_path": "/from/config"}}
        self.assertEqual(resolve_init_adapter_path(config, "/from/cli"), "/from/cli")

    def test_config_fallback_when_no_cli(self):
        config = {"sft": {"init_adapter_path": "/from/config"}}
        self.assertEqual(resolve_init_adapter_path(config, None), "/from/config")

    def test_no_config_no_cli_returns_none(self):
        self.assertIsNone(resolve_init_adapter_path({}))
        self.assertIsNone(resolve_init_adapter_path({}, None))

    def test_empty_sft_section(self):
        self.assertIsNone(resolve_init_adapter_path({"sft": {}}))

    def test_empty_string_cli_ignored_uses_config(self):
        config = {"sft": {"init_adapter_path": "/from/config"}}
        self.assertEqual(resolve_init_adapter_path(config, ""), "/from/config")

    def test_whitespace_cli_ignored_uses_config(self):
        config = {"sft": {"init_adapter_path": "/from/config"}}
        self.assertEqual(resolve_init_adapter_path(config, "  "), "/from/config")

    def test_none_config_value_returns_none(self):
        config = {"sft": {"init_adapter_path": None}}
        self.assertIsNone(resolve_init_adapter_path(config, None))

    def test_whitespace_config_value_returns_none(self):
        config = {"sft": {"init_adapter_path": "  "}}
        self.assertIsNone(resolve_init_adapter_path(config, None))


class TestResolveInitAdapterFile(unittest.TestCase):
    def test_directory_resolves_to_adapters_safetensors(self):
        with tempfile.TemporaryDirectory() as td:
            adapter_file = Path(td) / "adapters.safetensors"
            adapter_file.write_text("dummy")
            result = resolve_init_adapter_file(td)
            self.assertEqual(result, adapter_file)

    def test_direct_safetensors_path(self):
        with tempfile.TemporaryDirectory() as td:
            adapter_file = Path(td) / "adapters.safetensors"
            adapter_file.write_text("dummy")
            result = resolve_init_adapter_file(str(adapter_file))
            self.assertEqual(result, adapter_file)

    def test_missing_directory_raises(self):
        with self.assertRaises(FileNotFoundError):
            resolve_init_adapter_file("/nonexistent/path")

    def test_missing_file_in_directory_raises(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                resolve_init_adapter_file(td)

    def test_non_safetensors_extension_raises(self):
        with tempfile.TemporaryDirectory() as td:
            bad_file = Path(td) / "adapters.bin"
            bad_file.write_text("dummy")
            with self.assertRaises(ValueError) as ctx:
                resolve_init_adapter_file(str(bad_file))
            self.assertIn(".safetensors", str(ctx.exception))

    def test_non_safetensors_file_raises(self):
        with tempfile.TemporaryDirectory() as td:
            bad_file = Path(td) / "model.pt"
            bad_file.write_text("dummy")
            with self.assertRaises(ValueError):
                resolve_init_adapter_file(str(bad_file))


class TestBuildSftCheckpointProvenance(unittest.TestCase):
    def test_without_adapter(self):
        p = build_sft_checkpoint_provenance("models/qwen3", None, False)
        self.assertEqual(p["base_model_path"], "models/qwen3")
        self.assertIsNone(p["init_adapter_file"])
        self.assertFalse(p["init_adapter_loaded"])
        self.assertEqual(p["response_supervision"], "target_only_next_token")

    def test_with_adapter(self):
        p = build_sft_checkpoint_provenance(
            "models/qwen3", Path("/adapter/adapters.safetensors"), True,
        )
        self.assertEqual(p["init_adapter_file"], "/adapter/adapters.safetensors")
        self.assertTrue(p["init_adapter_loaded"])

    def test_required_keys(self):
        p = build_sft_checkpoint_provenance("m", None, False)
        for key in (
            "base_model_path",
            "init_adapter_file",
            "init_adapter_loaded",
            "response_supervision",
        ):
            self.assertIn(key, p)


if __name__ == "__main__":
    unittest.main()
