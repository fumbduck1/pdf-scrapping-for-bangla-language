"""Additional tests for configuration manager validation."""
import os
import tempfile
import unittest
from pathlib import Path

from config_manager import create_job_config, get_config_manager


class TestConfigManagerExtra(unittest.TestCase):
    """Broaden coverage for config validation edge cases."""

    def test_missing_pdf_triggers_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "out"
            cfg = create_job_config(input_path=str(Path(tmp) / "missing.pdf"), output_root=str(output_dir))
            errors = get_config_manager().validate_config(cfg)
            self.assertTrue(any("file not found" in e.lower() for e in errors))

    def test_output_dir_created_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf_file = Path(tmp) / "file.pdf"
            pdf_file.write_bytes(b"%PDF-1.4\n")
            output_dir = Path(tmp) / "nested" / "out"
            cfg = create_job_config(input_path=str(pdf_file), output_root=str(output_dir))
            errors = get_config_manager().validate_config(cfg)
            self.assertFalse(errors)
            self.assertTrue(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
