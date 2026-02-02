"""Tests for dependency detection helpers in deps.py."""
import unittest
from unittest import mock

import deps


class TestDeps(unittest.TestCase):
    """Validate torch device detection paths without requiring torch installed."""

    def test_detect_torch_device_cuda(self):
        class FakeCuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def get_device_name(_idx):
                return "Fake GPU"

        class FakeBackends:
            class mps:
                @staticmethod
                def is_available():
                    return False

        class FakeTorch:
            cuda = FakeCuda()
            backends = FakeBackends()

        with mock.patch("deps._ensure_torch_imported", return_value=True), mock.patch("deps.torch", FakeTorch):
            info = deps.detect_torch_device()
            self.assertTrue(info["installed"])
            self.assertEqual(info["backend"], "cuda")
            self.assertIn("Fake GPU", info["reason"])

    def test_detect_torch_device_cpu(self):
        class FakeCuda:
            @staticmethod
            def is_available():
                return False

            @staticmethod
            def get_device_name(_idx):
                return "Fake GPU"

        class FakeBackends:
            class mps:
                @staticmethod
                def is_available():
                    return False

        class FakeTorch:
            cuda = FakeCuda()
            backends = FakeBackends()

        with mock.patch("deps._ensure_torch_imported", return_value=True), mock.patch("deps.torch", FakeTorch):
            info = deps.detect_torch_device()
            self.assertTrue(info["installed"])
            self.assertEqual(info["backend"], "cpu")
            self.assertIn("fallback", info["reason"].lower())


if __name__ == "__main__":
    unittest.main()
