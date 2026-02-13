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
                """
                Indicates whether the CUDA backend is available.
                
                Returns:
                    bool: `True` if CUDA is available, `False` otherwise.
                """
                return True

            @staticmethod
            def get_device_name(_idx: int):
                """
                Return a device name string representing a fake CUDA GPU.
                
                Parameters:
                    _idx (int): Ignored index parameter kept for compatibility.
                
                Returns:
                    str: The fake device name "Fake GPU".
                """
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
        """
        Verify detect_torch_device reports a CPU fallback when neither CUDA nor MPS are available.
        
        Patches the deps module to appear installed and replaces torch with a fake module whose
        cuda.is_available() and backends.mps.is_available() both return False, then asserts
        the detected backend is "cpu" and the reason mentions a fallback.
        """
        class FakeCuda:
            @staticmethod
            def is_available():
                return False

            @staticmethod
            def get_device_name(_idx: int):
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