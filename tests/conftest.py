"""Shared pytest fixtures for gfa-lib tests."""

import sys
from pathlib import Path

# Make the ``src`` layout importable (``gfalib`` and ``interfaces`` packages).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from gfalib.motion_controller.MockMC import MockMC
from gfalib.spectrometer.MockSpec import MockSpec
from interfaces.scan import Scan


class _ConcreteScan(Scan):
    """Scan subclass with the abstract surface implemented for testing."""

    def zoom_fft(self, irr, axis=-1):  # type: ignore[override]
        return super().zoom_fft(irr, axis=axis)


@pytest.fixture
def mock_spec() -> MockSpec:
    """Yield an opened MockSpec, closing it afterwards."""
    spec = MockSpec()
    spec.open()
    yield spec
    spec.close()


@pytest.fixture
def mock_mc() -> MockMC:
    """Yield an opened MockMC, closing it afterwards."""
    mc = MockMC()
    mc.open()
    yield mc
    mc.close()


@pytest.fixture
def scan() -> Scan:
    """Build a Scan over a linear 780-920 nm calibration."""
    wavelength_nm = np.linspace(780.0, 920.0, 3648)
    return _ConcreteScan(
        npixels=3648,
        wavelength_nm=wavelength_nm,
        opd_min_nm=None,
        opd_max_nm=None,
        opd_length=None,
    )
