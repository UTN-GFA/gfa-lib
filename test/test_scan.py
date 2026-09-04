"""Tests for the spectral-to-depth Scan conversion."""

import numpy as np
from gfalib.interface import Scan


class _ConcreteScan(Scan):
    def zoom_fft(self, irr, axis=-1):  # type: ignore[override]
        return super().zoom_fft(irr, axis=axis)


def test_opd_band_defaults(scan):
    """Default OPD band is positive, ordered, and spans npixels."""
    assert scan.opd_min > 0
    assert scan.opd_max > scan.opd_min
    assert scan.opd_length == scan.npixels


def test_zoom_fft_recovers_reflector_separation(scan):
    """zoom_fft must distinguish reflectors by peak count inside the band."""
    k = 2 * np.pi / scan.wavelength
    d1, d2 = 120.0, 260.0  # nm

    def peak_count(irr):
        """Return the number of distinct in-band peaks above 50% of max."""
        spectrum = scan.zoom_fft(irr)
        assert np.all(np.isfinite(spectrum))
        assert np.all(spectrum >= 0.0)
        in_band = (scan.opd >= scan.opd_min) & (scan.opd < scan.opd_max)
        band = spectrum[in_band]
        # Count local maxima above 50% of the band maximum.
        above = band > 0.5 * band.max()
        return int(np.sum(above))

    single = peak_count(np.cos(k * d1))
    double = peak_count(np.cos(k * d1) + 0.6 * np.cos(k * d2))
    assert single >= 1
    assert double > single


def test_zoom_fft_frame_shape(scan):
    """zoom_fft on a frame returns (nlines, opd_length)."""
    k = 2 * np.pi / scan.wavelength
    irr = np.cos(k * 150.0)
    frame = np.tile(irr, (5, 1))
    spectrum = scan.zoom_fft(frame)
    assert spectrum.shape == (5, scan.opd_length)


def test_wavelength_setter_updates_npixels():
    """Assigning a new wavelength array updates npixels."""
    s = _ConcreteScan(
        npixels=100,
        wavelength_nm=np.linspace(800, 900, 100),
        opd_min_nm=None,
        opd_max_nm=None,
        opd_length=None,
    )
    new_wl = np.linspace(700, 1000, 250)
    s.wavelength = new_wl
    assert s.npixels == 250
    np.testing.assert_array_equal(s.wavelength, new_wl)
