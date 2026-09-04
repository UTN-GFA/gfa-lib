"""Spectral-to-depth conversion for Fourier-domain OCT.

Provides :class:`Scan`, which maps a measured irradiance spectrum onto an
optical path difference (OPD, i.e. depth) axis via zoom-FFT.
"""

from abc import ABC

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import interp1d
from scipy.signal import zoom_fft


class Scan(ABC):
    """Spectral-to-depth conversion for Fourier-domain OCT.

    Holds the wavelength calibration and OPD analysis band, and computes
    the depth (OPD) spectrum from measured irradiance via zoom-FFT.
    """

    def __init__(self,
                 npixels: int,
                 wavelength_nm: ArrayLike,
                 opd_min_nm: float | None,
                 opd_max_nm: float | None,
                 opd_length: int | None):
        """Store calibration and analysis-band parameters.

        Args:
            npixels: Number of detector pixels.
            wavelength_nm: Pixel wavelengths in nanometres.
            opd_min_nm: Lower bound of the analysed OPD band, or None for
                the default (``pi / wavenumber_range``).
            opd_max_nm: Upper bound of the analysed OPD band, or None for
                the default (``pi / (2 * wavenumber_step)``).
            opd_length: Number of samples in the OPD axis, or None for
                ``npixels``.
        """
        self._npixels: int = npixels
        self._wavelength: np.ndarray = np.array(wavelength_nm)
        self._opd_min: float | None = opd_min_nm
        self._opd_max: float | None = opd_max_nm
        self._opd_length: int | None = opd_length

    @property
    def npixels(self) -> int:
        """Length of the wavelength array."""
        return self._npixels

    @npixels.setter
    def npixels(self, npixels: int):
        self._npixels = npixels

    @property
    def wavelength(self) -> np.ndarray:
        """Wavelength array in nm, shape (npixels,)."""
        return self._wavelength

    @wavelength.setter
    def wavelength(self, wavelength_nm: ArrayLike):
        wavelength = np.array(wavelength_nm)

        if len(wavelength.shape) > 1:
            raise ValueError(
                f"Expected a 1D array. Received: {wavelength.shape}")

        self._wavelength = wavelength
        self.npixels = len(wavelength)

    @property
    def wavelength_min(self) -> float:
        """Minimum wavelength in nm."""
        return self.wavelength[0]

    @property
    def wavelength_max(self) -> float:
        """Maximum wavelength in nm."""
        return self.wavelength[-1]

    @property
    def wavelength_range(self) -> float:
        """Difference between maximum and minimum wavelength in nm."""
        return self.wavelength_max - self.wavelength_min

    @property
    def wavelength_step(self) -> float:
        """Distance between two consecutive wavelengths in nm."""
        return self.wavelength_range / (self.npixels - 1)

    @property
    def wavenumber_min(self) -> float:
        """Minimum wavenumber in rad/nm corresponding to maximum wavelength."""
        return 2 * np.pi / self.wavelength_max

    @property
    def wavenumber_max(self) -> float:
        """Maximum wavenumber in rad/nm corresponding to minimum wavelength."""
        return 2 * np.pi / self.wavelength_min

    @property
    def wavenumber_range(self) -> float:
        """Difference between maximum and minimum wavenumber in rad/nm."""
        return self.wavenumber_max - self.wavenumber_min

    @property
    def wavenumber_step(self) -> float:
        """Distance between two consecutive wavenumbers in rad/nm."""
        return self.wavenumber_range / (self.npixels - 1)

    @property
    def opd_min(self) -> float:
        """Lower bound of the analysed OPD band in nm."""
        if self._opd_min is not None:
            return self._opd_min
        return np.pi / self.wavenumber_range

    @opd_min.setter
    def opd_min(self, min_opd_nm: float | None) -> None:
        self._opd_min = min_opd_nm

    @property
    def opd_max(self) -> float:
        """Upper bound of the analysed OPD band in nm."""
        if self._opd_max is not None:
            return self._opd_max
        return np.pi / (2 * self.wavenumber_step)

    @opd_max.setter
    def opd_max(self, max_opd_nm: float | None) -> None:
        self._opd_max = max_opd_nm

    @property
    def wavelength_us(self) -> np.ndarray:
        """Uniformly spaced wavelength array in nm."""
        return np.linspace(self.wavelength_min, self.wavelength_max, self.npixels)

    @property
    def wavenumber(self) -> np.ndarray:
        """Wavenumber array in rad/nm, ordered from min to max."""
        return np.flip(2 * np.pi / self.wavelength)

    @property
    def wavenumber_us(self) -> np.ndarray:
        """Uniformly spaced wavenumber array in rad/nm."""
        return np.linspace(self.wavenumber_min, self.wavenumber_max, self.npixels)

    @property
    def opd_length(self) -> int:
        """Length of the opd array."""
        if self._opd_length is not None:
            return self._opd_length
        return self.npixels

    @opd_length.setter
    def opd_length(self, opd_length: int | None) -> None:
        self._opd_length = opd_length

    @property
    def opd(self) -> np.ndarray:
        """Optical path difference (OPD) axis in nm."""
        return np.linspace(self.opd_min, self.opd_max, self.opd_length, endpoint=False)

    def zoom_fft(self, irr: ArrayLike, axis: int = -1) -> np.ndarray:
        """Compute the optical path difference (OPD) spectrum.

        Resamples the interferogram onto a uniform wavenumber grid and applies
        :func:`scipy.signal.zoom_fft` over the analysis band ``[opd_min, opd_max)``.

        Args:
            irr: Measured irradiance ordered by pixel index (matching
                ``self.wavelength``). A single line of shape ``(npixels,)`` or a
                frame of shape ``(nlines, npixels)``.
            axis: Axis over which to compute the FFT. Defaults to the last axis.

        Returns:
            np.ndarray: Absolute OPD spectrum aligned with the ``opd`` axis, of
                shape ``(length_opd,)`` for a line or ``(nlines, length_opd)``
                for a frame.
        """
        irr = np.asarray(irr, dtype=float)
        # Wavenumber per pixel. wavelength is ascending with the pixel index,
        # so the wavenumber is descending; flip irradiance to ascending order.
        k_pixel = 2 * np.pi / self.wavelength
        irr_asc = np.flip(irr, axis=axis)
        irr_us = interp1d(k_pixel, irr_asc, axis=axis)(self.wavenumber_us)

        spectrum = np.abs(
            zoom_fft(
                x=irr_us,
                fn=[self.opd_min, self.opd_max],
                m=self.opd_length,
                fs=np.pi / self.wavenumber_step,  # type: ignore
                endpoint=False,
                axis=axis,
            )
        )

        return spectrum
