"""Simulated spectrometer for Fourier-domain OCT testing.

Generates a spectral interferogram from a set of synthetic axial reflectors,
with optional noise and dark counts, so the OCT pipeline can be exercised
without physical hardware.
"""

import logging
import time

import numpy as np

from gfalib.interface import Spectrometer

logger = logging.getLogger(__name__)


class Reflector:
    """A synthetic point reflector in the axial domain."""

    def __init__(self, depth_m: float, amplitude: float = 1.0):
        """Define a reflector.

        Args:
            depth_m: Optical path length of the reflector, in metres.
            amplitude: Reflectivity amplitude (default: 1.0).
        """
        self.depth_m = depth_m
        self.amplitude = amplitude


class MockSpec(Spectrometer):
    """Simulated spectrometer producing interferograms from reflectors.

    Args:
        wavelength_min_nm: Lower spectral bound in nanometres.
        wavelength_max_nm: Upper spectral bound in nanometres.
        npixels: Number of detector pixels.
        reflectors: Synthetic reflectors. Defaults to two reflectors at
            0.5 mm and 1.2 mm depth.
        noise_level: Relative noise level (0 = no noise).
        exposure_time_μs: Integration time in microseconds.
        dark_counts: Simulated dark offset, either a scalar or one value
            per pixel.
    """

    def __init__(self,
                 wavelength_min_nm: float = 780.0,
                 wavelength_max_nm: float = 920.0,
                 npixels: int = 3648,
                 reflectors: list[Reflector] | None = None,
                 noise_level: float = 0.02,
                 exposure_time_μs: int = 10000,
                 dark_counts: float = 0.0):
        """Initialise the simulated spectrometer (see class docstring for args)."""
        self.wavelength_min = wavelength_min_nm
        self.wavelength_max = wavelength_max_nm
        self.npixels = npixels
        self.noise_level = noise_level
        self._exposure_time = exposure_time_μs
        self._connected = False
        self._dark_enabled = False
        dark = np.asarray(dark_counts, dtype=float)
        if dark.ndim > 1 or (dark.ndim == 1 and dark.size != npixels):
            raise ValueError(
                "dark_counts debe ser escalar o tener un valor por píxel"
            )
        self._dark_counts = dark.copy()

        if not reflectors:
            self.reflectors = [
                Reflector(depth_m=0.5e-3, amplitude=1.0),
                Reflector(depth_m=1.2e-3, amplitude=0.6),
            ]
        else:
            self.reflectors = reflectors

        self._wavelengths_nm = np.linspace(
            wavelength_min_nm, wavelength_max_nm, npixels
        )

    def open(self) -> bool:
        self._connected = True
        logger.info("MockSpectrometer conectado (simulación)")
        return True

    def close(self):
        self._connected = False
        logger.info("MockSpectrometer desconectado")

    def set_exposure_time(self, exposure_time_μs: int):
        self._exposure_time = exposure_time_μs

    def read(self):
        """Acquire one spectrum from the simulated reflectors.

        Returns:
            tuple[np.ndarray, np.ndarray]: The wavelength array (nm) and the
            measured irradiances, with dark correction applied when enabled.

        Raises:
            RuntimeError: If called before :meth:`open`.
        """
        if not self._connected:
            raise RuntimeError(
                "MockSpectrometer: read() llamado sin conectar")

        time.sleep(min(self._exposure_time / 1000.0, 0.05))

        intensities = self._generate_interferogram()

        if self._dark_enabled:
            intensities = np.maximum(intensities - self._dark_counts, 0.0)

        # Nonlinearity: no-op en simulación (la señal sintética
        # ya es lineal, no hay respuesta de CCD que corregir).

        return self._wavelengths_nm.copy(), intensities

    def acquire(self,
                acquisition_time: int,
                save_data: bool = True,
                filename: str | None = None) -> np.ndarray:
        if not self._connected:
            raise RuntimeError(
                "MockSpectrometer: acquire() llamado sin conectar")

        # Software-timed loop: one frame per integration time.
        frame_time_us = max(1, self._exposure_time)
        nframes = max(1, int(acquisition_time / frame_time_us))

        frames = np.tile(
            self._generate_interferogram(), (nframes, 1)
        ).astype(np.float64)

        if save_data:
            from gfalib.util import save_frames
            save_frames(frames, filename=filename)

        return frames

    def _generate_interferogram(self) -> np.ndarray:
        """Build a synthetic interferogram from the configured reflectors.

        Returns:
            np.ndarray: Irradiance per pixel, scaled to a 16-bit range.
        """
        wavelengths_m = self._wavelengths_nm * 1e-9
        k = 2 * np.pi / wavelengths_m

        k_center = k.mean()
        k_sigma = (k.max() - k.min()) / 4
        envelope = np.exp(-0.5 * ((k - k_center) / k_sigma) ** 2)

        interference = np.zeros_like(k)
        for r in self.reflectors:
            interference += r.amplitude * np.cos(2 * k * r.depth_m)

        signal = envelope * (1.0 + 0.3 * interference)

        if self.noise_level > 0:
            noise = np.random.normal(0, self.noise_level * signal.max(),
                                     size=len(signal))
            signal += noise

        signal = np.clip(signal, 0, None)
        signal = signal / signal.max() * 50000

        return signal
