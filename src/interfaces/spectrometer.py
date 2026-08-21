"""Abstract spectrometer interface for gfa-lib."""

from abc import abstractmethod

import numpy as np

from .device import Device


class Spectrometer(Device):
    """Spectrometer capable of acquiring spectral irradiance."""

    @abstractmethod
    def set_exposure_time(self, exposure_time_μs: int):
        """Set the exposure time.

        Args:
            exposure_time_μs: Exposure time in microseconds.
        """
        ...

    @abstractmethod
    def read(self) -> tuple[np.ndarray, np.ndarray]:
        """Read a spectrum.

        Returns:
            tuple[np.ndarray, np.ndarray]: A (wavelengths, intensities) pair,
                where wavelengths has shape ``(npixels,)`` in nanometres and
                intensities has shape ``(nlines, npixels)``.
        """
        ...

    @abstractmethod
    def acquire(self,
                acquisition_time: int,
                save_data: bool = True,
                filename: str | None = None) -> np.ndarray:
        """Acquire multiple frames over a given duration.

        Args:
            acquisition_time: Total acquisition time in microseconds.
            save_data: If True, save the acquired frames to a ``.npy`` file.
            filename: Base file name (without extension) used when
                ``save_data`` is True.

        Returns:
            np.ndarray: Acquired frames of shape ``(nlines, npixels)``.

        Raises:
            ValueError: If no camera is selected.
        """
        ...
