"""Abstract spectrometer interface for gfa-lib."""

from abc import abstractmethod

import numpy as np

from .device import Device


class Spectrometer(Device):
    """Spectrometer capable of acquiring spectral irradiance.

    Connection follows the :class:`Device` contract: ``open`` returns a bool
    and never raises, ``close`` is idempotent and never raises.
    """

    @abstractmethod
    def set_exposure_time(self, exposure_time_μs: int):
        """Set the exposure (integration) time.

        Args:
            exposure_time_μs: Exposure time in microseconds.
        """
        ...

    @abstractmethod
    def read(self) -> tuple[np.ndarray, np.ndarray]:
        """Read a single spectrum.

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
            save_data: If True, persist the acquired frames (see each driver
                for the location/format). The frames are always returned
                regardless of this flag.
            filename: Base file name (without extension) used when
                ``save_data`` is True.

        Returns:
            np.ndarray: Acquired frames of shape ``(nlines, npixels)``.

        Raises:
            ValueError: If no camera/spectrometer is selected.
        """
        ...
