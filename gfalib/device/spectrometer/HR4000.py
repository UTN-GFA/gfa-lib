"""Ocean Optics HR4000 spectrometer driver (seabreeze/USB)."""

import logging
import time

import numpy as np
from seabreeze.spectrometers import Spectrometer as SeaBreeze

from gfalib.interface import Spectrometer
from gfalib.util import save_frames

logger = logging.getLogger(__name__)


class HR4000(Spectrometer):
    """Ocean Optics HR4000 control via seabreeze.

    Dark correction
    ---------------
    Uses the CCD's electrically masked pixels (Toshiba TCD1304AP): the average
    of these optically shielded pixels is subtracted from every reading. This
    removes the dark-current offset precisely, without depending on the signal
    minimum. seabreeze applies it internally through
    ``intensities(correct_dark_counts=True)``.

    Nonlinearity correction
    -----------------------
    The HR4000 stores nonlinearity correction coefficients in its EEPROM (a
    factory-calibrated polynomial up to order 7). seabreeze applies them when
    calling ``intensities(correct_nonlinearity=True)``. This corrects the CCD's
    nonlinear response, which matters so the FFT in the OCT pipeline works on
    intensities proportional to the true optical signal.

    Acquisition model
    -----------------
    seabreeze is request/response: there is no hardware frame buffer, so
    :meth:`acquire` performs a software-timed loop, requesting one spectrum per
    integration time until ``acquisition_time`` elapses.
    """

    def __init__(self):
        """Initialise the spectrometer in a disconnected state."""
        self._spec: SeaBreeze | None = None
        self._connected: bool = False
        self._dark_enabled: bool = True
        self._nonlinearity_enabled: bool = True
        self.integration_time: int = 10000

    def open(self) -> bool:
        if self._spec is None:
            try:
                self._spec = SeaBreeze.from_first_available()
            except Exception as e:
                logger.error("HR4000: no se pudo conectar: %s", e)
                self._spec = None
        self._connected = self._spec is not None
        return self._connected

    def close(self) -> None:
        self._connected = False
        if self._spec is not None:
            try:
                self._spec.close()
            except Exception:
                pass
        self._spec = None

    def _invalidate(self):
        """Mark the connection as lost after a communication error."""
        if self._connected:
            logger.error(
                "HR4000: communication lost — device disconnected")
        self._connected = False
        if self._spec is not None:
            try:
                self._spec.close()
            except Exception:
                pass
        self._spec = None

    @property
    def is_connected(self) -> bool:
        """Whether the spectrometer is currently connected."""
        return self._connected and self._spec is not None

    def set_exposure_time(self, exposure_time_μs: int):
        self.integration_time = exposure_time_μs
        if self._spec is None:
            raise RuntimeError("HR4000: no conectado")
        try:
            self._spec.integration_time_micros(  # type: ignore
                exposure_time_μs)
        except Exception as e:
            self._invalidate()
            raise RuntimeError(f"HR4000: comunicación perdida ({e})")

    def read(self):
        if not self._connected or self._spec is None:
            raise RuntimeError("HR4000: no conectado")

        try:
            wavelengths_nm = self._spec.wavelengths()
            intensities = self._spec.intensities(
                correct_dark_counts=self._dark_enabled,
                correct_nonlinearity=self._nonlinearity_enabled,
            )
        except Exception as e:
            self._invalidate()
            raise RuntimeError(f"HR4000: comunicación perdida ({e})")

        if wavelengths_nm is None or intensities is None:
            raise RuntimeError("HR4000: espectrómetro no responde")

        return np.asarray(wavelengths_nm), np.asarray(intensities)

    def acquire(self,
                acquisition_time: int,
                save_data: bool = True,
                filename: str | None = None) -> np.ndarray:
        if not self._connected or self._spec is None:
            raise RuntimeError("HR4000: no conectado")

        # seabreeze is request/response with no hardware frame buffer, so we
        # loop in software: one spectrum per integration time.
        frame_time_us = max(int(self._spec.integration_time_micros_limits[0]),
                            self.integration_time)
        nframes = max(1, int(acquisition_time / frame_time_us))

        frames = []
        for _ in range(nframes):
            try:
                _, intensities = self.read()
            except Exception as e:
                self._invalidate()
                raise RuntimeError(f"HR4000: comunicación perdida ({e})")
            frames.append(intensities)
            time.sleep(frame_time_us / 1_000_000.0)

        stacked = np.vstack(frames)

        if save_data:
            save_frames(stacked, filename=filename)

        return stacked
