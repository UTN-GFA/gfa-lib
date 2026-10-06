"""Wasatch Photonics Cobra HRC 2048 spectrometer driver (pylablib IMAQ)."""

import numpy as np
from pylablib.devices import IMAQ

from gfalib.interface import Spectrometer
from gfalib.util import save_frames

NPIXELS = 2048
BIT_DEPTH = 16
DTYPE = np.uint16
PIXEL_ARRAY = np.arange(2048)
CALIBRATION_CONSTANTS = np.array(
    [7.94223e2, 4.62979e-2, -2.60004e-6, -1.48385e-11])
WAVELENGTHS = np.polynomial.polynomial.polyval(
    PIXEL_ARRAY, CALIBRATION_CONSTANTS).astype(DTYPE)


class CobraHRC2048(Spectrometer):
    """API for controlling the Wasatch Photonics Cobra HRC 2048 spectrometer.

    Communicates with the hardware over IMAQ serial commands to configure
    sensor parameters and acquire raw spectral frames.
    """

    def __init__(self):
        """Initialize WasatchAPI with default sensor parameters."""
        super().__init__()
        self.camera: IMAQ.IMAQCamera | None = None
        self._camera_id: str | None = None
        self._gain: int = 194
        self._offset: int = 255
        self._integration_time: int = 90
        self._line_time: int = 100
        self._camera_on: bool = False

    @property
    def camera_id(self) -> str | None:
        """Connected camera identifier.

        Returns:
            str | None: Camera identifier string, or None if not connected.
        """
        return self._camera_id

    @camera_id.setter
    def camera_id(self, camera_id: str):
        self.select_camera(camera_id)

    @property
    def gain(self) -> int:
        """ADC analog gain register value.

        The Awaiba Dragster sensor uses an inverse ADC gain (analog gain step
        size) from -6 to 20 dB: a higher register value yields lower gain,
        since the firmware may invert this behaviour.

        Returns:
            int: Current gain register value (0-255, default 194).
        """
        return self._gain

    @gain.setter
    def gain(self, value: int):
        _min, _max = 0, 255
        value = max([min([value, _max]), _min])
        self._setter("gain", "gain", value)

    @property
    def offset(self) -> int:
        """ADC black-level offset.

        Lower values raise the baseline; higher values pull it toward black.

        Returns:
            int: Current offset register value (0-255, default 255).
        """
        return self._offset

    @offset.setter
    def offset(self, value: int):
        _min, _max = 0, 255
        value = max([min([value, _max]), _min])
        self._setter("offset", "offset", value)

    @property
    def integration_time(self) -> int:
        """Sensor integration (charge collection) time in microseconds.

        The signal is sampled and held for readout after integration. Must be
        at least 2 us shorter than the line time.

        Returns:
            int: Current integration time (1-32767, default 90).
        """
        return self._integration_time

    @integration_time.setter
    def integration_time(self, value: int):
        _min, _max = 1, 32767
        value = max([min([value, _max]), _min])

        if value > self.line_time:
            self.line_time = value + 2

        self._setter("int", "integration_time", value)

    @property
    def line_time(self) -> int:
        """Line period including transfer/reset overhead in microseconds.

        Sets the maximum achievable line rate.

        Returns:
            int: Current line time (25-65535, default 100).
        """
        return self._line_time

    @line_time.setter
    def line_time(self, value: int):
        _min, _max = 25, 65535
        value = max([min([value, _max]), _min])

        if value < self.integration_time:
            self.integration_time = value - 2

        self._setter("ltm", "line_time", value)

    @property
    def camera_on(self) -> int:
        """Whether the sensor is currently enabled.

        Returns:
            int: 1 if the sensor is enabled, 0 otherwise.
        """
        return self._camera_on

    @camera_on.setter
    def camera_on(self, camera_on: bool) -> None:
        lsc: int = 1 if camera_on else 0
        self._setter("lsc", "camera_on", lsc)

    def _setter(self, key: str, parameter: str, value: int):
        """Send a parameter command and update the local cache.

        Args:
            key: Hardware command key (e.g. 'gain', 'int', 'ltm').
            parameter: Internal attribute name without the leading underscore.
            value: Value to set.

        Raises:
            ValueError: If the hardware command fails.
        """
        try:
            self.set_parameter(key, value)
            self.__setattr__(f"_{parameter}", value)
        except Exception as e:
            raise ValueError(f"Error setting parameter: {e}")

    def list_cameras(self):
        """List available IMAQ cameras.

        Returns:
            list[str]: List of camera identifier strings.
        """
        return IMAQ.list_cameras()

    def message_camera(self, message: str):
        """Send a raw serial command and print the camera response.

        Args:
            message: Raw serial command string.

        Raises:
            ValueError: If no camera is selected.
        """
        if self.camera is None:
            raise ValueError("No camera selected")

        self.camera.serial_write(message)
        print(f"API: {message}\n Wasatch: {self.camera.serial_readline()}")

    def set_parameter(self, parameter: str, value: int):
        """Set a hardware parameter via a serial command.

        Args:
            parameter: Parameter command key.
            value: Integer value to set.

        Raises:
            ValueError: If no camera is selected.
        """
        if self.camera is None:
            raise ValueError("No camera selected")

        self.message_camera(f"{parameter} {value}" + chr(13))

    def select_camera(self, camera_id: str):
        """Open a camera connection and push the stored settings.

        Closes any previously open camera first, sends the ``init`` command,
        then pushes the current gain, offset, integration time and line time
        to the hardware.

        Args:
            camera_id: IMAQ camera identifier string.

        Raises:
            ValueError: If the camera cannot be opened.
        """
        if self.camera is not None:
            self.camera.close()

        try:
            self.camera = IMAQ.IMAQCamera(camera_id)
            self._camera_id = camera_id
            self.message_camera("init\r")
            print(f"Using camera {camera_id}")
            self.gain = self._gain
            self.offset = self._offset
            self.integration_time = self._integration_time
            self.line_time = self._line_time
            self.camera_on = True
        except Exception as e:
            raise ValueError(f"Error selecting camera: {e}")

    def open(self) -> bool:
        cameras = self.list_cameras()
        if not cameras:
            return False
        try:
            self.select_camera(cameras[0])
            return True
        except Exception:
            return False

    def close(self) -> None:
        if self.camera is None:
            return

        try:
            self.camera.close()
        except Exception:
            pass
        self.camera = None
        self._camera_id = None

    def set_exposure_time(self, exposure_time_μs: int):
        self.integration_time = exposure_time_μs

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        if self.camera is None:
            return WAVELENGTHS, np.zeros(NPIXELS, dtype=DTYPE)

        if not self.camera_on:
            self.camera_on = True

        return WAVELENGTHS, np.array(self.camera.snap(), dtype=DTYPE)

    def acquire(self,
                acquisition_time: int,
                save_data: bool = True,
                filename: str | None = None) -> np.ndarray:
        if self.camera is None:
            raise ValueError("No camera selected")

        nframes = 1 + int(acquisition_time / (100 * self.line_time))

        if not self.camera_on:
            self.camera_on = True

        self.camera.setup_acquisition(mode="sequence", nframes=nframes)
        self.camera.set_frame_format("chunks")
        self.camera.start_acquisition()
        self.camera.wait_for_frame(since="lastread", nframes=nframes,
                                   timeout=(None, 5.0))  # type: ignore
        rng = self.camera.get_new_images_range()
        if rng is None:
            frames = np.empty((0, NPIXELS), dtype=DTYPE)
        else:
            chunks = self.camera.read_multiple_images(
                rng=rng, missing_frame="skip",
            )
            # chunks may be a list of 3-D arrays (chunks mode) or a list of
            # 2-D frames (fallback); stack them all into (n_read, 2048).
            if len(chunks) and chunks[0].ndim == 3:
                frames = np.concatenate(chunks, axis=0, dtype=DTYPE)
            else:
                frames = np.vstack(chunks, dtype=DTYPE)

        self.camera.stop_acquisition()
        self.camera.clear_acquisition()

        if save_data:
            save_frames(frames, filename=filename)

        return frames
