"""Newport ESP301 3-axis motion controller driver (serial)."""

import logging
import time

import numpy as np
import serial

from gfalib.interface import MotionController

logger = logging.getLogger(__name__)

CR = chr(13)
AXIS_NAMES = {1: "X", 2: "Y", 3: "Z"}


class ESP301(MotionController):
    """Newport ESP301 motor controller — X/Y/Z axes in millimetres.

    Args:
        port: Serial port (default: COM3).
        baud: Baud rate (default: 921600).
    """

    def __init__(
        self,
        port: str = "COM3",
        baud: int = 921600,
        serial_timeout_s: float = 0.1,
    ):
        """Initialise the controller (connection opened by :meth:`open`).

        Args:
            port: Serial port (default: COM3).
            baud: Baud rate (default: 921600).
            serial_timeout_s: Serial read/write timeout in seconds.
        """
        self._port = port
        self._baud = baud
        self._serial_timeout_s = serial_timeout_s
        self._ser = None
        self._available_axes: set = set()
        self._stop_requested: set = set()

    def open(self) -> bool:
        try:
            self._ser = serial.Serial(
                self._port,
                self._baud,
                timeout=self._serial_timeout_s,
                write_timeout=self._serial_timeout_s,
            )
            time.sleep(0.1)
        except Exception as e:
            logger.error("Error abriendo %s: %s: %s",
                         self._port, type(e).__name__, e)
            return False

        self._detect_axes()
        return True

    def _detect_axes(self) -> None:
        """Probe axes 1-3 and record which ones respond to a position query."""
        # Detect available axes
        self._available_axes = set()
        for axis in (1, 2, 3):
            name = AXIS_NAMES[axis]
            try:
                resp = self._send(f"{axis}TP?")
                resp_clean = resp.strip() if resp else ""
                if resp_clean:
                    try:
                        pos = float(resp_clean)
                        logger.info(
                            "Axis %d (%s): OK pos=%.4f mm", axis, name, pos
                        )
                    except ValueError:
                        logger.info(
                            "Axis %d (%s): resp='%s' (non-numeric, included)",
                            axis,
                            name,
                            resp_clean,
                        )
                    self._available_axes.add(axis)
                else:
                    logger.info(
                        "Axis %d (%s): no response — not detected",
                        axis,
                        name,
                    )
            except Exception as e:
                logger.warning("Axis %d (%s): exception (%s)", axis, name, e)

    def move_absolute(self, axis: int, position_mm: float) -> None:
        """Move an axis to an absolute position in millimetres."""
        self._require_axis(axis)
        self._send(f"{axis}PA{position_mm:.6f}")

    def get_position(self, axis: int) -> float:
        """Read the current position of an axis in millimetres."""
        self._require_axis(axis)
        resp = self._send(f"{axis}TP?")
        try:
            p = float(resp)
        except (TypeError, ValueError):
            raise RuntimeError(
                f"ESP301: invalid position on axis {axis}: {resp}")
        if not np.isfinite(p):
            raise RuntimeError(f"ESP301: invalid position on axis {axis}: {p}")
        return p

    def goto_and_wait(
        self,
        axis: int,
        position_mm: float,
        tolerance_mm: float = 0.0005,
        timeout_s: float = 30.0,
    ) -> None:
        """Move to a position and block until arrival, with retries.

        Sequence:
        1. PA  -> move to the absolute position.
        2. WS  -> the controller blocks until the motor stops.
        3. TP? -> read the final position and check the tolerance.

        If the final position is outside tolerance (e.g. mechanical stall),
        retries with a small jiggle for up to max_attempts times.

        Args:
            axis: Axis index (1, 2 or 3).
            position_mm: Target position in millimetres.
            tolerance_mm: Arrival tolerance in millimetres.
            timeout_s: Maximum time to wait for the motor to stop.

        Raises:
            RuntimeError: If the move fails after all retries.
        """
        max_attempts = 3
        self._require_axis(axis)

        for attempt in range(1, max_attempts + 1):
            # Move
            self._send(f"{axis}PA{position_mm:.6f}")

            # Wait for the motor to stop (hardware)
            self._send_wait(f"{axis}WS", timeout_s=timeout_s)

            if axis in self._stop_requested:
                self._stop_requested.remove(axis)
                raise RuntimeError(
                    f"ESP301 axis {axis}: motion stopped")

            # Read the final position
            try:
                p = self.get_position(axis)
            except RuntimeError:
                if attempt >= max_attempts:
                    raise RuntimeError(
                        f"ESP301 axis {axis}: communication lost")
                continue

            if abs(p - position_mm) <= tolerance_mm:
                return  # success

            logger.warning(
                "ESP301 axis %d: WS done but pos=%.6f, target=%.6f "
                "(attempt %d/%d)",
                axis,
                p,
                position_mm,
                attempt,
                max_attempts,
            )

            # Jiggle before retrying
            if attempt < max_attempts:
                jiggle = 0.001 if position_mm > 0 else -0.001
                self._send(f"{axis}PR{jiggle:.6f}")
                self._send_wait(f"{axis}WS", timeout_s=5.0)

        raise RuntimeError(
            f"ESP301 axis {axis}: failed after {max_attempts} attempts. "
            f"Target: {position_mm:.6f} mm"
        )

    def stop_motion(self, axis: int) -> None:
        """Stop motion without blocking for a serial response."""
        if self._ser is not None:
            self._require_axis(axis)
            self._stop_requested.add(axis)
            # ST is an emergency command. Do not use _send(): that method
            # reads a response and may inherit the long timeout of
            # _send_wait(), blocking the GUI thread during the move.
            self._ser.write((f"{axis}ST" + CR).encode("ascii"))
            logger.warning("ESP301 axis %d: stop_motion requested", axis)

    def enable_axis(self, axis: int) -> None:
        """Enable the motor drivers on an axis."""
        self._require_axis(axis)
        self._send(f"{axis}MO")

    def disable_axis(self, axis: int) -> None:
        """Disable the motor drivers on an axis."""
        self._require_axis(axis)
        self._send(f"{axis}MF")

    def close(self) -> None:
        """Close the serial connection."""
        if self._ser:
            self._ser.close()
        self._ser = None

    @property
    def available_axes(self) -> set:
        """Set of available axes, e.g. {1, 2, 3}."""
        return self._available_axes.copy()

    def set_velocity(self, axis: int, vel_mm_s: float) -> None:
        """Set the axis velocity in mm/s (ESP301-specific)."""
        self._require_axis(axis)
        self._send(f"{axis}VA{vel_mm_s:.3f}")

    def _require_axis(self, axis: int) -> None:
        """Validate the connection and axis before a physical command."""
        if not self._ser or not self._ser.is_open:
            raise RuntimeError("ESP301: controller not connected")
        if axis not in AXIS_NAMES:
            raise ValueError(f"ESP301: invalid axis: {axis}")
        if self._available_axes and axis not in self._available_axes:
            raise RuntimeError(f"ESP301: axis {axis} not detected")

    def _send(self, cmd: str, max_wait: float = 0.05) -> str:
        """Send an ASCII command plus CR with an adaptive wait."""
        if not self._ser or not self._ser.is_open:
            return ""
        self._ser.write((cmd + CR).encode("ascii"))

        start = time.time()
        while (time.time() - start) < max_wait:
            if self._ser.in_waiting > 0:
                return self._read()
            time.sleep(0.001)
        return self._read()

    def _send_wait(self, cmd: str, timeout_s: float = 30.0) -> None:
        """Send a blocking command (e.g. WS) and wait until it completes.

        WS sends no response; it blocks the ESP301 command flow until the
        motor stops. To detect completion we send an innocuous query (VE?)
        right after: the controller only processes it once WS releases, at
        which point it answers with the version string.
        """
        if not self._ser or not self._ser.is_open:
            return
        # Send WS
        self._ser.write((cmd + CR).encode("ascii"))
        # Send VE? (version query) as the signal that WS finished
        self._ser.write(b"VE?\r")

        # Wait for the VE? response — it only arrives once WS released
        old_timeout = self._ser.timeout
        self._ser.timeout = timeout_s
        try:
            resp = self._ser.readline().decode(errors="ignore").strip()
            if not resp:
                logger.warning(
                    "_send_wait: timeout after %.1fs for '%s'",
                    timeout_s,
                    cmd,
                )
        finally:
            self._ser.timeout = old_timeout

    def _read(self) -> str:
        if not self._ser:
            return ""
        try:
            return self._ser.readline().decode(errors="ignore").strip()
        except Exception:
            return ""
