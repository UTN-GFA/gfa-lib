"""In-memory mock of the Newport ESP301 motion controller.

Useful for testing acquisition code without physical hardware: motions are
simulated in memory and positions advance over wall-clock time at a configurable
speed.
"""

import logging
import time

from gfalib.interface import MotionController

logger = logging.getLogger(__name__)

AXIS_NAMES = {1: "X", 2: "Y", 3: "Z"}


class MockMC(MotionController):
    """Simulated 3-axis motion controller with in-memory positions.

    Movements are progressive: :meth:`goto_and_wait` advances the position over
    real time at ``motion_speed_mm_s``, honouring stop requests and timeouts.

    Args:
        axes: Set of available axis indices (default: ``{1, 2, 3}`` = X, Y, Z).
        initial_position: Starting position applied to every axis, in mm.
        motion_speed_mm_s: Simulated travel speed in mm/s (must be > 0).
    """

    def __init__(self,
                 axes: set = set(),
                 initial_position: float = 0.0,
                 motion_speed_mm_s: float = 100.0):
        """Initialise the simulated controller (see class docstring for args)."""
        if motion_speed_mm_s <= 0:
            raise ValueError("motion_speed_mm_s debe ser mayor que cero")
        self._axes = axes or {1, 2, 3}
        self._positions = {ax: initial_position for ax in self._axes}
        self._motion_speed_mm_s = float(motion_speed_mm_s)
        self._enabled = {ax: False for ax in self._axes}
        self._connected = False
        self._stop_requested = set()

    def open(self) -> bool:
        self._connected = True
        detected = [AXIS_NAMES.get(a, str(a))
                    for a in sorted(self._axes)]
        logger.info("MockMC conectado, ejes: %s", detected)
        return True

    def close(self):
        self._connected = False
        logger.info("MockMotionController desconectado")

    def move_absolute(self, axis: int, position_mm: float):
        """Set an axis position instantly (no motion simulation).

        Args:
            axis: Axis index.
            position_mm: Target position in millimetres.
        """
        self._check_axis(axis)
        self._positions[axis] = position_mm

    def get_position(self, axis: int) -> float:
        """Return the current position of an axis in millimetres.

        Args:
            axis: Axis index.

        Returns:
            Current position in millimetres.
        """
        self._check_axis(axis)
        return self._positions[axis]

    def goto_and_wait(
        self,
        axis: int,
        position_mm: float,
        tolerance_mm: float = 0.0005,
        timeout_s: float = 30.0,
    ):
        """Move an axis to a target and block until it arrives.

        Advances the position over real time at ``motion_speed_mm_s``, checking
        for stop requests and a wall-clock timeout.

        Args:
            axis: Axis index.
            position_mm: Target position in millimetres.
            tolerance_mm: Position tolerance for arrival, in mm.
            timeout_s: Maximum time to wait, in seconds.

        Raises:
            RuntimeError: If a stop is requested for ``axis``.
            TimeoutError: If the move does not finish within ``timeout_s``.
        """
        self._check_axis(axis)
        if axis in self._stop_requested:
            self._stop_requested.remove(axis)
            raise RuntimeError(f"Movimiento del eje {axis} detenido")
        start_position = self._positions[axis]
        distance = position_mm - start_position
        if abs(distance) <= tolerance_mm:
            self._positions[axis] = position_mm
            return

        started = time.monotonic()
        while True:
            if axis in self._stop_requested:
                self._stop_requested.remove(axis)
                raise RuntimeError(
                    f"Movimiento del eje {axis} detenido")

            elapsed = time.monotonic() - started
            travelled = min(abs(distance), elapsed * self._motion_speed_mm_s)
            fraction = travelled / abs(distance)
            self._positions[axis] = start_position + distance * fraction
            if travelled >= abs(distance) - tolerance_mm:
                self._positions[axis] = position_mm
                break

            if elapsed >= timeout_s:
                raise TimeoutError(f"Timeout moviendo eje {axis}")
            time.sleep(0.005)
        logger.debug("MockMotor eje %d → %.4f mm", axis, position_mm)

    def stop_motion(self, axis: int):
        """Request an immediate stop of an axis' motion.

        Args:
            axis: Axis index.
        """
        self._check_axis(axis)
        self._stop_requested.add(axis)

    def enable_axis(self, axis: int):
        """Enable the motor drivers on an axis.

        Args:
            axis: Axis index.
        """
        self._check_axis(axis)
        self._enabled[axis] = True

    def disable_axis(self, axis: int):
        """Disable the motor drivers on an axis.

        Args:
            axis: Axis index.
        """
        self._check_axis(axis)
        self._enabled[axis] = False

    @property
    def available_axes(self) -> set:
        """Set of available axes, e.g. ``{1, 2, 3}``."""
        return self._axes.copy()

    def _check_axis(self, axis: int) -> None:
        """Raise if ``axis`` is unavailable or the controller is disconnected.

        Args:
            axis: Axis index.

        Raises:
            RuntimeError: If not connected, or ``axis`` is not in
                :attr:`available_axes`.
        """
        if not self._connected:
            raise RuntimeError("MockMotionController: no conectado")
        if axis not in self._axes:
            raise RuntimeError(
                f"MockMotionController: eje {axis} no disponible "
                f"(disponibles: {self._axes})")
