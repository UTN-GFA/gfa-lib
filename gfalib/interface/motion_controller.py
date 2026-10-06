"""Abstract motion-controller interface for gfa-lib."""

from abc import ABC, abstractmethod


class MotionController(ABC):
    """Motion controller with positions in canonical millimetres.

    Each implementation converts between millimetres and the hardware's
    native units before sending or interpreting commands.
    """

    @abstractmethod
    def open(self) -> bool:
        """Open the connection to the device.

        Returns:
            bool: True if the connection succeeded, False otherwise.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the connection.

        Idempotent: safe to call when already disconnected, and never raises.
        """
        ...

    @abstractmethod
    def move_absolute(self, axis: int, position_mm: float) -> None:
        """Move an axis to an absolute position in millimetres.

        Args:
            axis: Axis index (1, 2 or 3).
            position_mm: Target position in millimetres.
        """
        ...

    @abstractmethod
    def get_position(self, axis: int) -> float:
        """Read the current position of an axis.

        Args:
            axis: Axis index (1, 2 or 3).

        Returns:
            float: Current position in millimetres.
        """
        ...

    @abstractmethod
    def goto_and_wait(
        self,
        axis: int,
        position_mm: float,
        tolerance_mm: float = 0.0005,
        timeout_s: float = 30.0,
    ) -> None:
        """Move to a canonical position and block until arrival.

        Args:
            axis: Axis index (1, 2 or 3).
            position_mm: Target position in millimetres, regardless of the
                hardware's native unit.
            tolerance_mm: Arrival tolerance in millimetres.
            timeout_s: Maximum time to wait for arrival, in seconds.
        """
        ...

    @abstractmethod
    def stop_motion(self, axis: int) -> None:
        """Stop the current motion of an axis without disabling it."""
        ...

    @abstractmethod
    def enable_axis(self, axis: int) -> None:
        """Enable the motor drivers on an axis."""

    @abstractmethod
    def disable_axis(self, axis: int) -> None:
        """Disable the motor drivers on an axis."""

    @property
    @abstractmethod
    def available_axes(self) -> set:
        """Set of available axes, e.g. {1, 2, 3}."""
        ...
