"""Abstract device interface for gfa-lib hardware."""

from abc import ABC, abstractmethod


class Device(ABC):
    """Base class for any connectable hardware device.

    Connection contract
    -------------------
    ``open()`` returns ``False`` (never raises) when the device cannot be
    reached, so callers can branch on the boolean. ``close()`` is idempotent:
    calling it with no active connection is a no-op and it never raises.
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
