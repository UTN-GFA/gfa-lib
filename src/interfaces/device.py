"""Abstract device interface for gfa-lib hardware."""

from abc import ABC, abstractmethod


class Device(ABC):
    """Base class for any connectable hardware device."""

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

        Raises:
            ValueError: If no connection is currently active.
        """
        ...
