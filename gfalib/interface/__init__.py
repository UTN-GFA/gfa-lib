"""Shared interfaces and the docstring-inheritance helper.

Re-exports the abstract base classes (:class:`Spectrometer`,
:class:`MotionController`, :class:`Scan`) and exposes :func:`docstring`,
which copies a parent method's docstring onto a concrete implementation.
"""

from .motion_controller import MotionController
from .scan import Scan
from .spectrometer import Spectrometer

__all__ = ["Spectrometer", "MotionController", "Scan", "docstring"]


def docstring(parent_method):
    """Copy ``parent_method``'s docstring onto the decorated method.

    Args:
        parent_method: Method whose docstring should be inherited.

    Returns:
        callable: A decorator that assigns the parent docstring to the
        wrapped method and returns it unchanged.
    """

    def decorator(method):
        method.__doc__ = parent_method.__doc__
        return method

    return decorator
