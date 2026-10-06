"""Shared interfaces and the docstring-inheritance helper.

Re-exports the abstract base classes (:class:`Spectrometer`,
:class:`MotionController`, :class:`Scan`) and exposes :func:`docstring`,
which copies a parent method's docstring onto a concrete implementation.
"""

from .motion_controller import MotionController
from .scan import Scan
from .spectrometer import Spectrometer

__all__ = ["Spectrometer", "MotionController", "Scan"]
