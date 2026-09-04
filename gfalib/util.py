"""Shared helpers for the gfa-lib drivers."""

from datetime import datetime
from pathlib import Path

import numpy as np


def default_filename(prefix: str = "acquisition") -> str:
    """Build a timestamped file name (without extension).

    Args:
        prefix: Base name prepended to the timestamp.

    Returns:
        str: A file name such as ``acquisition_20260821_153000``.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def save_frames(frames: np.ndarray,
                filename: str | None = None,
                directory: str | Path = "data",
                prefix: str = "acquisition") -> Path:
    """Persist acquired frames to a ``.npy`` file.

    Args:
        frames: Array of shape ``(nlines, npixels)`` to save.
        filename: Base file name without extension. A timestamped name is
            used when None.
        directory: Output directory, created if it does not exist.
        prefix: Name prefix used when ``filename`` is None.

    Returns:
        Path: The path of the written ``.npy`` file.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    name = filename or default_filename(prefix)
    filepath = directory / f"{name}.npy"
    np.save(filepath, frames)
    return filepath
