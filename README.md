# gfa-lib

Hardware-control library for a Fourier-domain OCT (optical coherence
tomography) laboratory setup. It wraps the spectrometers and motion
controllers used to acquire spectral interferograms and to scan the sample
stage, plus the signal-processing core that converts a spectrum into a
depth (optical path difference, OPD) profile.

## Layout

Each concrete class implements the matching `interfaces` ABC, so the
acquisition code is independent of the specific hardware. Methods shared
across drivers copy the interface docstring at runtime via the `docstring`
decorator in `interfaces/__init__.py`, keeping documentation in one place.

## Dependencies

`numpy`, `scipy`, `pylablib-lightweight` (IMAQ camera interface),
`seabreeze` (HR4000) and `serial` (ESP301). Pinned in `pyproject.toml`;
install with `uv sync`.

## Lint

Docstrings follow the Google style, enforced by ruff (`D` rules with
`pydocstyle.convention = "google"`). Run:

```bash
uv run ruff check .
```

## Status

Spectrometer and motion-controller drivers are implemented; `Scan` provides
the OCT reconstruction (`zoom_fft`). Hardware-dependent paths require the
physical devices to be connected.
