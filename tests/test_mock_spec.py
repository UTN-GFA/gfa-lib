"""Tests for the MockSpec simulated spectrometer."""



def test_read_shape_and_type(mock_spec):
    """read() returns wavelength/intensity arrays of the right shape."""
    wl, irr = mock_spec.read()
    assert wl.shape == (mock_spec.npixels,)
    assert irr.shape == (mock_spec.npixels,)
    assert irr.dtype.kind == "f"


def test_acquire_returns_frames(mock_spec):
    """acquire() returns a 2D frame stack with at least one line."""
    frames = mock_spec.acquire(acquisition_time=2000, save_data=False)
    assert frames.ndim == 2
    assert frames.shape[1] == mock_spec.npixels
    assert frames.shape[0] >= 1


def test_acquire_requires_connection():
    """acquire() raises RuntimeError when called before open()."""
    from gfalib.spectrometer.MockSpec import MockSpec

    spec = MockSpec()
    try:
        spec.acquire(acquisition_time=1000)
        raise AssertionError("acquire should fail before open()")
    except RuntimeError:
        pass
