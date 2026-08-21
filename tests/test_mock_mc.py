"""Tests for the MockMC simulated motion controller."""

import pytest


def test_open_detects_axes(mock_mc):
    """open() detects the three default axes."""
    assert mock_mc.available_axes == {1, 2, 3}


def test_move_and_read_position(mock_mc):
    """move_absolute updates the reported position."""
    mock_mc.move_absolute(1, 2.5)
    assert mock_mc.get_position(1) == pytest.approx(2.5)


def test_goto_and_wait_arrives(mock_mc):
    """goto_and_wait blocks until the target position is reached."""
    mock_mc.goto_and_wait(2, 1.0, timeout_s=0.5)
    assert mock_mc.get_position(2) == pytest.approx(1.0, abs=1e-3)


def test_stop_request_interrupts_goto(mock_mc):
    """A stop request makes goto_and_wait raise RuntimeError."""
    # Request a stop before the move completes.
    mock_mc.stop_motion(3)
    with pytest.raises(RuntimeError):
        mock_mc.goto_and_wait(3, 5.0, timeout_s=0.5)
