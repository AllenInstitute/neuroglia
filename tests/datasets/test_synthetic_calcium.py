import pytest
import numpy as np
import numpy.testing as npt

from neuroglia.datasets import synthetic_calcium


@pytest.mark.unit
@pytest.mark.parametrize("kwargs, expected", [
    (
        {"N": 1, "T": 10, "firerate": 0.5, "framerate": 30, "seed": 1, },
        np.array([[False, False, False, False, False, False, False, False, False, False, ]]),
    ),
])
def test_gen_sinusoidal_spikes(kwargs, expected):
    npt.assert_equal(synthetic_calcium.gen_sinusoidal_spikes(**kwargs), expected)


@pytest.mark.smoke
@pytest.mark.parametrize("kwargs, expected", [
    (
        {"N": 1, "T": 10, },
        (
            None,  # will be different each time
            np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]),
            np.array([[False, False, False, False, False, False, False, False, False, False]]),
        ),
    )
])
def test_gen_sinusoidal_data(kwargs, expected):
    sinusoidal_data = synthetic_calcium.gen_sinusoidal_data(**kwargs)

    npt.assert_equal(sinusoidal_data[1], expected[1]) and \
        npt.assert_equal(sinusoidal_data[2], expected[2])


@pytest.mark.smoke
@pytest.mark.parametrize("kwargs", [
    {"oscillation": True, },
    {"oscillation": False, },
])
def test_make_calcium_traces(kwargs):
    synthetic_calcium.make_calcium_traces(**kwargs)
