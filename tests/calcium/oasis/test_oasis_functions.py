import pytest
import numpy as np
import numpy.testing as npt

from neuroglia.calcium.oasis import functions
from neuroglia.datasets.synthetic_calcium import gen_data


def calculate_corrcoef(truth, estimate):
    return np.corrcoef(truth, estimate)[0, 1]


@pytest.fixture
def synthetic_calcium(calcium_kwargs):
    return map(np.squeeze, gen_data(**calcium_kwargs))


@pytest.mark.slow
@pytest.mark.unit
@pytest.mark.parametrize("calcium_kwargs, deconvolve_kwargs, calcium_threshold, spike_threshold", [
    (
        {"N": 1, "b": 2, "seed": 0, },
        {"penalty": 1, },
        0.9,
        0.8,
    ),
    (
        {"g": (1.7, -0.712, ), "sn": 0.5, "N": 1, "b": 2, "seed": 0, },
        {"g": (1.7, -0.712, ), },
        0.9,
        0.7,
    ),
    (
        {"g": (1.7, -0.712, ), "sn": 0.5, "N": 1, "b": 2, "seed": 0, },
        {"g": (None, None, ), "penalty": 0, "optimize_g": 5, "max_iter": 5, },
        0.9,
        0.7,
    ),
    (
        {"sn": 0.5, "N": 1, "b": 2, "seed": 0, },
        {"g": (None, None, ), "penalty": 0, "optimize_g": 2, "max_iter": 2, "b": 1, },
        0.7,
        0.2,
    ),
])
def test_deconvolve(synthetic_calcium, calcium_kwargs, deconvolve_kwargs, calcium_threshold, spike_threshold):
    y, true_c, true_s = synthetic_calcium
    c, s, b, g, lam = functions.deconvolve(y, **deconvolve_kwargs)

    assert calculate_corrcoef(true_c, c) > calcium_threshold and \
        calculate_corrcoef(true_s, s) > spike_threshold


@pytest.mark.unit
@pytest.mark.parametrize("calcium_kwargs, estimate_kwargs, expected", [
    (
        {"N": 1, "b": 2, "seed": 0, },
        {"p": 2, "sn": 0.5, "nonlinear_fit": True, },
        np.array([0.06883, 0.94927]),
    ),
])
def test_estimate_time_constant(synthetic_calcium, calcium_kwargs, estimate_kwargs, expected):
    y, true_c, true_s = synthetic_calcium

    npt.assert_allclose(
        functions.estimate_time_constant(y, **estimate_kwargs),
        expected,
        atol=1e-4
    )


@pytest.mark.unit
@pytest.mark.parametrize("calcium_kwargs, onnls_kwargs, calcium_threshold, spike_threshold", [
    (
        {"g": (0.95, ), "N": 1, "b": 2, "seed": 0, },
        {"g": (0.95, ), },
        0.9,
        0.5,
    ),
    (
        {"g": (1.7, -0.712, ), "N": 1, "b": 2, "seed": 0, },
        {"g": (1.7, -0.712, ), },
        0.9,
        0.7,
    ),
])
def test_onnls(synthetic_calcium, calcium_kwargs, onnls_kwargs, calcium_threshold, spike_threshold):
    y, true_c, true_s = synthetic_calcium

    c, s = functions.onnls(y, **onnls_kwargs)

    assert calculate_corrcoef(true_c, c) > calcium_threshold and \
        calculate_corrcoef(true_s, s) > spike_threshold


@pytest.mark.unit
@pytest.mark.parametrize("calcium_kwargs, g, sn, b, optimize_g", [
    ({"g": [1.7, -0.712, ], "sn": 0.2, "N": 1, "seed": 3, }, [1.7, -0.712, ], 0.2, 0, 0, ),
    ({"g": [1.7, -0.712, ], "sn": 0.2, "N": 1, "seed": 3, }, [1.7, -0.712, ], 0.2, None, 0, ),
    ({"g": [1.7, -0.712, ], "sn": 0.2, "N": 1, "seed": 3, }, [1.7, -0.712, ], 0.2, 0, 10, ),
    ({"g": [1.7, -0.712, ], "sn": 0.2, "N": 1, "seed": 3, }, [1.7, -0.712, ], 0.2, None, 10, ),
])
def test_jfriedrich_deconvolve(synthetic_calcium, calcium_kwargs, g, sn, b, optimize_g):
    y, true_c, true_s = synthetic_calcium

    c, s, b, g, lam = functions.deconvolve(y, g, sn, b=b, optimize_g=optimize_g)

    npt.assert_allclose(calculate_corrcoef(c, true_c), 1, 0.03)
    npt.assert_allclose(calculate_corrcoef(s, true_s), 1, 0.2)


@pytest.mark.unit
@pytest.mark.parametrize("calcium_kwargs, g, lam", [
    ({"g": [0.95, ], "sn": 0.3, "N": 1, "seed": 0, }, [0.95, ], 0),
    ({"g": [0.95, ], "sn": 0.3, "N": 1, "seed": 0, }, [0.95, ], 2.4),
    ({"g": [0.95, ], "sn": 0.3, "N": 1, "seed": 0, }, 0.95 ** np.arange(1000), 0),
])
def test_jfriedrich_onnls(synthetic_calcium, calcium_kwargs, g, lam):
    y, true_c, true_s = synthetic_calcium

    c, s = functions.onnls(y, g, lam=lam)

    npt.assert_allclose(calculate_corrcoef(c, true_c), 1, 0.03)
    npt.assert_allclose(calculate_corrcoef(s, true_s), 1, 0.2)


@pytest.mark.unit
@pytest.mark.parametrize("calcium_kwargs, sn, g, nonlinear_fit", [
    ({"g": [0.95, ], "sn": 0.3, "N": 1, "seed": 0, }, 0.3, [0.95, ], True, ),
    ({"g": [1.7, -0.712, ], "sn": 0.3, "N": 1, "seed": 0, }, 0.3, [1.7, -0.712, ], True, ),
    ({"g": [0.95, ], "sn": 0.3, "N": 1, "seed": 0, }, 0.3, [0.95, ], False, ),
    ({"g": [1.7, -0.712, ], "sn": 0.3, "N": 1, "seed": 0, }, 0.3, [1.7, -0.712, ], False, ),
])
def test_jfriedrich_estimate_parameters(synthetic_calcium, calcium_kwargs, sn, g, nonlinear_fit):
    y, true_c, true_s = synthetic_calcium

    estimated_g, estimated_sn = functions.estimate_parameters(
        y, len(g), nonlinear_fit=nonlinear_fit
    )

    npt.assert_allclose(estimated_g, g, 0.02)
    npt.assert_allclose(estimated_sn, sn, 0.02)
