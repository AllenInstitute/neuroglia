import numpy as np
import pandas as pd


def gen_random_spikes(N, T, firerate, framerate, seed=None):

    if seed is not None:
        np.random.seed(seed)

    true_spikes = np.random.rand(N, T) < firerate / float(framerate)
    return true_spikes


def gen_sinusoidal_spikes(N, T, firerate, framerate, seed=None):

    if seed is not None:
        np.random.seed(seed)

    true_spikes = np.random.rand(N, T) < firerate / float(framerate) * \
        np.sin(np.arange(T) // 50)**3 * 4

    return true_spikes

def make_calcium(true_spikes, g):

    truth = true_spikes.astype(float)

    for i in range(2, truth.shape[1]):
        if len(g) == 2:
            truth[:, i] += g[0] * truth[:, i - 1] + g[1] * truth[:, i - 2]
        else:
            truth[:, i] += g[0] * truth[:, i - 1]

    return truth


def add_noise(truth, b, sn):

    noise = sn * np.random.randn(*truth.shape)

    return b + truth + noise


def gen_data(g=[.95], sn=.3, T=3000, framerate=30, firerate=.5, b=0, N=20, seed=None):
    """
    Generate data from homogenous Poisson Process

    Parameters
    ----------
    g : array, shape (p,), optional, default=[.95]
        Parameter(s) of the AR(p) process that models the fluorescence impulse response.
    sn : float, optional, default .3
        Noise standard deviation.
    T : int, optional, default 3000
        Duration.
    framerate : int, optional, default 30
        Frame rate.
    firerate : int, optional, default .5
        Neural firing rate.
    b : int, optional, default 0
        Baseline.
    N : int, optional, default 20
        Number of generated traces.
    seed : int, optional, default 13
        Seed of random number generator.

    Returns
    -------
    y : array, shape (N, T)
        Noisy fluorescence data.
    c : array, shape (N, T)
        Calcium traces (without sn).
    s : array, shape (N, T)
        Spike trains.
    """

    if seed is not None:
        np.random.seed(seed)

    true_spikes = gen_random_spikes(
        N=N,
        T=T,
        firerate=firerate,
        framerate=framerate,
    )
    true_calcium = make_calcium(true_spikes, g)
    observed = add_noise(true_calcium, b, sn)

    return observed, true_calcium, true_spikes


def gen_sinusoidal_data(
    g=(.95,),
    sn=.3,
    T=3000,
    framerate=30,
    firerate=.5,
    b=0,
    N=20,
    seed=None,
):
    """
    Generate data from inhomogenous Poisson Process with sinusoidal instantaneous activity

    Parameters
    ----------
    g : array, shape (p,), optional, default=[.95]
        Parameter(s) of the AR(p) process that models the fluorescence impulse response.
    sn : float, optional, default .3
        Noise standard deviation.
    T : int, optional, default 3000
        Duration.
    framerate : int, optional, default 30
        Frame rate.
    firerate : float, optional, default .5
        Neural firing rate.
    b : float, optional, default 0
        Baseline.
    N : int, optional, default 20
        Number of generated traces.
    seed : int, optional, default 13
        Seed of random number generator.

    Returns
    -------
    y : array, shape (N, T)
        Noisy fluorescence data.
    c : array, shape (N, T)
        Calcium traces (without sn).
    s : array, shape (N, T)
        Spike trains.
    """

    if seed is not None:
        np.random.seed(seed)

    true_spikes = gen_sinusoidal_spikes(
        N=N,
        T=T,
        firerate=firerate,
        framerate=framerate,
    )
    true_calcium = make_calcium(true_spikes, g)
    observed = add_noise(true_calcium, b, sn)

    return observed, true_calcium, true_spikes


def make_calcium_traces(
    n_neurons=10,
    duration=60.0,
    sampling_rate=30.0,
    oscillation=True,
):

    neuron_ids = ['neuron_{}'.format(n) for n in range(n_neurons)]

    gen_params = dict(
        g=[.95],
        sn=.3,
        T=int(sampling_rate*duration),
        framerate=sampling_rate,
        firerate=.5,
        b=0,
        N=n_neurons,
        seed=13,
    )

    if oscillation:
        make_traces = gen_sinusoidal_data
    else:
        make_traces = gen_data

    traces, _, spikes = map(np.squeeze, make_traces(**gen_params))

    time = np.arange(0, traces.shape[1]/sampling_rate, 1/sampling_rate)

    traces = pd.DataFrame(traces.T, index=time, columns=neuron_ids)
    spikes = pd.DataFrame(spikes.T, index=time, columns=neuron_ids)

    return traces, spikes
