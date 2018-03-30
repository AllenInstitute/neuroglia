import numpy as np


def gen_data(g=[.95], sn=.3, T=3000, framerate=30, firerate=.5, b=0, N=20, seed=13):
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

    np.random.seed(seed)
    Y = np.zeros((N, T))
    trueSpikes = np.random.rand(N, T) < firerate / float(framerate)
    truth = trueSpikes.astype(float)
    for i in range(2, T):
        if len(g) == 2:
            truth[:, i] += g[0] * truth[:, i - 1] + g[1] * truth[:, i - 2]
        else:
            truth[:, i] += g[0] * truth[:, i - 1]
    Y = b + truth + sn * np.random.randn(N, T)
    return Y, truth, trueSpikes


def gen_sinusoidal_data(g=[.95], sn=.3, T=3000, framerate=30, firerate=.5, b=0, N=20, seed=13):
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

    np.random.seed(seed)
    Y = np.zeros((N, T))
    trueSpikes = np.random.rand(N, T) < firerate / float(framerate) * \
        np.sin(np.arange(T) // 50)**3 * 4
    truth = trueSpikes.astype(float)
    for i in range(2, T):
        if len(g) == 2:
            truth[:, i] += g[0] * truth[:, i - 1] + g[1] * truth[:, i - 2]
        else:
            truth[:, i] += g[0] * truth[:, i - 1]
    Y = b + truth + sn * np.random.randn(N, T)
    return Y, truth, trueSpikes
