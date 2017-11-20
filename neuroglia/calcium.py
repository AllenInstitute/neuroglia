import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from oasis.functions import deconvolve
from scipy.signal import medfilt, savgol_filter


class MedianFilterDetrend(BaseEstimator, TransformerMixin):
    """
    Median filter detrending
    """
    def __init__(self,
        window=101,
        peak_std_threshold=4):

        self.window = window
        self.peak_std_threshold = peak_std_threshold

    def robust_std(self, x):
        '''
        Robust estimate of std
        '''
        MAD = np.median(np.abs(x - np.median(x)))
        return 1.4826*MAD

    def fit(self, X, y=None):
        self.fit_params = {}
        return self

    def transform(self,X):
        X_new = X.copy()
        for col in X.columns:
            tmp_data = X[col].values.astype(np.double)
            mf = medfilt(tmp_data, self.window)
            mf = np.minimum(mf, self.peak_std_threshold * self.robust_std(mf))
            self.fit_params[col] = dict(mf=mf)
            X_new[col] = tmp_data - mf

        return X_new


class SavGolFilterDetrend(BaseEstimator, TransformerMixin):
    """
    Savitzky-Golay filter detrending
    """
    def __init__(self,
        window=201,
        order=3):

        self.window = window
        self.order = order

    def fit(self, X, y=None):
        self.fit_params = {}
        return self

    def transform(self,X):
        X_new = X.copy()
        for col in X.columns:
            tmp_data = X[col].values.astype(np.double)
            sgf = savgol_filter(tmp_data, self.window, self.order)
            self.fit_params[col] = dict(sgf=sgf)
            X_new[col] = tmp_data - sgf

        return X_new


class EventRescale(BaseEstimator, TransformerMixin):
    """
    Savitzky-Golay filter detrending
    """
    def __init__(self,
        log_transform=True,
        scale=5):

        self.log_transform = log_transform
        self.scale = scale

    def fit(self, X, y=None):
        self.fit_params = {}
        return self

    def transform(self,X):
        X_new = X.copy()
        for col in X.columns:
            tmp_data = X[col].values.astype(np.double)
            tmp_data *= self.scale
            if self.log_transform:
                tmp_data = np.log(1 + tmp_data)
            X_new[col] = tmp_data

        return X_new


class OASISInferer(BaseEstimator, TransformerMixin):
    """docstring for OASISInferer."""
    def __init__(self,
        output='spikes',
        g=(None,),
        sn=None,
        b=None,
        b_nonneg=True,
        optimize_g=0,
        penalty=0,
        **kwargs
        ):
        super(OASISInferer, self).__init__()

        self.output = output
        self.g = g
        self.sn = sn
        self.b = b
        self.b_nonneg = b_nonneg
        self.optimize_g = optimize_g
        self.penalty = penalty
        self.kwargs = kwargs

    def fit(self, X, y=None):
        self.fit_params = {}
        return self

    def transform(self,X):

        X_new = X.copy()

        for col in X.columns:
            c, s, b, g, lam = deconvolve(
                X[col].values.astype(np.double),
                g = self.g,
                sn = self.sn,
                b = self.b,
                b_nonneg = self.b_nonneg,
                optimize_g = self.optimize_g,
                penalty = self.penalty,
                **self.kwargs
                )
            self.fit_params[col] = dict(b=b,g=g,lam=lam,)

            if self.output=='denoised':
                X_new[col] = c
            elif self.output=='spikes':
                X_new[col] = np.maximum(0, s)
            else:
                raise NotImplementedError

        return X_new


def normalize_trace(trace, window=3, percentile=8):
    """ normalized the trace by substracting off a rolling baseline


    Parameters
    ---------
    trace: pd.Series with time as index
    window: float
        time in minutes
    percentile: int
        percentile to subtract off
    """

    sampling_rate = np.diff(trace.index).mean()
    window = int(np.ceil(window/sampling_rate))

    p = lambda x: np.percentile(x,percentile) # suggest 8% in literature, but this doesnt work well for our data, use median
    baseline = trace.rolling(window=window,center=True).apply(func=p)
    baseline = baseline.fillna(method='bfill')
    baseline = baseline.fillna(method='ffill')
    dF = trace - baseline
    dFF = dF / baseline

    return dFF


class Normalize(BaseEstimator,TransformerMixin):
    """docstring for Normalize."""
    def __init__(self, window=3.0, percentile=8):
        super(Normalize, self).__init__()
        self.window = window
        self.percentile = percentile

    def fit(self, X, y=None):
        return self

    def transform(self,X):
        # this is where the magic happens

        df_norm = pd.DataFrame()
        for col in X.columns:
            df_norm[col] = normalize_trace(
                trace=X[col],
                window=self.window,
                percentile=self.percentile,
                )

        return df_norm
