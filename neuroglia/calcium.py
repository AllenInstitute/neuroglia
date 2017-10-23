import numpy as np
from sklearn.base import TransformerMixin
from oasis.functions import deconvolve
from scipy.signal import medfilt, savgol_filter

class MedianFilterDetrend(TransformerMixin):
    """
    Median filter detrending
    """
    def __init__(self,
        window=101,
        peak_std_threshold=4)

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
            mf = medfilt(X[col].values.astype(np.double), self.window)
            mf = np.minimum(mf, self.peak_std_threshold * self.robust_std(mf))
            X_new[col] = X[col].values.astype(np.double) - mf

        return X_new


class OASISInferer(TransformerMixin):
    """docstring for OASISInferer."""
    def __init__(self,
        output='spikes',
        g=(None,),
        sn=None,
        b=None,
        b_nonneg=True,
        optimize_g=0,
        penalty=0,
        **kwargs,
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
                **self.kwargs,
                )
            self.fit_params[col] = dict(b=b,g=g,lam=lam,)

            if self.output=='denoised':
                X_new[col] = c
            elif self.output=='spikes':
                X_new[col] = s
            else:
                raise NotImplementedError

        return X_new
