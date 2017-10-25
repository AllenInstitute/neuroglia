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

        super(MedianFilterDetrend, self).__init__(
            window = window,
            peak_std_threshold = peak_std_threshold)

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

        super(SavGolFilterDetrend, self).__init__(
            window = window,
            order = order)

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

        super(EventRescale, self).__init__(
            log_transform = log_transform,
            scale = scale)

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
        super(OASISInferer, self).__init__(
            output = output,
            g = g,
            sn = sn,
            b = b,
            b_nonneg = b_nonneg,
            optimize_g = optimize_g,
            penalty = penalty)

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
                penalty = self.penalty
                )
            self.fit_params[col] = dict(b=b,g=g,lam=lam,)

            if self.output=='denoised':
                X_new[col] = c
            elif self.output=='spikes':
                X_new[col] = np.maximum(0, s)
            else:
                raise NotImplementedError

        return X_new
