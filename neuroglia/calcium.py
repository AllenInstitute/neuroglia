import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from oasis.functions import deconvolve
from scipy.signal import medfilt, savgol_filter


class MedianFilterDetrender(BaseEstimator, TransformerMixin):
    """
    Median filter detrending
    """
    def __init__(self,
        window=101,
        peak_std_threshold=4):

        self.window = window
        self.peak_std_threshold = peak_std_threshold

    def _robust_std(self, x):
        '''
        Robust estimate of std
        '''
        MAD = np.median(np.abs(x - np.median(x)))
        return 1.4826*MAD

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged

        This method is here to implement the scikit-learn API and work in
        scikit-learn pipelines.

        Parameters
        ----------
        X : array-like

        Returns
        -------
        self

        """
        return self

    def transform(self,X):
        self.fit_params = {}
        X_new = X.copy()
        for col in X.columns:
            tmp_data = X[col].values.astype(np.double)
            mf = medfilt(tmp_data, self.window)
            mf = np.minimum(mf, self.peak_std_threshold * self._robust_std(mf))
            self.fit_params[col] = dict(mf=mf)
            X_new[col] = tmp_data - mf

        return X_new


class SavGolFilterDetrender(BaseEstimator, TransformerMixin):
    """
    Savitzky-Golay filter detrending
    """
    def __init__(self,
        window=201,
        order=3):

        self.window = window
        self.order = order

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged

        This method is here to implement the scikit-learn API and work in
        scikit-learn pipelines.

        Parameters
        ----------
        X : array-like

        Returns
        -------
        self

        """
        return self

    def transform(self,X):
        self.fit_params = {}
        X_new = X.copy()
        for col in X.columns:
            tmp_data = X[col].values.astype(np.double)
            sgf = savgol_filter(tmp_data, self.window, self.order)
            self.fit_params[col] = dict(sgf=sgf)
            X_new[col] = tmp_data - sgf

        return X_new


class EventRescaler(BaseEstimator, TransformerMixin):
    """
    rescale events
    """
    def __init__(self,log_transform=True,scale=5):

        self.log_transform = log_transform
        self.scale = scale

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged

        This method is here to implement the scikit-learn API and work in
        scikit-learn pipelines.

        Parameters
        ----------
        X : array-like

        Returns
        -------
        self

        """
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



def oasis_kwargs(penalty=None,model=None):

    kwargs = {}

    if penalty=='l0':
        kwargs.update(penalty=0)
    elif penalty=='l1':
        kwargs.update(penalty=1)

    if model.lower()=='exponential':
        kwargs.update(g=(None,))
    elif model.lower()=='double_exponential':
        kwargs.update(g=(None,None))

    return kwargs


class CalciumDeconvolver(BaseEstimator, TransformerMixin):
    """Deconvolve calcium traces to detect putative spiking events

    This transformer deconvolves each trace to yield a sparse trace where each
    bin is weighted according to the likelihood of spiking events.

    We use the OASIS algorithm from https://github.com/j-friedrich/OASIS/

    Note: you must install OASIS for `CalciumDeconvolver` to work.

    ::
        pip install cython
        pip install git+https://github.com/j-friedrich/OASIS.git

    Parameters
    ----------
    penalty : {'l0', 'l1'}
        Specify the norm used in the penalization when fitting.
    model : {'exponential','double_exponential'}
        What type of model to fit for the calcium dynamics. Typically, a fast
        calcium indicator can be fit with the single 'exponential' model,
        whereas an indicator with a slow rise will benefit from using the
        'double_exponential' model, which fits an exponential to the rise time
        of the calcium response as well.

    Notes
    -----

    This estimator is stateless (besides constructor parameters), the
    fit method does nothing but is useful when used in a pipeline.
    """
    def __init__(self,penalty='l0',model='exponential',threshold=0.001):
        self.penalty = penalty
        self.model = model
        self.threshold = threshold

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged

        This method is here to implement the scikit-learn API and work in
        scikit-learn pipelines.

        Parameters
        ----------
        X : array-like

        Returns
        -------
        self

        """
        return self

    def transform(self,X):
        """Deconvolve each column of X

        Parameters
        ----------
        X : DataFrame in `traces` structure [n_samples, n_traces]

        Returns
        -------
        Xt : DataFrame in `traces` structure [n_samples, n_traces]
            The deconvolved data events.
        """

        kwargs = oasis_kwargs(
            self.penalty,
            self.model,
            )

        X_new = X.copy()

        self.fit_params = {}
        for col in X.columns:
            denoised, spikes, b, g, lam = deconvolve(
                X[col].values.astype(np.double),
                **kwargs)
            self.fit_params[col] = dict(b=b,g=g,lam=lam,)
            X_new[col] = spikes

        return X_new

    def predict(self,X):
        """Find spikes

        Parameters
        ----------
        X : DataFrame in `traces` structure [n_samples, n_traces]

        Returns
        -------
        y : DataFrame in `traces` structure [n_samples, n_traces]
            Predicted spike events.
        """
        y = self.transform(X) > self.threshold
        return y



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
    """ Calculate rolling dF/F
    """
    def __init__(self, window=3.0, percentile=8):
        self.window = window
        self.percentile = percentile

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged

        This method is here to implement the scikit-learn API and work in
        scikit-learn pipelines.

        Parameters
        ----------
        X : array-like

        Returns
        -------
        self

        """
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
