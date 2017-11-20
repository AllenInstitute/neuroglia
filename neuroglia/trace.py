import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import binarize

class Binarizer(BaseEstimator, TransformerMixin):
    """Binarize data (set feature values to 0 or 1) according to a threshold

    This transformer is a DataFram-friendly alternative to
    sklearn.preprocessing.Binarizer

    Values greater than the threshold map to 1, while values less than
    or equal to the threshold map to 0. With the default threshold of 0,
    only positive values map to 1.

    Parameters
    ----------
    threshold : float, optional (0.0 by default)
        Feature values below or equal to this are replaced by 0, above it by 1.
        Threshold may not be less than 0 for operations on sparse matrices.
    copy : boolean, optional, default True
        set to False to perform inplace binarization and avoid a copy (if
        the input is already a numpy array or a scipy.sparse CSR matrix).

    Notes
    -----

    If the input is a sparse matrix, only the non-zero values are subject
    to update by the Binarizer class.

    This estimator is stateless (besides constructor parameters), the
    fit method does nothing but is useful when used in a pipeline.
    """

    def __init__(self, threshold=0.0, copy=True):
        self.threshold = threshold
        self.copy = copy

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

    def transform(self, X):
        """Binarize each element of X

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The data to binarize, element by element.
        """
        df = True
        try:
            index = X.index
            columns = X.columns
        except AttributeError:
            df = False

        X_ = binarize(X, threshold=self.threshold, copy=self.copy)

        if df:
            return pd.DataFrame(data=X_,index=index,columns=columns)
        else:
            return X_



def edge_detector(X,falling=False):

    df = True
    try:
        index = X.index
        columns = X.columns
    except AttributeError:
        df = False

    X = np.apply_along_axis(
        func1d=np.diff,
        axis=0,
        arr=X.copy(),
    )
    empty_row = np.zeros(shape=(1,X.shape[1]),dtype=X.dtype)
    X = np.vstack((empty_row,X))

    if falling:
        X = X < 0
    else:
        X = X > 0

    X = X.astype(int)

    if df:
        return pd.DataFrame(data=X,index=index,columns=columns)
    else:
        return X

class EdgeDetector(BaseEstimator,TransformerMixin):
    """docstring for EdgeDetector."""
    def __init__(self, falling=False):
        self.falling = falling

    def fit(self,X,y=None):
        return self

    def transform(self,X):

        return edge_detector(X,self.falling)

class WhenTrueFinder(BaseEstimator,TransformerMixin):
    """docstring for WhenTrueFinder."""
    def __init__(self):
        pass

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        return (X[X > 0]
            .stack()
            .reset_index()[['level_0','level_1']]
            .rename(columns={'level_0':'time','level_1':'neuron'})
        )
