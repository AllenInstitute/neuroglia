import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import binarize

class Binarizer(BaseEstimator, TransformerMixin):
    """docstring for scikit learn Binarizer
    """

    def __init__(self, threshold=0.0, copy=True):
        self.threshold = threshold
        self.copy = copy

    def fit(self, X, y=None):
        return self

    def transform(self, X):
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
