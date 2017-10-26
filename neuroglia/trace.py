import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator

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
