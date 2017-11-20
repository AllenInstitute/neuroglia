import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class ResponseReducer(BaseEstimator,TransformerMixin):
    """Reduces a response tensor by performing a function along one dimension


    Notes
    -----

    This estimator is stateless (besides constructor parameters), the
    fit method does nothing but is useful when used in a pipeline.
    """
    def __init__(self, func, dim='sample_times'):
        self.func = func
        self.dim = dim

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged

        This method is here to implement the scikit-learn API and work in
        scikit-learn pipelines.

        Parameters
        ----------
        X : xarray.DataArray in `response tensor` structre ['events','sample_times','neurons']

        Returns
        -------
        self

        """
        return self

    def transform(self, X):
        """Reduces a response tensor by performing a function along one dimension

        Parameters
        ----------
        X : xarray.DataArray in `response tensor` structre ['events','sample_times','neurons']

        Returns
        -------
        Xt : xarray.DataArray with remaining dimensions
        """
        return X.reduce(self.func,dim=self.dim)
