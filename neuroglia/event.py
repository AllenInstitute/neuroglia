import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin

from .trace import TraceTensorizer

class MeanResponseExtractor(BaseEstimator,TransformerMixin):
    """docstring for Annotator."""
    def __init__(self, traces,bins,range):
        super(MeanResponseExtractor, self).__init__()
        self.traces = traces
        self.bins = bins
        self.range = range

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        tensorizer = TraceTensorizer(events=X,bins=self.bins,range=self.range)
        tensor = tensorizer.fit_transform(self.traces)
        return tensor.mean(dim='time_from_event').data
