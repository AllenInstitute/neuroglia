import numpy as np
from sklearn.base import TransformerMixin

from .trace import TraceTensorizer

class MeanResponseExtractor(TransformerMixin):
    """docstring for Annotator."""
    def __init__(self, arg):
        super(MeanResponseExtractor, self).__init__(traces,bins,range)
        self.traces = traces
        self.bins = bins
        self.range = range

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        tensorizer = TraceTensorizer(events=X,bins=self.bins,range=self.range))
        tensor = tensorizer.fit_transform(self.traces)
        return tensor.mean(dim='time').to_dataframe()
