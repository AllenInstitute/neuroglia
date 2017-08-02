import numpy as np
from neuroglia.core import BaseTensorizer

def align(trace_series,time_bins):
    t = trace_series.index

    return


class TraceTensorizer(BaseTensorizer):
    """docstring for SpikeTensorizer."""
    def __init__(self, events, window, alignment=None):
        super(TraceTensorizer, self).__init__(events,window)

    def fit(self, X, y=None):
        tres = X.index[1]-X.index[0]
        nbins = int(np.round((self.window[1]-self.window[0])/tres))
        self.target_shape = len(self.events), nbins, len(X.columns),
        dtypes = X.dtypes.unique()
        assert len(dtypes)==1
        self.target_dtype = dtypes[0]
        # 
        # if alignment is None:
        #     if self.target_dtype == np.int_:
        #         self.alignment = 'nearest'
        #     else:
        #         self.alignment = 'cubic_spline'
        # else:
        #     self.alignment = alignment

        return self

    def transform(self, X):
        tensor = np.empty(self.target_shape,self.target_dtype)
        for ii, (_,ev) in enumerate(self.events.iterrows()):
            # assert X.index.contains(ev['time'])

            assert np.isclose(X.index, ev['time'], atol=0.001).any() # need to determine approach if events are not aligned... interpolate? align events to nearest bin?
            mask = (
                (X.index >= ev['time']+self.window[0]) # replace with searchsorted
                & (X.index < ev['time']+self.window[1])
                )
            tensor[ii,:,:] = X[mask].values
        return tensor
