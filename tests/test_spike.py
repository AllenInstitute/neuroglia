import pandas as pd
import xarray as xr
import numpy as np

import numpy.testing as npt
import xarray.testing as xrt

from neuroglia.spike import Smoother

# create fake spike data
SPIKES = pd.DataFrame({'neuron':[0,0,1],'time':[0.01,0.2,0.83]})

# create bins attribute
BINS  = np.arange(0,1,0.01)

def test_Smoother():
    smoother = Smoother(bins=BINS)
    smoothed = smoother.fit_transform(SPIKES)

    assert len(smoothed)==(len(BINS)-1)

def test_Smoother_noresp():
    smoother = Smoother(bins=BINS+100.0)
    smoothed = smoother.fit_transform(SPIKES)

    npt.assert_equal(
        smoothed.values,
        np.zeros((len(BINS)-1,2),np.float),
        )
