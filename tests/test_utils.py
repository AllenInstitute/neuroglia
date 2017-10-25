from neuroglia.utils import events_to_xr_dim, create_bin_array

import pandas as pd
import xarray as xr
import xarray.testing as xrt

import numpy as np
import numpy.testing as npt

TIME = [0.1, 0.2, 0.5]
LBL = ['a', 'b', 'b']

EVENTS = pd.DataFrame(dict(time=TIME, lbl=LBL))

CONCAT_DIM = xr.DataArray(
    range(3),
    name='event',
    dims=['event'],
    coords={
        # 'index': ('event',range(3)),
        'time': ('event', TIME),
        'lbl': ('event',LBL),
    }
)

def test_events_to_xr_dim():
    # from xarray.testing import assert_allclose
    concat_dim = events_to_xr_dim(EVENTS)
    xrt.assert_allclose(concat_dim,CONCAT_DIM)

def test_create_bin_array():
    bins = np.linspace(0,1,10)
    bin_arr = create_bin_array(bins,None)
    npt.assert_array_equal(bin_arr,bins)

def test_create_bin_array():
    bins, window = 10, (0,1)
    bin_arr = create_bin_array(bins,window)
    npt.assert_array_equal(bin_arr,np.linspace(window[0],window[1],bins+1))

if __name__ == '__main__':
    test_events_to_xr_dim()
