from neuroglia.core import BaseTensorizer

import numpy as np
import pandas as pd
import xarray as xr

import numpy.testing as npt
import xarray.testing as xrt


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

def test_BaseTensorizer_wo_range():
    bins = [0.1,0.2,0.3]
    tensorizer = BaseTensorizer(EVENTS,bins=bins)
    npt.assert_allclose(tensorizer.bins,bins)
    
def test_BaseTensorizer_w_range():
    # from numpy.testing import assert_allclose
    tensorizer = BaseTensorizer(EVENTS,bins=2,range=(0.1,0.3))
    npt.assert_allclose(tensorizer.bins,[0.1,0.2,0.3])

def test_BaseTensorizer_concat_dim():
    # from xarray.testing import assert_allclose
    tensorizer = BaseTensorizer(EVENTS,bins=[0.1,0.5,1.0])
    xrt.assert_allclose(tensorizer.concat_dim,CONCAT_DIM)
    
if __name__ == '__main__':
    test_BaseTensorizer()
