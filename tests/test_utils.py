from neuroglia.utils import events_to_xr_dim

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


if __name__ == '__main__':
    test_events_to_xr_dim()
