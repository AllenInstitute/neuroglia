import pandas as pd
import xarray as xr
import numpy as np

import numpy.testing as npt
import xarray.testing as xrt

from neuroglia.tensor import ResponseReducer

LBL = ['a','b','a','b']
NRN = ['roi_1','roi_2']
TIME = [0.1,0.2,0.3]

TENSOR = xr.DataArray(
    np.array([
        [[0,1,2],[3,4,5]],
        [[6,7,8],[9,0,1]],
        [[0,1,2],[3,4,5]],
        [[6,7,8],[9,0,1]]
    ]),
    dims=('event','neuron','sample_times'),
    coords={
        'event': LBL,
        'neuron': NRN,
        'sample_times': TIME,
    },
)

def test_ResponseReducer_smoke():
    extractor = ResponseReducer()
    responses = extractor.fit_transform(TENSOR)

    npt.assert_array_equal(responses['event'],LBL)
    npt.assert_array_equal(responses['neuron'],NRN)
