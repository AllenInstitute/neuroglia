import pandas as pd
import xarray as xr
import numpy as np

import numpy.testing as npt
import xarray.testing as xrt

from neuroglia.tensor import ResponseExtractor

TENSOR = xr.DataArray(
    np.array([
        [[[0,1,2],[3,4,5]],[[6,7,8],[9,0,1]]],
        [[[0,1,2],[3,4,5]],[[6,7,8],[9,0,1]]]
    ]),
    dims=('event','neuron','time_from_event'),
    coords={
        'event':['a','b'],
        'neuron': ['roi_1','roi_2'],
        'time': [0.1,0.2,0.3]
    },
)

def test_ResponseExtractor_smoke():
    extractor = ResponseExtractor()
    responses = extractor.fit_transform(TENSOR)
