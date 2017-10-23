import pandas as pd
import xarray as xr

import xarray.testing as xrt

from neuroglia.event import EventTraceTensorizer, EventSpikeTensorizer

import numpy as np

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

TIME = np.arange(0, 100, 1/30.)
NEURON = [0, 1, 2]
data = np.random.randn(len(TIME), 3)

DFF = pd.DataFrame(data, TIME, LBL)

BINS  = np.arange(0,1,0.01)

SPIKES = pd.DataFrame({'neurons':[0,0,1],'time':[0.01,0.2,1.6]})

def test_EventTraceTensorizer_dims():
    tensor = EventTraceTensorizer(DFF,bins=BINS)

def test_EventSpikeTensorizer():
    tensor = EventSpikeTensorizer(SPIKES,bins=BINS)
