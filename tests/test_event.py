import pandas as pd
import xarray as xr

import numpy.testing as npt
import xarray.testing as xrt

from neuroglia.event import EventTraceTensorizer, EventSpikeTensorizer

import numpy as np

# create fake event data
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

# create fake calcium data
TIME = np.arange(0, 100, 1/30.)
NEURON = [0, 1, 2]
data = np.random.randn(len(TIME), 3)
DFF = pd.DataFrame(data, TIME, NEURON)

# create fake spike data
SPIKES = pd.DataFrame({'neuron':[0,0,1],'time':[0.01,0.2,0.83]})

# create bins attribute
BINS  = np.arange(0,1,0.01)


def test_EventTraceTensorizer_dims():
    tensorizer = EventTraceTensorizer(DFF,bins=BINS)
    tensor = tensorizer.fit_transform(EVENTS)
    print(tensor)

    npt.assert_equal(tensor['neuron'].data,NEURON)
    npt.assert_equal(tensor['time_from_event'].data,BINS[:-1])
    npt.assert_equal(tensor['lbl'].data,LBL)

def test_EventSpikeTensorizer():
    tensorizer = EventSpikeTensorizer(SPIKES,bins=BINS)
    tensor = tensorizer.fit_transform(EVENTS)
    print(tensor)

    npt.assert_equal(tensor['neuron'].data,SPIKES['neuron'].unique())
    npt.assert_equal(tensor['time_from_event'].data,BINS[:-1])
    npt.assert_equal(tensor['lbl'].data,LBL)
