import pandas as pd
import xarray as xr
import numpy as np

import numpy.testing as npt
import xarray.testing as xrt

from neuroglia.spike import Smoother
from neuroglia.event import PeriEventTraceSampler, PeriEventSpikeSampler

from sklearn.base import clone


# create fake event data
TIME = [0.1, 0.2, 0.5]
LBL = ['a', 'b', 'b']
EVENTS = pd.DataFrame(dict(time=TIME, lbl=LBL))
CONCAT_DIM = xr.DataArray(
    range(3),
    name='event',
    dims=['event'],
    coords={
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
TS  = np.arange(0,1,0.01)


def test_PeriEventTraceSampler_dims():
    tensorizer = PeriEventTraceSampler(DFF,sample_times=TS)
    tensor = tensorizer.fit_transform(EVENTS)

    npt.assert_equal(tensor['neuron'].data,NEURON)
    npt.assert_equal(tensor['sample_times'].data,TS)
    npt.assert_equal(tensor['lbl'].data,LBL)

    clone(tensorizer)

def test_PeriEventTraceReducer_dims():
    tensorizer = PeriEventTraceReducer(DFF,sample_times=TS,func=np.mean)
    tensor = tensorizer.fit_transform(EVENTS)

    npt.assert_equal(tensor['neuron'].data,NEURON)
    npt.assert_equal(tensor['sample_times'].data,TS[:-1])
    npt.assert_equal(tensor['lbl'].data,LBL)

    clone(tensorizer)

def test_PeriEventSpikeSampler():
    tensorizer = PeriEventSpikeSampler(SPIKES,sample_times=TS,sampler=Smoother)
    tensor = tensorizer.fit_transform(EVENTS)

    npt.assert_equal(tensor['neuron'].data,SPIKES['neuron'].unique())
    npt.assert_equal(tensor['sample_times'].data,TS)
    npt.assert_equal(tensor['lbl'].data,LBL)

    clone(tensorizer)


def test_PeriEventSpikeSampler_no_response():

    spikes = pd.DataFrame({'neuron':[0,0,1],'time':[0.01,0.2,1.6]})

    tensorizer = PeriEventSpikeSampler(spikes,sample_times=TS)
    tensor = tensorizer.fit_transform(EVENTS)
