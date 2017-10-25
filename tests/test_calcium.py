from neuroglia.calcium import MedianFilterDetrend, SavGolFilterDetrend
from neuroglia.calcium import OASISInferer
from oasis.functions import gen_data
from sklearn.base import clone

import numpy as np
import pandas as pd
import xarray as xr

import numpy.testing as npt
import xarray.testing as xrt

# Test for proper parameter structure
def test_params():
    fn_list = [MedianFilterDetrend(), SavGolFilterDetrend(), OASISInferer()]
    for fn in fn_list:
        new_object_params = fn.get_params(deep=False)
        for name, param in new_object_params.items():
            new_object_params[name] = clone(param, safe=False)

# Test functions perform as expected
true_b = 2
y, true_c, true_s = map(np.squeeze, gen_data(N=3, b=true_b, seed=0))
y = y.T
TIME = np.arange(0, len(y)/30, 1/30.)
LBL = ['a', 'b', 'c']
sin_scale = 5

data = y + sin_scale*np.sin(.05*TIME)[:,None]
DFF = pd.DataFrame(data, TIME, LBL)

assert np.all(np.mean(DFF) > 2)

def test_MedianFilterDetrend():
    tmp = MedianFilterDetrend().fit_transform(DFF)
    assert np.all(np.isclose(np.mean(tmp), 0, atol=.1))

def test_SavGolFilterDetrend():
    tmp = SavGolFilterDetrend().fit_transform(DFF)
    assert np.all(np.isclose(np.mean(tmp), 0, atol=.1))

def test_OASISInferer():
    tmp = OASISInferer().fit_transform(SavGolFilterDetrend().fit_transform(DFF))
    assert np.all(np.array([np.corrcoef(true_s[n], np.array(tmp[a]))[0][1] for n,a in zip(range(3), LBL)]) > 0.6)
    tmp = OASISInferer().fit_transform(MedianFilterDetrend().fit_transform(DFF))
    assert np.all(np.array([np.corrcoef(true_s[n], np.array(tmp[a]))[0][1] for n,a in zip(range(3), LBL)]) > 0.6)


if __name__ == '__main__':
    test_MedianFilterDetrend()
    test_SavGolFilterDetrend()
    test_OASISInferer()
    test_params()
