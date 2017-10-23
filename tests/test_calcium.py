from neuroglia.calcium import MedianFilterDetrend, SavGolFilterDetrend
from neuroglia.calcium import OASISInferer

import numpy as np
import pandas as pd
import xarray as xr

import numpy.testing as npt
import xarray.testing as xrt


TIME = np.arange(0, 100, 1/30.)
LBL = ['a', 'b', 'c']
data = np.random.randn(len(TIME), 3)

DFF = pd.DataFrame(data, TIME, LBL)

def test_MedianFilterDetrend():
    tmp = MedianFilterDetrend().fit_transform(DFF)

def test_SavGolFilterDetrend():
    tmp = SavGolFilterDetrend().fit_transform(DFF)

def test_OASISInferer():
    tmp = OASISInferer().fit_transform(DFF)
    tmp = OASISInferer().fit_transform(SavGolFilterDetrend().fit_transform(DFF))
    
if __name__ == '__main__':
    test_MedianFilterDetrend()
    test_SavGolFilterDetrend()
    test_OASISInferer()
