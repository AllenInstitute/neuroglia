import xarray as xr
from scipy import interpolate

def events_to_xr_dim(events,dim='event'):
    # builds the event dataframe into coords for xarray
    coords = events.to_dict(orient='list')
    coords = {k:(dim,v) for k,v in coords.items()}
    # define a DataArray that will describe the event dimension
    return xr.DataArray(
        events.index,
        name=dim,
        dims=[dim],
        coords=coords,
    )

def create_interpolator(t,y):
    """ creates a cubic spline interpolator

    Parameters
    ---------
    y : array-like of floats

    Returns
    ---------
    interpolator function that accepts a list of times

    """
    interpolator = interpolate.InterpolatedUnivariateSpline(t, y)
    return interpolator
