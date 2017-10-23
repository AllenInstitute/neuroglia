from scipy import interpolate

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
