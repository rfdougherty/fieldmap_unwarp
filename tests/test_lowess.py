import numpy as np
import numpy.testing as npt
import ..lowess as lo

def test_lowess3d():
    """ 

    Test local linear regression in 3d

    """

    x = np.random.randn(1000)
    y = np.random.randn(1000)
    z = np.random.randn(1000)
    w = sin(x) + cos(y) + tan(z)
    x0,y0,z0 = np.mgrid[-1:1:.1, -1:1:.1, -1:1:.1]
    w0 = lo.lowess3d (x,y,z,w,x0,y0,z0)
    w0actual = sin(x0) + cos(y0) + tan(z0)
    npt.assert_equal(w0, w0actual)
