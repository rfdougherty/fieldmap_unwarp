import numpy as np
import numpy.testing as npt
import ..lowess as lo

def test_lowess3d():
    """ 

    Test local linear regression in 3d

    """
    # The space is sampled uniformly from -1 to 1:
    xyz = np.mgrid[-1:1:.1, -1:1:.1, -1:1:.1].reshape(3,20**3)
    x,y,z = xyz[0],xyz[1],xyz[2]
    # The w = f(x,y,z)
    w = sin(x) + cos(y) + tan(z)
    # Random sample of x,y,z combinations (between -1 and 1):
    x0 = np.random.randn(1000)
    y0 = np.random.randn(1000)
    z0 = np.random.randn(1000)
    # lowess3d is used to find the values at these sampling points:
    w0 = lo.lowess3d (x,y,z,w,x0.ravel(), y0.ravel(), z0.ravel())
    # evaluate f(x,y,z) in the uniformly sampled points:
    w0actual = sin(x0) + cos(y0) + tan(z0)
    # How did we do?
    npt.assert_array_almost_equal(w0, w0actual)
