import numpy as np
import numpy.testing as npt
import lowess as lo


def test_lowess():
    """
    Test 1-d local linear regression with lowess

    """
    x = np.random.randn(100)
    f = np.sin(x)
    x0 = np.linspace(-1,1,10)
    f_hat = lo.lowess(x, f, x0)
    f_real = np.sin(x0)
    npt.assert_array_almost_equal(f_hat, f_real, decimal=1)


def test_lowess2d(): 
    """

    """
    x = np.random.randn(2, 100)
    f = -1 * np.sin(x[0]) + 0.5 * np.cos(x[1])
    x0 = np.mgrid[-1:1:.1, -1:1:.1]
    x0 = np.vstack([x0[0].ravel(), x0[1].ravel()])
    f_hat = lo.lowess(x, f, x0, kernel=lo.tri_cube) # Try this one for a change.
    f_real = -1 * np.sin(x0[0]) + 0.5 * np.cos(x0[1])
    npt.assert_array_almost_equal(f_hat, f_real, decimal=1)
                                               

    
def test_lowess3d():
     """ 
     Test local linear regression in 3d with lowess
     """

     # Random sample of x,y,z combinations (between 0 and 1):
     x = np.random.randn(100)
     y = np.random.randn(100)
     z = np.random.randn(100)
     # The w = f(x,y,z)
     w = -1 * np.sin(x) + 0.5 * np.cos(y) + np.cos(z)

     # We sample within the bounds:
     max_min = np.max(np.min(np.vstack([x,y,z]),-1))
     min_max = np.min(np.max(np.vstack([x,y,z]),-1))
     xyz0 = np.mgrid[max_min+0.1:min_max-0.1:.1,
                     max_min+0.1:min_max-0.1:.1,
                     max_min+0.1:min_max-0.1:.1]
     
     x0,y0,z0 = xyz0[0].ravel(),xyz0[1].ravel(),xyz0[2].ravel()

     # lowess3d is used to find the values at these sampling points:
     w0 = lo.lowess(np.vstack([x,y,z]),
                    w,
                    np.vstack([x0, y0, z0]),
                    l=0.2,
                    kernel=lo.tri_cube)

     # evaluate f(x,y,z) in the uniformly sampled points:
     w0actual = np.sin(x0) + np.cos(y0) + np.tan(z0)
     # This will be undefined in manay places 
     npt.assert_array_almost_equal(w0, w0actual, decimal=1)
