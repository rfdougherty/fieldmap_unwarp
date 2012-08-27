"""
lowess: Locally linear regression  
==================================

Implementation of the LOWESS algorithm in n dimensions.

Friedman, Hastie and Tibshirani (2008). The Elements of Statistical Learning;
Chapter 6 

Cleveland (1979). Robust Locally Weighted Regression and Smoothing
Scatterplots. J American Statistical Association, 74: 829-836.

"""
import numpy as np
import scipy.linalg as la

# Kernel functions:
def epanechnikov(xx, idx):
    ans = np.zeros(xx.shape)
    ans[idx] = 0.75 * (1-xx[idx]**2)
    return ans

def tri_cube(xx, idx):
    ans = np.zeros(xx.shape)
    ans[idx] = (1-np.abs(xx[idx])**3)**3
    return ans

def do_kernel(x0, x, l=1.0, kernel=epanechnikov):
    """
    Calculate a kernel function on x in the neighborhood of x0

    Parameters
    ----------
    x: float array
       All values of x
    x0: float
       The value of x around which we evaluate the kernel
    l: float or float array (with shape = x.shape)
       Width parameter (metric window size)
    
    """
    # xx is the norm of x-x0. Note that we broadcast on the second axis for the
    # nd case and then sum on the first to get the norm in each value of x:
    xx = np.sum(np.sqrt(np.power(x-x0[:,np.newaxis], 2)), 0)
    idx = np.where(np.abs(xx<=1))
    return kernel(xx,idx)/l


def lowess(x, w, x0, kernel=epanechnikov, l=1):
    """
    Locally linear regression with the LOWESS algorithm.

    Parameters
    ----------
    x: float array
       Values of x for which f(x) is known (e.g. measured)
    
    w: float array
       The known values of f(x) at these points 

    x0: float or float array.
        Values of x for which we estimate the value of f(x) 

    kernel: callable
        A kernel function. {'epanechnikov', 'tri_cube'}

    l: float or float array with shape = x.shape
       The metric window size for the kernel
 
    Returns
    -------
    The function estimated at x0. 

    Notes
    -----

    The solution to this problem is given by equation 6.8 in Friedman, Hastie
    and Tibshirani (2008). The Elements of Statistical Learning (Chapter 6).

    Example
    -------
    >>> import lowess as lo
    >>> import numpy as np

    # For the 1D case:
    >>> x = np.random.randn(100)
    >>> f = np.sin(x)
    >>> x0 = np.linspace(-1,1,10)
    >>> f_hat = lo.lowess(x, f, x0)
    >>> import matplotlib.pyplot as plt
    >>> fig,ax = plt.subplots(1)
    >>> ax.scatter(x,f)
    >>> ax.plot(x0,f_hat,'ro')
    >>> plt.show()

    # 2D case (and more...)
    >>> x = np.random.randn(2, 100)
    >>> f = -1 * np.sin(x[0]) + 0.5 * np.cos(x[1])
    >>> x0 = np.mgrid[-1:1:.1, -1:1:.1]
    >>> x0 = np.vstack([x0[0].ravel(), x0[1].ravel()])
    >>> f_hat = lo.lowess(x, f, x0, kernel=lo.tri_cube)
    >>> from mpl_toolkits.mplot3d import Axes3D
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111, projection='3d')
    >>> ax.scatter(x[0], x[1], f)
    >>> ax.scatter(x0[0], x0[1], f_hat, color='r')

    """
    # For the case where x0 is provided as a scalar: 
    if not np.iterable(x0):
       x0 = np.asarray([x0])
    ans = np.zeros(x0.shape[-1]) 
    # We only need one design matrix:
    B = np.vstack([np.ones(x.shape[-1]), x]).T
    for idx, this_x0 in enumerate(x0.T):
        # This is necessary in the 1d case (?):
        if not np.iterable(this_x0):
            this_x0 = np.asarray([this_x0])
        # Different weighting kernel for each x0:
        W = np.diag(do_kernel(this_x0, x, l=l, kernel=kernel))
        # XXX It should be possible to calculate W outside the loop, if x0 and
        # x are both sampled in some regular fashion (that is, if W is the same
        # matrix in each iteration). That should save time.
        try: 
            # Equation 6.8 in FHT:
            BtWB = np.dot(np.dot(B.T, W), B)
            BtW = np.dot(B.T, W)
            # Get the params:
            beta = np.dot(np.dot(la.inv(BtWB), BtW), w.T)
            # Estimate the answer based on the parameters:
            ans[idx] += beta[0] + np.dot(beta[1:], this_x0)
        # If we are trying to sample far away from where the function is
        # defined, we will be trying to invert a singular matrix. In that case,
        # the regression should not work for you and you should get a nan:
        except la.LinAlgError:
            ans[idx] += np.nan

    return ans.T


