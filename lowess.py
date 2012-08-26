"""
lowess: Locally linear regression  
==================================

Implementation of the LOWESS algorithm in 3 dimensions

Friedman, Hastie and Tibshirani (2008). The Elements of Statistical Learning;
Chapter 6 

Cleveland (1979). Robust Locally Weighted Regression and Smoothing
Scatterplots. J American Statistical Association, 74: 829-836.

Loosely based on Matlab(c) code by Kendrick Kay 

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
    xx = np.abs(x-x0)
    idx = np.where(np.abs(xx<=1))
    return kernel(xx,idx)/l


def lowess(x, w, x0, kernel=epanechnikov, l=1):
    """
    Locally linear regression with the LOWESS algorithm in 1d

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
       The metric window size for 
    Returns
    -------
    The function estimated 

    Notes
    -----

    The solution to this problem is given by equation 6.8 in Friedman, Hastie
    and Tibshirani (2008). The Elements of Statistical Learning; Chapter 6 

    Example
    -------
    >>> import lowess as lo
    >>> import numpy as np
    >>> x = np.random.randn(100)
    >>> f = np.sin(x)
    >>> x0 = np.linspace(-1,1,10)
    >>> f_hat = lo.lowess(x, f, x0)
    >>> import matplotlib.pyplot as plt
    >>> fig,ax = plt.subplots(1)
    >>> ax.scatter(x,f)
    >>> ax.plot(x0,f_hat,'ro')
    >>> plt.show()

    """
    # For the case where x0 is provided as a scalar: 
    if not np.iterable(x0):
       x0 = np.asarray([x0])
    ans = np.empty(x0.shape) 
    # We only need one design matrix:
    B = np.vstack([np.ones(x.shape[0]), x]).T
    for idx, this_x0 in enumerate(x0):
        # Different weighting kernel for each x0:
        W = np.diag(do_kernel(this_x0, x, l=l, kernel=kernel))
        # Equation 6.8 in FHT:
        BtWB = np.dot(np.dot(B.T, W), B)
        BtW = np.dot(B.T, W)
        alpha, beta = np.dot(np.dot(la.inv(BtWB), BtW), w)
        # Estimate the answer based on the parameters:
        ans[idx] =  alpha + beta * this_x0 
    return ans

def lowess3d(x,y,z, w, x0, y0, z0, degree, kernel):
    """
    Locally linear regression with the LOWESS algorithm in 3d

    Parameters
    ----------

    x, y, z: float arrays of shapes (n,)
       These are the values of x,y and z at the different locations at which a
       measurement of w was performed.

    w: float array of shape (n,)
       w = f(x,y,z) is the value of a function of interest at the x,y,z
       values. 

    x0, y0, z0: float arrays of shape (m,)


    Returns
    -------

    The estimated values of f(x0,y0,z0)
    

    """
    pass
    



    

