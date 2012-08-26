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


def lowess3d(x,y,z, w, x0, y0, z0, degree, kernel):
    """
    Locally linear regression with the LOWESS algorithm.

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

    


