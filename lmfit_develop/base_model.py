#import
import numpy as np
import sys 
from scipy.interpolate import CubicSpline
from lmfit import minimize, Minimizer, Parameters, report_fit
import matplotlib.pyplot as plt
from pathlib import Path

#functions 
def Gaussian_distribution(x_grid, a,b,c):
    """
    Creates a gaussian distribution for a given ABC on a grid of x points
    a: amplitutde
    b: Variance (sqrt of stdev)
    c: mean value (centre of curve)
    """

    #gauss_y = a/np.sqrt(2*np.pi*b**2) * np.exp(-(x_grid-c)**2/(2*b**2))
    gauss_y = a * np.exp(-(x_grid-c)**2/(2*b**2))

    return gauss_y