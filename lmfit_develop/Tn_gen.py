"""
A script to generate temperature and density profiles for ray tracing
"""

#standard imports 
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.constants as c
#import logger
#import netCDF4 as nc4
from scipy.interpolate import interp1d, RectBivariateSpline

rho = np.linspace(0,1,20)

def analytic_profile(x, y_0, y_1, a, b):
    """
    This function takes rho and produces an analytic value for a given paramater

    x = normalised radius array
    y_0 = paramater value at centre of flux profile
    y_1 = paramater value at LCFS
    a,b are powers to aid scailing (usually a=2, b=1)
    """

    return y_1 + (y_0-y_1)*(1-x**a)**b

#Tom suggested I scale the density relative to a frequency of the launched ray 

#freq = 

n_0 = 10; n_1 =0
T_0 = 10; T_1 = 0

n_analytic = analytic_profile(rho, n_0, n_1, 9, 1)
T_analytic = analytic_profile(rho, T_0, T_1, 9, 1)
print(n_analytic)

np.savetxt("n_analytic.txt", np.column_stack([rho,n_analytic]), delimiter=",")
np.savetxt("T_analytic.txt", np.column_stack([rho,T_analytic]), delimiter=",")