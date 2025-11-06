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

def Profile_from_gaussians(x_grid, params, Te_data, ne_data, zeta, area_element, vol_element):
    """"
    Creates a profile that is the sum of a number of gaussians 
    to begin with I will pass a set number of points.
    In the future I will pass a list of amplitudes and positions so it can be modular
    """
    #unpack params from dictionary
    #amplitude
    pow_1 = params['power_1']
    pow_2 = params['power_2']
    pow_3 = params['power_3']
    pow_4 = params['power_4']
    pow_5 = params['power_5']
    #width
    b_1 = params['width_1']
    b_2 = params['width_2']
    b_3 = params['width_3']
    b_4 = params['width_4']
    b_5 = params['width_5']
    #centre
    c_1 = params['centre_1']
    c_2 = params['centre_2']
    c_3 = params['centre_3']
    c_4 = params['centre_4']
    c_5 = params['centre_5']

    alpha = params['alpha']

    #create Te and ne spline 
    Te_spline = CubicSpline(Te_data[:,0], Te_data[:,1])
    ne_spline = CubicSpline(ne_data[:,0], ne_data[:,1])

    Te_1 = Te_spline(c_1)
    Te_2 = Te_spline(c_2)
    Te_3 = Te_spline(c_3)
    Te_4 = Te_spline(c_4)
    Te_5 = Te_spline(c_5)

    ne_1 = ne_spline(c_1)
    ne_2 = ne_spline(c_2)
    ne_3 = ne_spline(c_3)
    ne_4 = ne_spline(c_4)
    ne_5 = ne_spline(c_5)

    eta_1 = calculate_eta(Te_1, ne_1)
    eta_2 = calculate_eta(Te_2, ne_2)
    eta_3 = calculate_eta(Te_3, ne_3)
    eta_4 = calculate_eta(Te_4, ne_4)
    eta_5 = calculate_eta(Te_5, ne_5)

    index_1 = np.argmin(np.abs(x_grid-c_1))
    index_2 = np.argmin(np.abs(x_grid-c_2))
    index_3 = np.argmin(np.abs(x_grid-c_3))
    index_4 = np.argmin(np.abs(x_grid-c_4))
    index_5 = np.argmin(np.abs(x_grid-c_5))
    
    a_1 = (pow_1/vol_element[index_1]) * eta_1 * zeta[index_1] * area_element[index_1]
    a_2 = (pow_2/vol_element[index_2]) * eta_2 * zeta[index_2] * area_element[index_2]
    a_3 = (pow_3/vol_element[index_3]) * eta_3 * zeta[index_3] * area_element[index_3]
    a_4 = (pow_4/vol_element[index_4]) * eta_4 * zeta[index_4] * area_element[index_4]
    a_5 = (pow_5/vol_element[index_5]) * eta_5 * zeta[index_5] * area_element[index_5]

    gauss_1 = Gaussian_distribution(x_grid,a_1,b_1,c_1)
    gauss_2 = Gaussian_distribution(x_grid,a_2,b_2,c_2)
    gauss_3 = Gaussian_distribution(x_grid,a_3,b_3,c_3)
    gauss_4 = Gaussian_distribution(x_grid,a_4,b_4,c_4)
    gauss_5 = Gaussian_distribution(x_grid,a_5,b_5,c_5)

    gauss_1 = np.append(gauss_1, np.abs(pow_1)*alpha)
    gauss_2 = np.append(gauss_2, np.abs(pow_2)*alpha)
    gauss_3 = np.append(gauss_3, np.abs(pow_3)*alpha)
    gauss_4 = np.append(gauss_4, np.abs(pow_4)*alpha)
    gauss_5 = np.append(gauss_5, np.abs(pow_5)*alpha)

    Total_profile = gauss_1 + gauss_2 + gauss_3 + gauss_4 + gauss_5
    #amps = [pow_1, pow_2, pow_3, pow_4, pow_5]
    #Total_amplitude = np.sum(np.abs(amps))
    #Product_of_amplitude = np.prod(np.abs(amps))

    return Total_profile

def calculate_eta(Te_2,ne_2):
    """
    Calculates eta from relative changes in ne and te

    eta_1/eta_2 = (Te_1/Te_2)*(ne_2/ne_1)
    """

    eta_1   = 20 * 10**-3 #A/W
    Te_1    = 10          #Kev
    ne_1    = 10          #10^19 m^-3

    eta_2 = eta_1 * (Te_2/Te_1) * (ne_1/ne_2)

    return eta_2

def Calculate_CD_diff(x_grid, old_CD_data, new_CD_data):
    #creates interpolators for old and new data
    old_CD_rho = old_CD_data[0,:]
    old_CD_profile = old_CD_data[1,:]
    CD_old_interp = CubicSpline(old_CD_rho, old_CD_profile)
    CD_new_interp = CubicSpline(new_CD_data[0,:], new_CD_data[1,:])

    #Interpolate grid for CD calculation
    old_CD = CD_old_interp(x_grid)
    new_CD = CD_new_interp(x_grid)
    
    CD_diff = new_CD - old_CD

    return CD_diff

def residual(params, x_grid, CD_diff, Te_data, ne_data, zeta, dA, dV):
    #pass parameters to model
    model = Profile_from_gaussians(x_grid, params, Te_data, ne_data, zeta, dA, dV)

    return model - CD_diff 