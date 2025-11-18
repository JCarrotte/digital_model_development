# Standard imports.
import netCDF4 as nc4
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

def profile_extension(
        n_ramp:int, 
        n_flat_top:int,
        plasma_profile
        ):
    """
    When given a plasma profiles and a number of points in ramp-up/down and flat-top,
    creates a time series of artificial ramp-up, steady state and ramp-down.
    Assumes ramp-up and down are identical if reversed

    inputs:
    n_ramp: number of time points in ramp-up and ramp-down 
    n_flat_top: number of time points in flat-top
    plasma_profile: Plasma paramater to be extended in time

    output: 
    2D numpy array of paramater evoloution against time and radius 
    """
    #First create empty array in correct shape
    time_evolution = np.empty(np.shape(2 * n_ramp + n_flat_top, len(plasma_profile)))
    
    #create linear array of products to multiply profile by
    ramp_multiplier = np.linspace(0.25, 1, n_ramp)

    #iterate over number of points to create "time" evolution of ramp up
    for i, x in enumerate(ramp_multiplier):
        time_evolution[i, :] = plasma_profile * x

    #add flat top elements
    for i in n_ramp + range(n_flat_top):
        time_evolution[i, :] = plasma_profile

    #add ramp down
    for i, x in enumerate(ramp_multiplier[::-1]):
        i =+ n_ramp + n_flat_top
        time_evolution[i, :] = plasma_profile * x

    return time_evolution



def reference_generator(
    n_ramp:int, 
    n_flat_top:int,
    rho_grid,
    ne_profile,
    Te_profile,
    plasma_current_profile,
    boostrap_current_profile,
    plasma_area_profile,
    plasma_volume_profile
    ):
    """
    desired plasma profiles are extended using the profile extension function.
    Each profile is returned as  an extended 2D array 
    """