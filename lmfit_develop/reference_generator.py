# Standard imports.
import netCDF4 as nc4
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

from base_model import gaussian_distribution


def profile_extension(n_ramp: int, n_flat_top: int, plasma_profile):
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
    # First create empty array in correct shape
    time_evolution = np.empty((2 * n_ramp + n_flat_top, len(plasma_profile)))

    # create linear array of products to multiply profile by
    ramp_multiplier = np.linspace(0.25, 1, n_ramp)

    # iterate over number of points to create "time" evolution of ramp up
    for i, x in enumerate(ramp_multiplier):
        time_evolution[i, :] = plasma_profile * x

    # add flat top elements
    for i in range(n_flat_top):
        i += n_ramp
        time_evolution[i, :] = plasma_profile

    # add ramp down
    for i, x in enumerate(ramp_multiplier[::-1]):
        i += n_ramp + n_flat_top
        time_evolution[i, :] = plasma_profile * x

    return time_evolution


def reference_generator(
    n_ramp: int,
    n_flat_top: int,
    rho_grid,
    ne_profile,
    Te_profile,
    plasma_current_profile,
    bootstrap_current_profile,
    plasma_area_profile,
    plasma_volume_profile,
):
    """
    desired plasma profiles are extended using the profile extension function.
    Each profile is returned as an extended 2D array

    inputs:
    n_ramp & n_flat_top: intigers defining the length of ramp-up/down and flat-top
    rho_grid: the rho grid that the profiles are defined on
    the rest are profiles against rho_grid depicting a property of the plasma
    """

    # create time evolution of each property
    ne_evolution = profile_extension(n_ramp, n_flat_top, ne_profile)

    Te_evolution = profile_extension(n_ramp, n_flat_top, Te_profile)

    plasma_current_evolution = profile_extension(
        n_ramp, n_flat_top, plasma_current_profile
    )

    bootstrap_evolution = profile_extension(
        n_ramp, n_flat_top, bootstrap_current_profile
    )

    area_evolution = profile_extension(n_ramp, n_flat_top, plasma_area_profile)

    volume_evolution = profile_extension(n_ramp, n_flat_top, plasma_volume_profile)

    return (
        ne_evolution,
        Te_evolution,
        plasma_current_evolution,
        bootstrap_evolution,
        area_evolution,
        volume_evolution,
    )


# test
def test():
    x = np.linspace(0, 1, 1001)
    test_profile = gaussian_distribution(x, 1, 0.01, 0.1)

    ramp_steps = 16
    flat_top_steps = 10

    test_evol = profile_extension(ramp_steps, flat_top_steps, test_profile)
    print(np.shape(test_evol))

    fig, ax = plt.subplots()
    for i in range(2 * ramp_steps + flat_top_steps):
        shift = i * 0.8 / (2 * ramp_steps + flat_top_steps)
        ax.plot(x + shift, test_evol[i, :])

    ax.set_xlim(0, 1)
    plt.show()


if __name__ == "__main__":
    test()
