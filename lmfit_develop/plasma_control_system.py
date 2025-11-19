# imports
from scipy.interpolate import CubicSpline


def plasma_control_system_diff_rho(x_grid, old_CD_data, recipe_CD_data, bootstrap_data):
    """
    Takes the Current Drive (CD) profiles from the Beam-to-Plasma
    and plasma recipe and calculates the required increase in current.

    Also takes into account the bootstrap current
    """

    # creates interpolators for all data to insure they are compatible
    CD_old_interp = CubicSpline(old_CD_data[0, :], old_CD_data[1, :])
    CD_recipe_interp = CubicSpline(recipe_CD_data[0, :], recipe_CD_data[1, :])
    bootstrap_interp = CubicSpline(bootstrap_data[0, :], bootstrap_data[1, :])
    # In the future I would like to calculate bs_current/perturbation from ne and Te

    # interpolate onto grid used for CD calculation
    old_CD = CD_old_interp(x_grid)
    recipe_CD = CD_recipe_interp(x_grid)
    bs_CD = bootstrap_interp(x_grid)

    CD_diff = recipe_CD - old_CD - bs_CD

    return CD_diff


def plasma_control_system(
    x_grid, old_CD_profile, recipe_CD_profile, bootstrap_CD_profile
):
    """
    Takes the Current Drive (CD) profiles from the Beam-to-Plasma
    and plasma recipe and calculates the required increase in current.
    Assumes all profiles are defined on the same grid

    Also takes into account the bootstrap current
    """

    CD_diff = recipe_CD_profile - old_CD_profile - bootstrap_CD_profile

    return CD_diff
