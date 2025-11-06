# imports
from scipy.interpolate import CubicSpline
from base_model import Calculate_CD_diff


def plasma_control_system(x_grid, old_CD_data, recipe_CD_data, bootstrap_data):
    """
    Takes the Current Drive (CD) profiles from the Beam-to-Plasma
    and plasma recipe and calculates the required increase in current.

    Also takes into account the bootstrap current
    """

    # creates interpolators for all data to insure they are compatible
    CD_old_interp = CubicSpline(old_CD_data[0, :], old_CD_data[1, :])
    CD_recipe_interp = CubicSpline(recipe_CD_data[0, :], recipe_CD_data[1, :])
    bootstrap_interp = CubicSpline(bootstrap_data[0, :], bootstrap_data[1, :])

    # interpolate onto grid used for CD calculation
    old_CD = CD_old_interp(x_grid)
    recipe_CD = CD_recipe_interp(x_grid)
    bs_CD = bootstrap_interp(x_grid)

    CD_diff = recipe_CD - old_CD

    return CD_diff
