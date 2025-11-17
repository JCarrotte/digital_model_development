#imports 
from lmfit import Minimizer, Parameters, report_fit
import matplotlib.pyplot as plt 
from pathlib import Path
import numpy as np

from base_model import residual, gaussian_distribution, profile_from_gaussians
from plasma_control_system import plasma_control_system

def HCD_controller(CD_diff, fitting_params, x_grid, Te_data, ne_data, zeta, area_elements, volume_elements):
    """
    
    """


    minimization = Minimizer(
        residual,
        fitting_params,
        fcn_args=(x_grid, CD_diff, Te_data, ne_data, zeta, area_elements, volume_elements)
    )

    min_result = minimization.minimize()

    return(min_result)

#A test to run through the new functions 

# paramaters
# Create and populate parameter space
fit_params = Parameters()
#fit_params.add("total_power", value = 3000, vary = False)
fit_params.add("power_1", value=0.1, min=0.0)
fit_params.add("power_2", value=0.1, min=0.0)
fit_params.add("power_3", value=0.1, min=0.0)
fit_params.add("power_4", value=0.1, min=0.0)
fit_params.add("power_5", value=0.1, min=0.0)
fit_params.add("width_1", value=0.2, vary=False)
fit_params.add("width_2", value=0.2, vary=False)
fit_params.add("width_3", value=0.2, vary=False)
fit_params.add("width_4", value=0.2, vary=False)
fit_params.add("width_5", value=0.2, vary=False)
fit_params.add("centre_1", value=0.0, vary=False)  # min=(0.1-0.01), max=(0.1+0.01))
fit_params.add("centre_2", value=0.2, vary=False)  # min=(0.3-0.01), max=(0.3+0.01))
fit_params.add("centre_3", value=0.4, vary=False)  # min=(0.5-0.01), max=(0.5+0.01))
fit_params.add("centre_4", value=0.6, vary=False)  # min=(0.7-0.01), max=(0.7+0.01))
fit_params.add("centre_5", value=0.8, vary=False)  # min=(0.9-0.01), max=(0.9+0.01))
fit_params.add("alpha", value=0.0, vary=False)
#fit_params.add("total_power", expr='power_1 + power_2 + power_3 + power_4 + power_5', max = 9000)


# create CD_data
# CD old data
CD_old_coords = np.linspace(0.1, 1.0, 101)
CD_old = 0.2 * np.sin(CD_old_coords) + 5
CD_old_data = np.empty((2, len(CD_old_coords)))
CD_old_data[0, :] = CD_old_coords
CD_old_data[1, :] = CD_old

# CD new data
CD_new_coords = np.linspace(0, 1, 101)
# CD_new = 0.2*np.cos(4*CD_new_coords) + 20
CD_new = (
    CD_old
    + gaussian_distribution(CD_old_coords, 2, 0.3, 0.4)
    + gaussian_distribution(CD_old_coords, 3, 0.2, 0.8)
)
CD_new_data = np.empty((2, len(CD_new_coords)))
CD_new_data[0, :] = CD_new_coords
CD_new_data[1, :] = CD_new

BS_data = np.empty((2, len(CD_new_coords)))
BS_data[0, :] = CD_new_coords
BS_data[1, :] = np.zeros(len(CD_new_coords))

# area and volume array
R = 3  # m
r = np.linspace(0.1, 1.5, 101)  # m
dArea = np.pi * r * r  # m^2
dVolume = 2 * np.pi * R * dArea  # m^3

def test():
    # filepaths
    base_directory = Path(__file__).parent
    Te_file = base_directory.joinpath("T_analytic.txt")
    ne_file = base_directory.joinpath("n_analytic.txt")

    # load in data files
    Te_profile = np.loadtxt(Te_file, delimiter=",")
    ne_profile = np.loadtxt(ne_file, delimiter=",")

    # rho_grid
    rho = np.linspace(0, 1, 101)

    # Zeta array
    Zeta = np.ones(len(rho))

    # load in data files
    Te_profile = np.loadtxt(Te_file, delimiter=",")
    ne_profile = np.loadtxt(ne_file, delimiter=",")

    CD_diff = plasma_control_system(rho, CD_old_data, CD_new_data, BS_data)
    CD_diff = np.append(CD_diff, 0)

    result = HCD_controller(CD_diff, fit_params, rho, Te_profile, ne_profile, Zeta, dArea, dVolume)

    report_fit(result)

    fit_gaussian = profile_from_gaussians(
        rho, result.params, Te_profile, ne_profile, Zeta, dArea, dVolume
    )
    
    fig, ax_CD = plt.subplots()
    ax_CD.set_xlabel(r"Rho ($\rho_P$)")
    ax_CD.set_ylabel("Current (Arb units)")

    ax_CD.plot(rho, CD_diff[:-1], "+-", alpha=0.5)
    ax_CD.plot(rho, fit_gaussian[:-1], "rx--", label="Unperturbed profiles")

    plt.show()


if __name__ == "__main__":
    test()