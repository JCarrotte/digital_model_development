#imports 
from lmfit import Minimizer, Parameters, report_fit
import matplotlib.pyplot as plt 
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline

from plasma_control_system import plasma_control_system

def gaussian_distribution(x_grid, a, b, c):
    """
    Creates a gaussian distribution for a given ABC on a grid of x points
    a: amplitutde
    b: Variance (sqrt of stdev)
    c: mean value (centre of curve)
    """

    # gauss_y = a/np.sqrt(2*np.pi*b**2) * np.exp(-(x_grid-c)**2/(2*b**2))
    gauss_y = a * np.exp(-((x_grid - c) ** 2) / (2 * b**2))

    return gauss_y


def profile_from_gaussians(
    x_grid, params, Te_data, ne_data, zeta, area_element, vol_element
):
    """ "
    Creates a profile that is the sum of a number of gaussians
    to begin with I will pass a set number of points.
    In the future I will pass a list of amplitudes and positions so it can be modular
    """
    # unpack params from dictionary
    # amplitude
    pow_1 = params["power_1"]
    pow_2 = params["power_2"]
    pow_3 = params["power_3"]
    pow_4 = params["power_4"]
    pow_5 = params["power_5"]
    # width
    b_1 = params["width_1"]
    b_2 = params["width_2"]
    b_3 = params["width_3"]
    b_4 = params["width_4"]
    b_5 = params["width_5"]
    # centre
    c_1 = params["centre_1"]
    c_2 = params["centre_2"]
    c_3 = params["centre_3"]
    c_4 = params["centre_4"]
    c_5 = params["centre_5"]

    alpha = params["alpha"]

    # create Te and ne spline
    Te_spline = CubicSpline(Te_data[:, 0], Te_data[:, 1])
    ne_spline = CubicSpline(ne_data[:, 0], ne_data[:, 1])

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

    index_1 = np.argmin(np.abs(x_grid - c_1))
    index_2 = np.argmin(np.abs(x_grid - c_2))
    index_3 = np.argmin(np.abs(x_grid - c_3))
    index_4 = np.argmin(np.abs(x_grid - c_4))
    index_5 = np.argmin(np.abs(x_grid - c_5))

    a_1 = (pow_1 / vol_element[index_1]) * eta_1 * zeta[index_1] * area_element[index_1]
    a_2 = (pow_2 / vol_element[index_2]) * eta_2 * zeta[index_2] * area_element[index_2]
    a_3 = (pow_3 / vol_element[index_3]) * eta_3 * zeta[index_3] * area_element[index_3]
    a_4 = (pow_4 / vol_element[index_4]) * eta_4 * zeta[index_4] * area_element[index_4]
    a_5 = (pow_5 / vol_element[index_5]) * eta_5 * zeta[index_5] * area_element[index_5]

    gauss_1 = gaussian_distribution(x_grid, a_1, b_1, c_1)
    gauss_2 = gaussian_distribution(x_grid, a_2, b_2, c_2)
    gauss_3 = gaussian_distribution(x_grid, a_3, b_3, c_3)
    gauss_4 = gaussian_distribution(x_grid, a_4, b_4, c_4)
    gauss_5 = gaussian_distribution(x_grid, a_5, b_5, c_5)

    gauss_1 = np.append(gauss_1, np.abs(pow_1) * alpha)
    gauss_2 = np.append(gauss_2, np.abs(pow_2) * alpha)
    gauss_3 = np.append(gauss_3, np.abs(pow_3) * alpha)
    gauss_4 = np.append(gauss_4, np.abs(pow_4) * alpha)
    gauss_5 = np.append(gauss_5, np.abs(pow_5) * alpha)

    Total_profile = gauss_1 + gauss_2 + gauss_3 + gauss_4 + gauss_5
    # amps = [pow_1, pow_2, pow_3, pow_4, pow_5]
    # Total_amplitude = np.sum(np.abs(amps))
    # Product_of_amplitude = np.prod(np.abs(amps))

    return Total_profile


def calculate_eta(Te_2, ne_2):
    """
    Calculates eta from relative changes in ne and te

    eta_1/eta_2 = (Te_1/Te_2)*(ne_2/ne_1)
    """

    eta_1 = 20 * 10**-3  # A/W
    Te_1 = 10  # Kev
    ne_1 = 10  # 10^19 m^-3

    eta_2 = eta_1 * (Te_2 / Te_1) * (ne_1 / ne_2)

    return eta_2

def residual(params, x_grid, CD_diff, Te_data, ne_data, zeta, dA, dV):
    # pass parameters to model
    model = profile_from_gaussians(x_grid, params, Te_data, ne_data, zeta, dA, dV)

    return model - CD_diff

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