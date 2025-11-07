#imports 
from lmfit import Minimizer, Paramaters, report_fit
import matplotlib.pyplot as plt 

from base_model import residual

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