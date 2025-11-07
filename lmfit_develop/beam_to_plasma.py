# imports
from base_model import perturb_plasma_profile


def beam_to_plasma(
    last_CD_profile,
    last_Te_profile,
    last_ne_profile,
    CD_change,
    Te_change,
    ne_change,
):
    """
    Takes the last set of plasma profiles and updates them based on the change from the plant
    Assumes all profiles and changes defined on the same grid
    """

    updated_CD = perturb_plasma_profile(last_CD_profile, CD_change)
    updated_Te = perturb_plasma_profile(last_Te_profile, Te_change)
    updated_ne = perturb_plasma_profile(last_ne_profile, ne_change)

    return updated_CD, updated_Te, updated_ne
