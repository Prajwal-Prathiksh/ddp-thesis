r"""
Energy Spectrum of the Flow
#####################
References
-----------
    .. [navah2020high] Navah, Farshad, Marta de la Llave Plata, and Vincent
    Couaillier. "A high-order multiscale approach to turbulence for compact
    nodal schemes." Computer methods in applied mechanics and engineering 363
    (2020): 112885.

    .. [energyspectrum] Energy_Spectrum: Script (with Example) to Compute the
    Kinetic Energy Spectrum of Periodic Turbulent Flows. Accessed 7 Nov. 2022.
"""
# Library imports.
import numpy as np
from compyle.api import annotate, Elementwise, wrap

# Get normalised velocity spectrum (numpy fft)
# Get E(kx...)
# Compyle iterative functions (backend)
# Get E(k)


def calculate_energy_spectrum(
    u, v=None, w=None, U0=1., debug=False
):
    """
    Calculate the point-wise energy spectrum of the flow, from the normalised
    velocity spectrum of a flow.
    Note: For the calculation of the velocity spectrum, the flow is assumed
    to be periodic, and the velocity field data is from an equidistant grid of
    points.

    Parameters
    ----------
    u : array_like
        Velocity field in x-direction.
    v : array_like, optional
        Velocity field in y-direction.
    w : array_like, optional
        Velocity field in z-direction.
    U0 : float, optional
        Reference velocity. The default is 1.
    debug : bool, optional
        Return the velocity spectrum as well. The default is False.

    Returns
    -------
    Ek_U : array_like
        Point-wise energy spectrum of the flow in x-direction.
    Ek_V : array_like
        Point-wise energy spectrum of the flow in y-direction.
    Ek_W : array_like
        Point-wise energy spectrum of the flow in z-direction.
    """
    # Import FFT-functions
    from numpy.fft import fftn as fftn
    from numpy.fft import fftshift as fftshift

    # Velocity field data
    v = v if v is not None else np.array([0.])
    w = w if w is not None else np.array([0.])

    # Get normalised velocity spectrum
    u_spectrum = np.abs(fftn(u / U0) / u.size)
    v_spectrum = np.abs(fftn(v / U0) / v.size)
    w_spectrum = np.abs(fftn(w / U0) / w.size)

    Ek_U = fftshift(u_spectrum**2)
    Ek_V = fftshift(v_spectrum**2)
    Ek_W = fftshift(w_spectrum**2)

    if debug:
        return Ek_U, Ek_V, Ek_W, u_spectrum, v_spectrum, w_spectrum
    else:
        return Ek_U, Ek_V, Ek_W
