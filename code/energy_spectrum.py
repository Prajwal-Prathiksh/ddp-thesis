r"""
Energy Spectrum of the Flow
#####################
References
-----------
    .. [energyspectrum] Energy_Spectrum: Script (with Example) to Compute the
    Kinetic Energy Spectrum of Periodic Turbulent Flows. Accessed 7 Nov. 2022.
"""
# Library imports.
import numpy as np
from pysph.base.kernels import WendlandQuinticC4
from compyle.api import annotate, Elementwise, wrap
from pysph.tools.interpolator import Interpolator
from pysph.solver.utils import load

#TODO: Compyle iterative functions


def calculate_energy_spectrum(
    u:np.ndarray, v:np.ndarray=None, w:np.ndarray=None, U0:float=1.0,
    debug:bool=False
):
    """
    Calculate the point-wise energy spectrum of the flow E(kx, ky, kz), from
    the normalised velocity spectrum of a flow.
    Note: For the calculation of the velocity spectrum, the flow is assumed
    to be periodic, and the velocity field data is from an equidistant grid of
    points.

    Parameters
    ----------
    u : np.ndarray
        Velocity field in x-direction.
    v : np.ndarray, optional
        Velocity field in y-direction.
    w : np.ndarray, optional
        Velocity field in z-direction.
    U0 : float, optional
        Reference velocity. The default is 1.
    debug : bool, optional
        Return the velocity spectrum as well. The default is False.

    Returns
    -------
    EK_U : np.ndarray
        Point-wise energy spectrum of the flow in x-direction.
    EK_V : np.ndarray
        Point-wise energy spectrum of the flow in y-direction.
    EK_W : np.ndarray
        Point-wise energy spectrum of the flow in z-direction.
    """
    # Import FFT-functions
    from numpy.fft import fftn as fftn
    from numpy.fft import fftshift as fftshift

    # Check shape of velocity components for given dimensions
    dim = len(u.shape)
    if dim == 1:
        if v is not None or w is not None:
            raise ValueError(
                "Velocity components v and w should be None for 1D data."
            )
    elif dim == 2:
        if v is None:
            raise ValueError(
                "Velocity component v should not be None for 2D data."
            )
        if w is not None:
            raise ValueError(
                "Velocity component w should be None for 2D data."
            )
    elif dim == 3:
        if v is None or w is None:
            raise ValueError(
                "Velocity components v or w should not be None for 3D data."
            )

    # Velocity field data
    v = v if v is not None else np.array([0.])
    w = w if w is not None else np.array([0.])

    # Get normalised velocity spectrum
    u_spectrum = np.abs(fftn(u / U0) / u.size)
    v_spectrum = np.abs(fftn(v / U0) / v.size)
    w_spectrum = np.abs(fftn(w / U0) / w.size)

    EK_U = fftshift(u_spectrum**2)
    EK_V = fftshift(v_spectrum**2)
    EK_W = fftshift(w_spectrum**2)

    if debug:
        return EK_U, EK_V, EK_W, u_spectrum, v_spectrum, w_spectrum
    else:
        return EK_U, EK_V, EK_W


def calculate_scalar_energy_spectrum(
    EK_U:np.ndarray, EK_V:np.ndarray=None, EK_W:np.ndarray=None,
    debug:bool=False
):
    """
    Calculate 1D energy spectrum of the flow E(k), from the point-wise energy
    spectrum E(kx, ky, kz), by integrating it over the surface of a sphere of
    radius k = (kx**2 + ky**2 + kz**2)**0.5.

    Parameters
    ----------
    EK_U : np.ndarray
        Point-wise energy spectrum of the flow in x-direction.
    EK_V : np.ndarray, optional
        Point-wise energy spectrum of the flow in y-direction.
    EK_W : np.ndarray, optional
        Point-wise energy spectrum of the flow in z-direction.
    debug : bool, optional
        Return the averaged energy spectrum as well. The default is False.

    Returns
    -------
    k : np.ndarray
        1D array of wave numbers.
    Ek : np.ndarray
        1D array of energy spectrum.
    """
    # Import numpy functions
    from numpy.linalg import norm as norm

    # Check shape of velocity components for given dimensions
    dim = len(np.shape(EK_U))
    if dim == 1:
        if EK_V is not None or EK_W is not None:
            raise ValueError(
                "Energy components EK_V and EK_W should be None for 1D data."
            )
        EK_U = np.array(EK_U)
    elif dim == 2:
        if EK_V is None:
            raise ValueError(
                "Energy component EK_V should not be None for 2D data."
            )
        if EK_W is not None:
            raise ValueError(
                "Energy component EK_W should be None for 2D data."
            )
        EK_U, EK_V = np.array(EK_U), np.array(EK_V)
    elif dim == 3:
        if EK_V is None or EK_W is None:
            raise ValueError(
                "Energy component EK_V or EK_W should not be None for 3D data."
            )
        EK_U, EK_V, EK_W = np.array(EK_U), np.array(EK_V), np.array(EK_W)

    eps = 1e-50

    box_side_x = np.shape(EK_U)[0]
    box_side_y = np.shape(EK_U)[1] if dim > 1 else 0
    box_side_z = np.shape(EK_U)[2] if dim > 2 else 0

    box_radius = int(
        1 + np.ceil(
            norm(np.array([box_side_x, box_side_y, box_side_z])) / 2
        )
    )

    center_x = int(box_side_x / 2)
    center_y = int(box_side_y / 2)
    center_z = int(box_side_z / 2)

    EK_U_sphere = np.zeros((box_radius, )) + eps
    EK_V_sphere = np.zeros((box_radius, )) + eps
    EK_W_sphere = np.zeros((box_radius, )) + eps

    if dim == 1:
        for i in range(box_side_x):
            wn = np.round(norm(i - center_x))
            wn = int(wn)

            EK_U_sphere[wn] += EK_U[i]

    elif dim == 2:
        for i in range(box_side_x):
            for j in range(box_side_y):
                wn = np.round(norm([i - center_x, j - center_y]))
                wn = int(wn)

                EK_U_sphere[wn] += EK_U[i, j]
                EK_V_sphere[wn] += EK_V[i, j]

    elif dim == 3:
        for i in range(box_side_x):
            for j in range(box_side_y):
                for k in range(box_side_z):
                    wn = np.round(
                        norm([i - center_x, j - center_y, k - center_z]))
                    wn = int(wn)

                    EK_U_sphere[wn] += EK_U[i, j, k]
                    EK_V_sphere[wn] += EK_V[i, j, k]
                    EK_W_sphere[wn] += EK_W[i, j, k]

    Ek = 0.5 * (EK_U_sphere + EK_V_sphere + EK_W_sphere)
    k = np.arange(0, len(Ek))

    if debug:
        return k, Ek, EK_U_sphere, EK_V_sphere, EK_W_sphere
    else:
        return k, Ek

def velocity_intepolator(
    fname:str, dim:int, kernel:object=None, Ni:int=101,
    domain_manager:object=None, **kwargs
):
    """
    Interpolate the energy spectrum of the flow from the given file.

    Parameters
    ----------
    fname : str
        Name of the file containing the flow data.
    dim : int
        Dimension of the flow.
    kernel : object, optional
        Kernel object. The default is WendlandQuinticC4.
    Ni : int, optional
        Number of points to interpolate the energy spectrum (Ni**2 for 2D data, Ni**3 for 3D data). The default is 101.
    domain_manager : object, optional
        DomainManager object. The default is None.
    **kwargs : dict, optional
        Additional keyword arguments to pass to the interpolator.

    Returns
    -------
    t : float
        Time of the flow data.
    ui : np.ndarray
        Interpolated velocity in x-direction.
    vi : np.ndarray
        Interpolated velocity in y-direction.
    wi : np.ndarray
        Interpolated velocity in z-direction.
    """
    # Load data
    data = load(fname)
    t = data["solver_data"]["t"]

    # Create meshgrid based on dimension
    _x = np.linspace(0, 1, Ni)
    if dim == 1:
        x = _x
        y = z = None
    elif dim == 2:
        x, y = np.meshgrid(_x, _x)
        z = None
    elif dim == 3:
        x, y, z = np.meshgrid(_x, _x, _x)
    else:
        raise ValueError("Dimension should be 1, 2 or 3.") 

    # Setup default interpolator properties
    if kernel is None:
        kernel = WendlandQuinticC4(dim=dim)
    
    # Interpolate velocity
    interp = Interpolator(
        list(data['arrays'].values()), x=x, y=y, z=z,
        kernel=kernel, domain_manager=domain_manager, **kwargs
    )
    if dim == 1:
        _u = interp.interpolate('u')
        ui = _u
        return t, ui
    elif dim == 2:
        _u = interp.interpolate('u')
        _v = interp.interpolate('v')
        ui = _u.reshape(Ni, Ni)
        vi = _v.reshape(Ni, Ni)
        return t, ui, vi
    elif dim == 3:
        _u = interp.interpolate('u')
        _v = interp.interpolate('v')
        _w = interp.interpolate('w')
        ui = _u.reshape(Ni, Ni, Ni)
        vi = _v.reshape(Ni, Ni, Ni)
        wi = _w.reshape(Ni, Ni, Ni)
        return t, ui, vi, wi