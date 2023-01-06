r"""
Energy Spectrum of the Flow
###
References
-----------
    .. [energyspectrum] Energy_Spectrum: Script (with Example) to Compute the
    Kinetic Energy Spectrum of Periodic Turbulent Flows. Accessed 7 Nov. 2022.
"""
# Library imports
import itertools as IT
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from pysph.base.kernels import WendlandQuinticC4
from pysph.tools.interpolator import Interpolator
from pysph.solver.utils import load
from compyle.api import annotate, wrap, declare
from compyle.low_level import cast
from compyle.parallel import elementwise
from compyle.config import use_config
from math import sqrt, floor

# Local imports
from automate_utils import styles


def compute_energy_spectrum(
    u: np.ndarray, v: np.ndarray = None, w: np.ndarray = None, U0: float = 1.0,
    debug: bool = False
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
        Reference velocity. Default is 1.
    debug : bool, optional
        Return the velocity spectrum as well. Default is False.

    Returns
    -------
    ek_u : np.ndarray
        Point-wise energy spectrum of the flow in x-direction.
    ek_v : np.ndarray
        Point-wise energy spectrum of the flow in y-direction.
    ek_w : np.ndarray
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

    ek_u = fftshift(0.5 * u_spectrum**2)
    ek_v = fftshift(0.5 * v_spectrum**2)
    ek_w = fftshift(0.5 * w_spectrum**2)

    if debug:
        return ek_u, ek_v, ek_w, u_spectrum, v_spectrum, w_spectrum
    else:
        return ek_u, ek_v, ek_w


def _get_scalar_energy_spectrum_params(ek_u, ek_v, ek_w):
    """
    Get parameters for the scalar energy spectrum calculation.

    Parameters
    ----------
    ek_u : np.ndarray
        Point-wise energy spectrum of the flow in x-direction.
    ek_v : np.ndarray
        Point-wise energy spectrum of the flow in y-direction.
    ek_w : np.ndarray
        Point-wise energy spectrum of the flow in z-direction.

    Returns
    -------
    dim : int
        Dimension of the flow.
    box_dimensions : dict
        Dimensions of the box containing the flow.
    ek_u_sphere, ek_v_sphere, ek_w_sphere : np.ndarray
        1D zero arrays for the energy spectrum of the flow in spherical
        coordinates.
    """
    # Check shape of velocity components for given dimensions
    dim = len(np.shape(ek_u))
    if dim == 1:
        if ek_v is not None or ek_w is not None:
            raise ValueError(
                "Energy components ek_v and ek_w should be None for 1D data."
            )
        ek_u = np.array(ek_u, dtype=np.float64)
    elif dim == 2:
        if ek_v is None:
            raise ValueError(
                "Energy component ek_v should not be None for 2D data."
            )
        if ek_w is not None:
            raise ValueError(
                "Energy component ek_w should be None for 2D data."
            )
        ek_u = np.array(ek_u, dtype=np.float64)
        ek_v = np.array(ek_v, dtype=np.float64)
    elif dim == 3:
        if ek_v is None or ek_w is None:
            raise ValueError(
                "Energy component ek_v or ek_w should not be None for 3D data."
            )
        ek_u = np.array(ek_u, dtype=np.float64)
        ek_v = np.array(ek_v, dtype=np.float64)
        ek_w = np.array(ek_w, dtype=np.float64)

    box_side_x = np.shape(ek_u)[0]
    box_side_y = np.shape(ek_u)[1] if dim > 1 else 0
    box_side_z = np.shape(ek_u)[2] if dim > 2 else 0

    tmp = np.array([box_side_x, box_side_y, box_side_z], dtype=np.float64)
    box_radius = int(1 + np.ceil(np.linalg.norm(tmp) / 2))

    center_x = int(box_side_x / 2)
    center_y = int(box_side_y / 2)
    center_z = int(box_side_z / 2)

    box_dimensions = dict(
        box_side_x=box_side_x, box_side_y=box_side_y, box_side_z=box_side_z,
        box_radius=box_radius,
        center_x=center_x, center_y=center_y, center_z=center_z
    )

    ek_u_sphere = np.zeros((box_radius, ))
    ek_v_sphere = np.zeros((box_radius, ))
    ek_w_sphere = np.zeros((box_radius, ))

    return dim, box_dimensions, ek_u_sphere, ek_v_sphere, ek_w_sphere


def compute_scalar_energy_spectrum_python(
    ek_u: np.ndarray, ek_v: np.ndarray = None, ek_w: np.ndarray = None,
    ord: int = 2, debug: bool = False
):
    """
    Calculate 1D energy spectrum of the flow E(k), from the point-wise energy
    spectrum E(kx, ky, kz), by integrating it over the surface of a sphere of
    radius k = (kx**2 + ky**2 + kz**2)**0.5.

    Parameters
    ----------
    ek_u : np.ndarray
        Point-wise energy spectrum of the flow in x-direction.
    ek_v : np.ndarray, optional
        Point-wise energy spectrum of the flow in y-direction.
    ek_w : np.ndarray, optional
        Point-wise energy spectrum of the flow in z-direction.
    ord : int, optional
        Order of the norm. Default is 2.
    debug : bool, optional
        Return the averaged energy spectrum as well. Default is False.

    Returns
    -------
    k : np.ndarray
        1D array of wave numbers.
    ek : np.ndarray
        1D array of energy spectrum.
    """
    # Import numpy functions
    from numpy.linalg import norm as norm

    dim, box_dimensions, ek_u_sphere, ek_v_sphere, ek_w_sphere =\
        _get_scalar_energy_spectrum_params(ek_u, ek_v, ek_w)
    box_side_x = box_dimensions['box_side_x']
    box_side_y = box_dimensions['box_side_y']
    box_side_z = box_dimensions['box_side_z']
    center_x = box_dimensions['center_x']
    center_y = box_dimensions['center_y']
    center_z = box_dimensions['center_z']

    if dim == 1:
        for i in range(box_side_x):
            wn = np.round(norm([i - center_x], ord=ord))
            wn = int(wn)

            ek_u_sphere[wn] += ek_u[i]

    elif dim == 2:
        for i in range(box_side_x):
            for j in range(box_side_y):
                wn = np.round(norm([i - center_x, j - center_y], ord=ord))
                wn = int(wn)

                ek_u_sphere[wn] += ek_u[i, j]
                ek_v_sphere[wn] += ek_v[i, j]

    elif dim == 3:
        for i in range(box_side_x):
            for j in range(box_side_y):
                for k in range(box_side_z):
                    wn = np.round(
                        norm(
                            [i - center_x, j - center_y, k - center_z], ord=ord
                        )
                    )
                    wn = int(wn)

                    ek_u_sphere[wn] += ek_u[i, j, k]
                    ek_v_sphere[wn] += ek_v[i, j, k]
                    ek_w_sphere[wn] += ek_w[i, j, k]

    ek = 0.5 * (ek_u_sphere + ek_v_sphere + ek_w_sphere)
    k = np.arange(0, len(ek))

    if debug:
        return k, ek, ek_u_sphere, ek_v_sphere, ek_w_sphere
    return k, ek


@njit
def _compute_ek_from_1d_numba_helper(
    ek_u: np.ndarray, ek_u_sphere: np.ndarray, box_side_x: int,
    center_x: int, ord: int = np.inf,
):
    """
    Numba helper function for computing the scalar energy spectrum from 1D
    data.

    Parameters
    ----------
    ek_u : np.ndarray
        Point-wise energy spectrum of the flow in x-direction. Should be 1D.
    ek_u_sphere : np.ndarray
        1D array of energy spectrum for x-component of velocity.
    box_side_x : int
        Side length of the box in x-direction.
    center_x : int
        Center length of the box in x-direction.
    ord : int, optional
        Order of the norm. Default is np.inf.

    Returns
    -------
    ek_u_sphere : np.ndarray
    """
    for i in range(box_side_x):
        tmp = np.array([i - center_x], dtype=np.float64)
        wn = int(np.round(np.linalg.norm(tmp, ord=ord)))

        ek_u_sphere[wn] += ek_u[i]

    return ek_u_sphere


@njit
def _compute_ek_from_2d_numba_helper(
    ek_u: np.ndarray, ek_v: np.ndarray, ek_u_sphere: np.ndarray,
    ek_v_sphere: np.ndarray, box_side_x: int, box_side_y: int, center_x: int,
    center_y: int, ord: int = np.inf
):
    """
    Numba helper function for computing the scalar energy spectrum from 2D
    data.

    Parameters
    ----------
    ek_u : np.ndarray
        Point-wise energy spectrum of the flow in x-direction. Should be 2D.
    ek_v : np.ndarray
        Point-wise energy spectrum of the flow in y-direction. Should be 2D.
    ek_u_sphere : np.ndarray
        1D array of energy spectrum for x-component of velocity.
    ek_v_sphere : np.ndarray
        1D array of energy spectrum for y-component of velocity.
    box_side_x : int
        Side length of the box in x-direction.
    box_side_y : int
        Side length of the box in y-direction.
    center_x : int
        Center length of the box in x-direction.
    center_y : int
        Center length of the box in y-direction.
    ord : int, optional
        Order of the norm. Default is np.inf.

    Returns
    -------
    ek_u_sphere : np.ndarray
    ek_v_sphere : np.ndarray
    """
    for i in range(box_side_x):
        for j in range(box_side_y):
            tmp = np.array([i - center_x, j - center_y], dtype=np.float64)
            wn = int(np.round(np.linalg.norm(tmp, ord=ord)))

            ek_u_sphere[wn] += ek_u[i, j]
            ek_v_sphere[wn] += ek_v[i, j]

    return ek_u_sphere, ek_v_sphere


@njit
def _compute_ek_from_3d_numba_helper(
    ek_u: np.ndarray, ek_v: np.ndarray, ek_w: np.ndarray,
    ek_u_sphere: np.ndarray, ek_v_sphere: np.ndarray, ek_w_sphere: np.ndarray,
    box_side_x: int, box_side_y: int, box_side_z: int, center_x: int,
    center_y: int, center_z: int, ord: int = np.inf
):
    """
    Numba helper function for computing the scalar energy spectrum from 3D
    data.

    Parameters
    ----------
    ek_u : np.ndarray
        Point-wise energy spectrum of the flow in x-direction. Should be 3D.
    ek_v : np.ndarray
        Point-wise energy spectrum of the flow in y-direction. Should be 3D.
    ek_w : np.ndarray
        Point-wise energy spectrum of the flow in z-direction. Should be 3D.
    ek_u_sphere : np.ndarray
        1D array of energy spectrum for x-component of velocity.
    ek_v_sphere : np.ndarray
        1D array of energy spectrum for y-component of velocity.
    ek_w_sphere : np.ndarray
        1D array of energy spectrum for z-component of velocity.
    box_side_x : int
        Side length of the box in x-direction.
    box_side_y : int
        Side length of the box in y-direction.
    box_side_z : int
        Side length of the box in z-direction.
    center_x : int
        Center length of the box in x-direction.
    center_y : int
        Center length of the box in y-direction.
    center_z : int
        Center length of the box in z-direction.
    ord : int, optional
        Order of the norm. Default is np.inf.

    Returns
    -------
    ek_u_sphere : np.ndarray
    ek_v_sphere : np.ndarray
    ek_w_sphere : np.ndarray
    """
    for i in range(box_side_x):
        for j in range(box_side_y):
            for k in range(box_side_z):
                tmp = np.array(
                    [i - center_x, j - center_y, k - center_z],
                    dtype=np.float64
                )
                wn = int(np.round(np.linalg.norm(tmp, ord=ord)))

                ek_u_sphere[wn] += ek_u[i, j, k]
                ek_v_sphere[wn] += ek_v[i, j, k]
                ek_w_sphere[wn] += ek_w[i, j, k]

    return ek_u_sphere, ek_v_sphere, ek_w_sphere


def compute_scalar_energy_spectrum_numba(
    ek_u: np.ndarray, ek_v: np.ndarray = None, ek_w: np.ndarray = None,
    ord: int = 2, debug: bool = False
):
    """
    Calculate 1D energy spectrum of the flow E(k), from the point-wise energy
    spectrum E(kx, ky, kz), by integrating it over the surface of a sphere of
    radius k = (kx**2 + ky**2 + kz**2)**0.5.
    This function is a wrapper around the numba-compiled functions
    `_compute_scalar_ek_from_1d_numba_helper`,
    `_compute_scalar_ek_from_2d_numba_helper` and
    `_compute_scalar_ek_from_3d_numba_helper`.

    Parameters
    ----------
    ek_u : np.ndarray
        Point-wise energy spectrum of the flow in x-direction.
    ek_v : np.ndarray, optional
        Point-wise energy spectrum of the flow in y-direction.
    ek_w : np.ndarray, optional
        Point-wise energy spectrum of the flow in z-direction.
    ord : int, optional
        Order of the norm. Default is 2.
    debug : bool, optional
        Return the averaged energy spectrum as well. Default is False.

    Returns
    -------
    k : np.ndarray
        1D array of wave numbers.
    ek : np.ndarray
        1D array of energy spectrum.
    """
    dim, box_dimensions, ek_u_sphere, ek_v_sphere, ek_w_sphere =\
        _get_scalar_energy_spectrum_params(ek_u, ek_v, ek_w)
    box_side_x = box_dimensions['box_side_x']
    box_side_y = box_dimensions['box_side_y']
    box_side_z = box_dimensions['box_side_z']
    center_x = box_dimensions['center_x']
    center_y = box_dimensions['center_y']
    center_z = box_dimensions['center_z']

    if dim == 1:
        ek_u_sphere = _compute_ek_from_1d_numba_helper(
            ek_u=ek_u, ek_u_sphere=ek_u_sphere, box_side_x=box_side_x,
            center_x=center_x, ord=ord
        )
        ek_v_sphere = np.zeros_like(ek_u_sphere)
        ek_w_sphere = np.zeros_like(ek_u_sphere)
    elif dim == 2:
        ek_u_sphere, ek_v_sphere = _compute_ek_from_2d_numba_helper(
            ek_u=ek_u, ek_v=ek_v, ek_u_sphere=ek_u_sphere,
            ek_v_sphere=ek_v_sphere, box_side_x=box_side_x,
            box_side_y=box_side_y, center_x=center_x, center_y=center_y,
            ord=ord
        )
        ek_w_sphere = np.zeros_like(ek_u_sphere)
    elif dim == 3:
        ek_u_sphere, ek_v_sphere, ek_w_sphere =\
            _compute_ek_from_3d_numba_helper(
                ek_u=ek_u, ek_v=ek_v, ek_w=ek_w, ek_u_sphere=ek_u_sphere,
                ek_v_sphere=ek_v_sphere, ek_w_sphere=ek_w_sphere,
                box_side_x=box_side_x, box_side_y=box_side_y,
                box_side_z=box_side_z, center_x=center_x, center_y=center_y,
                center_z=center_z, ord=ord
            )
    ek = 0.5 * (ek_u_sphere + ek_v_sphere + ek_w_sphere)
    k = np.arange(0, len(ek))

    if debug:
        return k, ek, ek_u_sphere, ek_v_sphere, ek_w_sphere
    return k, ek


@elementwise
@annotate
def _compute_ek_from_1d_inf_norm_compyle_helper(
    i, ek_u, ek_u_sphere, center_x
):
    """
    Compyle helper function for computing the scalar energy spectrum from 1D
    data, using the infinity norm.
    """
    wn = declare('int')
    wn = cast(abs(i - center_x), 'int')
    ek_u_sphere[wn] += ek_u[i]


@elementwise
@annotate
def _compute_ek_from_1d_2_norm_compyle_helper(
    i, ek_u, ek_u_sphere, center_x
):
    """
    Compyle helper function for computing the scalar energy spectrum from 1D
    data, using the 2-norm.
    """
    wn = declare('int')
    tmp, frac_tmp, flr_tmp = declare('double', 3)

    tmp = cast(sqrt((i - center_x)**2), 'double')
    flr_tmp = cast(floor(tmp), 'double')
    frac_tmp = cast(tmp - flr_tmp, 'double')

    if frac_tmp < 0.5:
        wn = cast(flr_tmp, 'int')
    else:
        wn = cast(flr_tmp + 1.0, 'int')
    ek_u_sphere[wn] += ek_u[i]


@elementwise
@annotate
def _compute_ek_from_2d_inf_norm_compyle_helper(
    i, ek_u, ek_v, ek_u_sphere, ek_v_sphere, box_side_y, center_x, center_y
):
    """
    Compyle helper function for computing the scalar energy spectrum from 2D
    data, using the infinity norm.
    """
    iter_i, iter_j, wn = declare('int', 3)

    iter_i = i // box_side_y
    iter_j = i - iter_i * box_side_y

    tmp1, tmp2 = declare('int', 2)
    tmp1 = cast(abs(iter_i - center_x), 'int')
    tmp2 = cast(abs(iter_j - center_y), 'int')
    wn = cast(max(tmp1, tmp2), 'int')

    ek_u_sphere[wn] += ek_u[i]
    ek_v_sphere[wn] += ek_v[i]


@elementwise
@annotate
def _compute_ek_from_2d_2_norm_compyle_helper(
    i, ek_u, ek_v, ek_u_sphere, ek_v_sphere, box_side_y, center_x, center_y
):
    """
    Compyle helper function for computing the scalar energy spectrum from 2D
    data, using the 2-norm.
    """
    iter_i, iter_j, wn = declare('int', 3)

    iter_i = i // box_side_y
    iter_j = i - iter_i * box_side_y

    tmp, frac_tmp, flr_tmp = declare('double', 3)

    tmp1, tmp2 = declare('double', 2)
    tmp1 = cast((iter_i - center_x)**2, 'double')
    tmp2 = cast((iter_j - center_y)**2, 'double')
    tmp = cast(sqrt(tmp1 + tmp2), 'double')

    flr_tmp = cast(floor(tmp), 'double')
    frac_tmp = tmp - flr_tmp

    if frac_tmp < 0.5:
        wn = cast(flr_tmp, "int")
    else:
        wn = cast(flr_tmp + 1.0, "int")

    ek_u_sphere[wn] += ek_u[i]
    ek_v_sphere[wn] += ek_v[i]


@elementwise
@annotate
def _compute_ek_from_3d_inf_norm_compyle_helper(
    i, ek_u, ek_v, ek_w, ek_u_sphere, ek_v_sphere, ek_w_sphere,
    box_side_y, box_side_z, center_x, center_y, center_z
):
    """
    Compyle helper function for computing the scalar energy spectrum from 3D
    data, using the infinity norm.
    """
    iter_i, iter_j, iter_k, wn = declare('int', 4)

    iter_i = i // (box_side_y * box_side_z)
    iter_j = (i - iter_i * box_side_y * box_side_z) // box_side_z
    iter_k = i - iter_i * box_side_y * box_side_z - iter_j * box_side_z

    tmp1, tmp2, tmp3 = declare('int', 3)
    tmp1 = cast(abs(iter_i - center_x), 'int')
    tmp2 = cast(abs(iter_j - center_y), 'int')
    tmp3 = cast(abs(iter_k - center_z), 'int')
    wn = cast(max(tmp1, tmp2, tmp3), 'int')

    ek_u_sphere[wn] += ek_u[i]
    ek_v_sphere[wn] += ek_v[i]
    ek_w_sphere[wn] += ek_w[i]


@elementwise
@annotate
def _compute_ek_from_3d_2_norm_compyle_helper(
    i, ek_u, ek_v, ek_w, ek_u_sphere, ek_v_sphere, ek_w_sphere,
    box_side_y, box_side_z, center_x, center_y, center_z
):
    """
    Compyle helper function for computing the scalar energy spectrum from 3D
    data, using the 2-norm.
    """
    iter_i, iter_j, iter_k, wn = declare('int', 4)

    iter_i = i // (box_side_y * box_side_z)
    iter_j = (i - iter_i * box_side_y * box_side_z) // box_side_z
    iter_k = i - iter_i * box_side_y * box_side_z - iter_j * box_side_z

    tmp, frac_tmp, flr_tmp = declare('double', 3)

    tmp1, tmp2, tmp3 = declare('double', 3)
    tmp1 = cast((iter_i - center_x)**2, 'double')
    tmp2 = cast((iter_j - center_y)**2, 'double')
    tmp3 = cast((iter_k - center_z)**2, 'double')
    tmp = cast(sqrt(tmp1 + tmp2 + tmp3), 'double')

    flr_tmp = cast(floor(tmp), 'double')
    frac_tmp = tmp - flr_tmp

    if frac_tmp < 0.5:
        wn = cast(flr_tmp, "int")
    else:
        wn = cast(flr_tmp + 1.0, "int")

    ek_u_sphere[wn] += ek_u[i]
    ek_v_sphere[wn] += ek_v[i]
    ek_w_sphere[wn] += ek_w[i]


def compute_scalar_energy_spectrum_compyle(
    ek_u: np.ndarray, ek_v: np.ndarray = None, ek_w: np.ndarray = None,
    ord: int = 2, debug: bool = False
):
    """
    Calculate 1D energy spectrum of the flow E(k), from the point-wise energy
    spectrum E(kx, ky, kz), by integrating it over the surface of a sphere of
    radius k = (kx**2 + ky**2 + kz**2)**0.5.
    This function is a wrapper around the Compyle helper functions
    `_compute_ek_from_1d_inf_norm_compyle_helper`,
    `_compute_ek_from_1d_2_norm_compyle_helper`,
    `_compute_ek_from_2d_inf_norm_compyle_helper`,
    `_compute_ek_from_2d_2_norm_compyle_helper`,
    `_compute_ek_from_3d_inf_norm_compyle_helper`,
    `_compute_ek_from_3d_2_norm_compyle_helper`.

    Parameters
    ----------
    ek_u : np.ndarray
        Point-wise energy spectrum of the flow in x-direction.
    ek_v : np.ndarray, optional
        Point-wise energy spectrum of the flow in y-direction.
    ek_w : np.ndarray, optional
        Point-wise energy spectrum of the flow in z-direction.
    ord : int, optional
        Order of the norm. Default is 2.
    debug : bool, optional
        Return the averaged energy spectrum as well. Default is False.

    Returns
    -------
    k : np.ndarray
        1D array of wave numbers.
    ek : np.ndarray
        1D array of energy spectrum.
    """
    dim, box_dimensions, ek_u_sphere, ek_v_sphere, ek_w_sphere =\
        _get_scalar_energy_spectrum_params(ek_u, ek_v, ek_w)
    box_side_y = int(box_dimensions['box_side_y'])
    box_side_z = int(box_dimensions['box_side_z'])
    center_x = int(box_dimensions['center_x'])
    center_y = int(box_dimensions['center_y'])
    center_z = int(box_dimensions['center_z'])

    if ord not in [np.inf, 2]:
        raise ValueError(
            f"Order of the norm should be either np.inf or 2, not {ord}."
        )

    if dim == 1:
        with use_config(use_openmp=True):
            ek_u = wrap(ek_u.ravel())
            ek_u_sphere = wrap(ek_u_sphere.ravel())

            if ord == np.inf:
                _compute_ek_from_1d_inf_norm_compyle_helper(
                    ek_u, ek_u_sphere, center_x
                )
            elif ord == 2:
                _compute_ek_from_1d_2_norm_compyle_helper(
                    ek_u, ek_u_sphere, center_x
                )
            ek_u_sphere.pull()
            ek_u_sphere = ek_u_sphere.data
    elif dim == 2:
        with use_config(use_openmp=True):
            ek_u, ek_v = wrap(ek_u.ravel(), ek_v.ravel())
            ek_u_sphere, ek_v_sphere = wrap(
                ek_u_sphere.ravel(), ek_v_sphere.ravel()
            )

            if ord == np.inf:
                _compute_ek_from_2d_inf_norm_compyle_helper(
                    ek_u, ek_v, ek_u_sphere, ek_v_sphere, box_side_y,
                    center_x, center_y
                )
            elif ord == 2:
                _compute_ek_from_2d_2_norm_compyle_helper(
                    ek_u, ek_v, ek_u_sphere, ek_v_sphere, box_side_y,
                    center_x, center_y
                )
            ek_u_sphere.pull()
            ek_v_sphere.pull()
            ek_u_sphere, ek_v_sphere = ek_u_sphere.data, ek_v_sphere.data
    elif dim == 3:
        with use_config(use_openmp=True):
            ek_u, ek_v, ek_w = wrap(ek_u.ravel(), ek_v.ravel(), ek_w.ravel())
            ek_u_sphere, ek_v_sphere, ek_w_sphere = wrap(
                ek_u_sphere.ravel(), ek_v_sphere.ravel(), ek_w_sphere.ravel()
            )

            if ord == np.inf:
                _compute_ek_from_3d_inf_norm_compyle_helper(
                    ek_u, ek_v, ek_w, ek_u_sphere, ek_v_sphere, ek_w_sphere,
                    box_side_y, box_side_z, center_x, center_y, center_z
                )
            elif ord == 2:
                _compute_ek_from_3d_2_norm_compyle_helper(
                    ek_u, ek_v, ek_w, ek_u_sphere, ek_v_sphere, ek_w_sphere,
                    box_side_y, box_side_z, center_x, center_y, center_z
                )
            ek_u_sphere.pull()
            ek_v_sphere.pull()
            ek_w_sphere.pull()
            ek_u_sphere, ek_v_sphere, ek_w_sphere = (
                ek_u_sphere.data, ek_v_sphere.data, ek_w_sphere.data
            )

    ek = 0.5 * (ek_u_sphere + ek_v_sphere + ek_w_sphere)
    k = np.arange(0, len(ek))

    if debug:
        return k, ek, ek_u_sphere, ek_v_sphere, ek_w_sphere
    return k, ek


def velocity_intepolator(
    fname: str, dim: int, kernel: object = None, i_nx: int = 101,
    domain_manager: object = None, **kwargs
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
        Kernel object. Default is WendlandQuinticC4.
    i_nx : int, optional
        Number of points to interpolate the energy spectrum (i_nx**2 for
        2D data, i_nx**3 for 3D data). Default is 101.
    domain_manager : object, optional
        DomainManager object. Default is None.
    **kwargs : dict, optional
        Additional keyword arguments to pass to the interpolator.

    Returns
    -------
    t : float
        Time of the flow data.
    res : list[np.ndarray]
        List of interpolated energy spectrum of the flow for each direction.
    """
    # Load data
    data = load(fname)
    t = data["solver_data"]["t"]

    # Create meshgrid based on dimension
    _x = np.linspace(0, 1, i_nx)
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
        res = [ui, None, None]
    elif dim == 2:
        _u = interp.interpolate('u')
        _v = interp.interpolate('v')
        ui = _u.reshape(i_nx, i_nx)
        vi = _v.reshape(i_nx, i_nx)
        res = [ui, vi, None]
    elif dim == 3:
        _u = interp.interpolate('u')
        _v = interp.interpolate('v')
        _w = interp.interpolate('w')
        ui = _u.reshape(i_nx, i_nx, i_nx)
        vi = _v.reshape(i_nx, i_nx, i_nx)
        wi = _w.reshape(i_nx, i_nx, i_nx)
        res = [ui, vi, wi]
    return t, res


class EnergySpectrum(object):
    """
    Class to compute the energy spectrum of the flow.
    Note: len(shape(u)) == dim

    Initialization
    --------------
    1. From velocity field
        >>> EnergySpectrum(dim, u, v, w....)
    2. From PySPH file (Interpolates the velocity field implicitly)
        >>> EnergySpectrum.from_pysph_file(fname....)
    3. From an example
        >>> EnergySpectrum.from_example(dim....)

    Parameters
    ----------
    dim : int
        Dimension of the flow.
    L : float
        Length of the box.
    dx : float
        Grid spacing.
    u : np.ndarray
        Velocity field in x-direction.
    v : np.ndarray
        Velocity field in y-direction. None for 1D data.
    w : np.ndarray
        Velocity field in z-direction. None for 1D and 2D data.
    t : float, optional
        Time of the flow data. Default is 0. Optional, required for plotting.
    U0: float, optional
        Reference velocity. Default is 1.

    Class Methods
    -------------
    1. Initialize from PySPH file
        >>> EnergySpectrum.from_pysph_file(fname....)
    2. Initialize from an example
        >>> EnergySpectrum.from_example(dim....)

    Instance Methods
    ----------------
    1. Compute energy spectrum
        >>> EnergySpectrum.compute()
    2. Plot scalar energy spectrum
        >>> EnergySpectrum.plot_scalar_ek()
    3. Plot vector energy spectrum
        >>> EnergySpectrum.plot_vector_ek()
    """

    def __init__(
        self, dim: int, L: float, dx: float, u: np.ndarray,
        v: np.ndarray = None, w: np.ndarray = None, t: float = 0.,
        U0: float = 1.
    ):
        """
        Initialize the class.
        """
        self.dim = dim
        self.L = L
        self.dx = dx
        self.u, self.v, self.w = u, v, w
        self.t = t
        self.U0 = U0

        self.n_1d = u.shape[0]

        if dim not in [1, 2, 3]:
            raise ValueError("Dimension should be 1, 2 or 3.")

        self._check_format_of_list_data([u, v, w])

        self.k, self.ek = None, None

    # Class methods
    @classmethod
    def from_pysph_file(
        cls, fname: str, dim: int, L: float, i_nx: int,
        kernel: object = None, radius_scale: float = None,
        domain_manager: object = None, U0=1., debug=False, **kwargs
    ):
        """
        Create an EnergySpectrum object from a PySPH output file by
        interpolating the velocity field.

        Parameters
        ----------
        fname : str
            Name of the file containing the flow data.
        dim : int
            Dimension of the flow.
        L : float
            Length of the domain.
        i_nx : int, optional
            Number of points to interpolate the energy spectrum (i_nx**2 for 2D
            data, i_nx**3 for 3D data).
        kernel : object, optional
            Kernel object. Default is WendlandQuinticC4.
        radius_scale : float, optional
            Radius scale for the kernel. Default is None.
        domain_manager : object, optional
            DomainManager object. Default is None.
        U0 : float, optional
            Reference velocity. Default is 1.
        debug : bool, optional
            If True, returns the Interpolator object. Default is False.
        **kwargs : dict, optional
            Additional keyword arguments to pass to the PySPH interpolator.

        Returns
        -------
        EnergySpectrum object.
        """
        data = load(fname)
        t = data["solver_data"]["t"]
        u = data["arrays"]["fluid"].get("u")

        if i_nx is None:
            i_nx = int(np.power(len(u), 1 / dim))

        # Create meshgrid based on dimension
        _x = np.linspace(0, L, i_nx)
        dx = _x[1] - _x[0]
        if dim == 1:
            x = _x
            y = z = None
        elif dim == 2:
            x, y = np.meshgrid(_x, _x)
            z = None
        elif dim == 3:
            x, y, z = np.meshgrid(_x, _x, _x)

        # Setup default interpolator properties
        if kernel is None:
            kernel = WendlandQuinticC4(dim=dim)

        # Check if radius_scale is instance of float
        if isinstance(radius_scale, float):
            kernel.radius_scale = radius_scale

        # Interpolate velocity
        interp_ob = Interpolator(
            list(data['arrays'].values()), x=x, y=y, z=z,
            kernel=kernel, domain_manager=domain_manager, **kwargs
        )
        if dim == 1:
            _u = interp_ob.interpolate('u')
            ui = _u
            vi = wi = None
        elif dim == 2:
            _u = interp_ob.interpolate('u')
            _v = interp_ob.interpolate('v')
            ui = _u.reshape(i_nx, i_nx)
            vi = _v.reshape(i_nx, i_nx)
            wi = None
        elif dim == 3:
            _u = interp_ob.interpolate('u')
            _v = interp_ob.interpolate('v')
            _w = interp_ob.interpolate('w')
            ui = _u.reshape(i_nx, i_nx, i_nx)
            vi = _v.reshape(i_nx, i_nx, i_nx)
            wi = _w.reshape(i_nx, i_nx, i_nx)

        cls_ob = cls(dim=dim, L=L, dx=dx, u=ui, v=vi, w=wi, t=t, U0=U0)
        if debug:
            return cls_ob, interp_ob
        else:
            return cls_ob

    @classmethod
    def from_example(
        cls, dim: int, nx: int, custom_formula: list = None
    ):
        """
        Create an EnergySpectrum object from an example.

        Parameters
        ----------
        dim : int
            Dimension of the flow.
        nx : int
            Number of points in each direction.
        custom_formula : list, optional
            Custom formula to generate the velocity field for each dimension.
            Default is None.
            Numpy functions can be used. Spatial coordinates are x, y and z.
            pi, twopi, cos and sin can be used as well.

        Returns
        -------
        EnergySpectrum object.

        Notes
        -----
        dim = 1:
            x = arange(0, 1, 1/nx)
            u = - cos(2πx)
        dim = 2:
            _x = arange(0, 1, 1/nx)
            x, y = meshgrid(_x, _x)
            u = + cos(2πx) * sin(2πy)
            v = - sin(2πx) * cos(2πy)
        dim = 3:
            _x = arange(0, 1, 1/nx)
            x, y, z = meshgrid(_x, _x, _x)
            u = + sin(2πx) * cos(2πy) * cos(2πz)
            v = - cos(2πx) * sin(2πy) * cos(2πz)
            w = 0.0
        """
        pi = np.pi
        twopi = 2 * pi
        cos, sin = np.cos, np.sin

        L = 1.
        U0 = 1.

        _x = np.arange(0., L, 1. / nx)
        dx = _x[1] - _x[0]
        if dim == 1:
            if custom_formula is None:
                x = _x
                u = - cos(twopi * x)
            else:
                u = eval(custom_formula[0])
            v = w = None
        elif dim == 2:
            x, y = np.meshgrid(_x, _x)
            if custom_formula is None:
                u = + cos(twopi * x) * sin(twopi * y)
                v = - sin(twopi * x) * cos(twopi * y)
            else:
                u = eval(custom_formula[0])
                v = eval(custom_formula[1])
            w = None
        elif dim == 3:
            x, y, z = np.meshgrid(_x, _x, _x)
            if custom_formula is None:
                u = + sin(twopi * x) * cos(twopi * y) * cos(twopi * z)
                v = - cos(twopi * x) * sin(twopi * y) * cos(twopi * z)
                w = np.zeros_like(u)
            else:
                u = eval(custom_formula[0])
                v = eval(custom_formula[1])
                w = eval(custom_formula[2])
        else:
            raise ValueError("Dimension should be 1, 2 or 3.")

        return cls(
            dim=dim, L=L, dx=dx, u=u, v=v, w=w, t=0.0, U0=U0
        )

    # Static methods
    @staticmethod
    def plot_from_npz_file(
        fnames: list, figname: str = None, styles: IT.cycle = styles
    ):
        """
        Plot energy spectrum from npz files.

        Parameters
        ----------
        fnames : list
            List of npz files.
        figname : str, optional
            Name of the figure file. Default is "./energy_spectrum.png".
        styles : itertools.cycle, optional
            Styles to use for plotting. Default is styles.
        """
        plt.clf()
        plt.figure()
        ls = styles(None)
        for fname in fnames:
            data = np.load(fname)
            k = data["k"]
            ek = data["ek"]
            t = data["t"]
            plt.loglog(k, ek, label=f"t = {t:.2f}", **next(ls))

        plt.xlabel(r"$k$")
        plt.ylabel(r"$E(k)$")
        plt.legend()
        plt.title("Energy spectrum evolution")
        if figname is None:
            figname = "./energy_spectrum.png"
        plt.savefig(figname, dpi=300, bbox_inches="tight")
    
    @staticmethod
    def calculate_wavenumber_of_dx(L, dx, box_radius):
        if dx > L:
            raise ValueError("dx is larger than the box size.")
        tmp_linspace = np.linspace(0., L, box_radius)
        diff = tmp_linspace[1] - tmp_linspace[0]
        wn = np.ceil(np.abs(dx/diff))
        wn = int(box_radius - wn + 1)
        return wn

    # Private methods
    def _check_format_of_list_data(self, data: list):
        """
        Check the format of the list data.
        For 1D data, the list should be of the form [u, None, None].
        For 2D data, the list should be of the form [u, v, None].
        For 3D data, the list should be of the form [u, v, w].

        Parameters
        ----------
        data : list
            List of data to check.
        """
        if len(data) != 3:
            raise ValueError("The data should be a list of 3 arrays.")

        if self.dim == 1:
            if data[1] is not None or data[2] is not None:
                raise ValueError(
                    "The data should be a list of 1 array for dim = 1."
                )
            if data[0] is None:
                raise ValueError(f"{data[0]} is None.")
        elif self.dim == 2:
            if data[2] is not None:
                raise ValueError(
                    "The data should be a list of 2 arrays for dim = 2."
                )
            if data[0] is None or data[1] is None:
                raise ValueError(f"{data[0]} or {data[1]} is None.")
        elif self.dim == 3:
            if data[0] is None or data[1] is None or data[2] is None:
                raise ValueError(
                    f"{data[0]} or {data[1]} or {data[2]} is None."
                )

    def _correct_format_of_list_data(self, data: list):
        """
        Correct the format of the list data.

        Parameters
        ----------
        data : list
            List of data to correct.

        Returns
        -------
        Corrected list of data.
        """
        if self.dim == 1:
            corrected_data = [data[0], None, None]
        elif self.dim == 2:
            corrected_data = [data[0], data[1], None]
        elif self.dim == 3:
            corrected_data = data

        self._check_format_of_list_data(corrected_data)
        return corrected_data

    def _compute_energy_spectrum(self):
        """
        Compute the energy spectrum of the flow.
        """
        ek_u, ek_v, ek_w, u_spectrum, v_spectrum, w_spectrum =\
            compute_energy_spectrum(
                u=self.u,
                v=self.v,
                w=self.w,
                U0=self.U0,
                debug=True
            )

        return ek_u, ek_v, ek_w, u_spectrum, v_spectrum, w_spectrum

    def _compute_scalar_energy_spectrum(self, order: int, func_config: str):
        """
        Compute the energy spectrum of the flow.

        Parameters
        ----------
        order : int
            Order of the norm.
        func_config : str
            Configuration of the function to use for computing the scalar
            energy spectrum.
        """
        if func_config == 'python':
            k, ek, ek_u_sphere, ek_v_sphere, ek_w_sphere =\
                compute_scalar_energy_spectrum_python(
                    ek_u=self.ek_u,
                    ek_v=self.ek_v,
                    ek_w=self.ek_w,
                    ord=order,
                    debug=True
                )
        elif func_config == 'numba':
            k, ek, ek_u_sphere, ek_v_sphere, ek_w_sphere =\
                compute_scalar_energy_spectrum_numba(
                    ek_u=self.ek_u,
                    ek_v=self.ek_v,
                    ek_w=self.ek_w,
                    ord=order,
                    debug=True
                )
        elif func_config == 'compyle':
            k, ek, ek_u_sphere, ek_v_sphere, ek_w_sphere =\
                compute_scalar_energy_spectrum_compyle(
                    ek_u=self.ek_u,
                    ek_v=self.ek_v,
                    ek_w=self.ek_w,
                    ord=order,
                    debug=True
                )
        return k, ek, ek_u_sphere, ek_v_sphere, ek_w_sphere

    # Public methods
    def compute(self, order: int = np.inf, func_config: str = 'compyle'):
        """
        Compute the energy spectrum of the flow.

        Parameters
        ----------
        order : int, optional
            Order of the norm. Default is np.inf.
        func_config : str, optional
            Configuration of the function. Default is 'compyle'.
            Options: python, numba, 'compyle'

        """
        # Compute energy spectrum
        res = self._compute_energy_spectrum()
        self.ek_u, self.ek_v, self.ek_w, self.u_spectrum, self.v_spectrum,\
            self.w_spectrum = res

        self.ek_u, self.ek_v, self.ek_w = self._correct_format_of_list_data(
            [self.ek_u, self.ek_v, self.ek_w]
        )
        self.u_spectrum, self.v_spectrum, self.w_spectrum =\
            self._correct_format_of_list_data(
                [self.u_spectrum, self.v_spectrum, self.w_spectrum]
            )

        # Compute scalar energy spectrum
        FUNC_CONFIG = ['python', 'numba', 'compyle']
        if func_config not in FUNC_CONFIG:
            raise ValueError(
                f"{func_config} is not a valid function configuration. "
                f"Valid configurations are {FUNC_CONFIG}."
            )
        res = self._compute_scalar_energy_spectrum(
            order=order, func_config=func_config
        )
        self.k, self.ek, self.ek_u_sphere, self.ek_v_sphere, self.ek_w_sphere \
            = res
        self.ek_u_sphere, self.ek_v_sphere, self.ek_w_sphere =\
            self._correct_format_of_list_data(
                [self.ek_u_sphere, self.ek_v_sphere, self.ek_w_sphere]
            )

    def plot_scalar_ek(
        self, show=False, savefig=False, fname=None, dpi=300, plot_type="log",
    ):
        """
        Plot the scalar energy spectrum of the flow.

        Parameters
        ----------
        show : bool, optional
            Show the plot. Default is False.
        savefig : bool, optional
            Save the figure. Default is False.
        fname : str, optional
            Filename to save the figure. Default is "./energy_spectrum.png".
        dpi : int, optional
            Dots per inch. Default is 300.
        plot_type : str, optional
            Type of plot. Default is "log". Options: "log", "stem".
        """
        if self.k is None:
            self.compute()

        if fname is None:
            fname = "./energy_spectrum.png"

        n = self.n_1d
        plt.clf()
        if plot_type == "log":
            plt.loglog(self.k[0:n], self.ek[0:n], 'k')
            plt.loglog(self.k[n:], self.ek[n:], 'k--')
        elif plot_type == "stem":
            plt.stem(self.ek)
        plt.xlabel(r'$k$')
        plt.ylabel(r'$E(k)$')
        plt.grid()
        plt.tight_layout()
        plt.title(f"Energy spectrum at t = {self.t:.2f}")

        if savefig:
            plt.savefig(fname, dpi=dpi)
        if show:
            plt.show()

    def plot_vector_ek(
        self, show=False, savefig=False, fname=None, dpi=300, shift_fft=False
    ):
        """
        Plot each component of the energy spectrum of the flow.

        Parameters
        ----------
        show : bool, optional
            Show the plot. Default is False.
        savefig : bool, optional
            Save the figure. Default is False.
        fname : str, optional
            Filename to save the figure.
            Default is "./EK_spectrum.png".
        dpi : int, optional
            Dots per inch. Default is 300.
        shift_fft : bool, optional
            Shift the FFT. Default is False.
        """

        if self.k is None:
            self.compute()

        dim = self.dim
        if fname is None:
            fname = "./EK_spectrum.png"

        if shift_fft:
            fftshift = np.fft.fftshift
        else:
            def fftshift(x):
                """
                Do nothing.
                """
                return x

        if dim == 1:
            plt.clf()
            plt.stem(fftshift(self.ek_u))
            plt.xlabel(r'$k$')
            plt.ylabel(r'$E_{u}(k)$')
            plt.grid()
            plt.tight_layout()
            plt.title(r"$E_u(k)$ at t = {:.2f}".format(self.t))
            if savefig:
                plt.savefig(fname, dpi=dpi)
            if show:
                plt.show()

        elif dim == 2:
            plt.clf()

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(fftshift(self.ek_u))
            axes[0].set_title(r"$E_u(k)$ at t = {:.2f}".format(self.t))
            axes[0].set_xlabel(r"$k_x$")
            axes[0].set_ylabel(r"$k_y$")
            axes[0].invert_yaxis()

            axes[1].imshow(fftshift(self.ek_v))
            axes[1].set_title(r"$E_v(k)$ at t = {:.2f}".format(self.t))
            axes[1].set_xlabel(r"$k_x$")
            axes[1].set_ylabel(r"$k_y$")
            axes[1].invert_yaxis()

            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(axes[0].images[0], cax=cbar_ax)

            if savefig:
                plt.savefig(fname, dpi=dpi)
            if show:
                plt.show()

        elif dim == 3:
            import warnings
            warnings.warn("The feature is not implemented yet.")


if __name__ == "__main__":
    test_espec_obj = EnergySpectrum.from_example(dim=2, nx=15)
    test_espec_obj.compute(order=2, func_config='compyle')
