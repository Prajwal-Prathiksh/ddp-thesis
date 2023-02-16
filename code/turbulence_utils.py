r"""
Utilitary functions required for turbulent flow simulations and analysis.
Author: K T Prajwal Prathiksh
###
"""
# Library imports
import numpy as np
from pysph.base.kernels import (
    CubicSpline, WendlandQuinticC2_1D, WendlandQuintic, WendlandQuinticC4_1D,
    WendlandQuinticC4, WendlandQuinticC6_1D, WendlandQuinticC6,
    Gaussian, SuperGaussian, QuinticSpline
)

# Kernel choices
KERNEL_CHOICES = [
    'CubicSpline', 'WendlandQuinticC2', 'WendlandQuinticC4',
    'WendlandQuinticC6', 'Gaussian', 'SuperGaussian', 'QuinticSpline'
]

# Interpolating method choices
INTERPOLATING_METHOD_CHOICES = [
    'sph', 'shepard', 'order1', 'order1BL', 'order1MC'
]


def get_kernel_cls(name: str, dim: int):
    """
        Return the kernel class corresponding to the name initialized with the
        dimension.

        Parameters
        ----------
        name : str
            Name of the kernel class.
        dim : int
            Dimension of the kernel.

        Returns
        -------
        kernel_cls : class
            Kernel class (dim).
    """
    if dim not in [1, 2, 3]:
        raise ValueError("Dimension must be 1, 2 or 3.")
    mapper = {
        'CubicSpline': CubicSpline,
        'WendlandQuinticC2': WendlandQuinticC2_1D if dim == 1
        else WendlandQuintic,
        'WendlandQuinticC4': WendlandQuinticC4_1D if dim == 1
        else WendlandQuinticC4,
        'WendlandQuinticC6': WendlandQuinticC6_1D if dim == 1
        else WendlandQuinticC6,
        'Gaussian': Gaussian,
        'SuperGaussian': SuperGaussian,
        'QuinticSpline': QuinticSpline
    }

    name = name.lower().strip()
    not_found = True
    for key in mapper.keys():
        if key.lower() == name:
            name = key
            not_found = False
            break

    if not_found:
        raise ValueError(
            f"Kernel {name} not supported. Valid choices are {KERNEL_CHOICES}"
        )

    return mapper[name](dim=dim)


def compute_curl(
    dx: float, u: np.ndarray, v: np.ndarray, w: np.ndarray = None,
    edge_order: int = 2
):
    """
    Compute the curl of a vector field (2D or 3D), assuming a uniform grid,
    of spacing `dx`.
    The function computes the curl of the vector field (u, v, w) using
    `np.gradient`.
    
    Parameters
    ----------
    dx : float
        Spacing between grid points.
    u : np.ndarray
        x-component of the vector field.
    v : np.ndarray
        y-component of the vector field.
    w : np.ndarray, optional
        z-component of the vector field. Default is None.
    edge_order : int, optional
        Order of the edge gradient. Default is 2.
    
    Returns
    -------
    curl_z (2D) or curl_x, curl_y, curl_z (3D) : np.ndarray
        The curl of the vector field.
    """
    def _dim(u):
        """
        Return the dimensionality of the array `u`.
        """
        return len(np.shape(u))
    
    def _check_shape(*args):
        """
        Check if the arrays have the same shape.
        """
        shape = np.shape(args[0])
        for arg in args:
            if np.shape(arg) != shape:
                raise ValueError('All arrays must have the same shape')
    
    if _dim(u) == 2:
        _check_shape(u, v)
        du_dy, du_dx = np.gradient(u, dx, edge_order=edge_order)
        dv_dy, dv_dx = np.gradient(v, dx, edge_order=edge_order)

        curl_z = dv_dx - du_dy

        return curl_z

    elif _dim(u) == 3:
        _check_shape(u, v, w)
        du_dy, du_dx, du_dz = np.gradient(u, dx, edge_order=edge_order)
        dv_dy, dv_dx, dv_dz = np.gradient(v, dx, edge_order=edge_order)
        dw_dy, dw_dx, dw_dz = np.gradient(w, dx, edge_order=edge_order)

        curl_x = dw_dy - dv_dz
        curl_y = du_dz - dw_dx
        curl_z = dv_dx - du_dy
        
        return curl_x, curl_y, curl_z

    else:
        msg = f"Invalid dimensionality: {_dim(u)}. Must be 2 or 3."
        raise ValueError(msg)
