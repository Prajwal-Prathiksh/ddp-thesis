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


def curl_numerical(u, v, w, dx):
    """
    Returns the numerical curl of a vector field.
    """
    def _dim(u):
        return len(np.shape(u))
    if _dim(u) != _dim(v):
        raise ValueError('u and v must have the same dimensionality')
    if _dim(u) == 2:
        du_dy, du_dx = np.gradient(u, dx, edge_order=2)
        dv_dy, dv_dx = np.gradient(v, dx, edge_order=2)
        curl_z = dv_dx - du_dy
        return curl_z
    elif _dim(u) == 3:
        du_dy, du_dx, du_dz = np.gradient(u, dx, edge_order=2)
        dv_dy, dv_dx, dv_dz = np.gradient(v, dx, edge_order=2)
        dw_dy, dw_dx, dw_dz = np.gradient(w, dx, edge_order=2)
        curl_x = dw_dy - dv_dz
        curl_y = du_dz - dw_dx
        curl_z = dv_dx - du_dy
        return curl_x, curl_y, curl_z
    else:
        raise ValueError('Invalid dimensionality')
