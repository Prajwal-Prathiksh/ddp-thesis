''' F. Maciá, M. Antuono, L. M. González, and A. Colagrossi,“Theoretical
Analysis of the No-Slip Boundary Condition Enforcement in SPH Methods,”
Progress of Theoretical Physics, vol. 125, no. 6, pp. 1091–1121, Jun. 2011,
doi: 10.1143/PTP.125.1091.

The normalization in this paper does not seem correct. Both the numerator
and denominator is approximation of same velocity profile. Dividing them
will not provide a laplacian of a arbitrary field.
'''


from pysph.sph.equation import Equation
from compyle.api import declare
import numpy as np


def boundary_props():
    '''
    bid: Boundary particle closest to fluid
    cid: greater than zero if corner particle
    cbid: Boundary particle closed to fluid particle
    '''
    return ['xb', 'yb', 'zb', 'xf', 'yf', 'zf', 'wij', 'swij']


def get_bc_names():
    return ['solid0', 'solid1']


def requires():
    is_mirror = False
    is_boundary = True
    is_ghost = True
    is_ghost_mirror = False
    is_boundary_shift = False

    return is_mirror, is_boundary, is_ghost, is_ghost_mirror, is_boundary_shift