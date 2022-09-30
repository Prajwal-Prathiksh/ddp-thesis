''' M. Ferrand, D. R. Laurence, B. D. Rogers, D. Violeau, and C. Kassiotis,
“Unified semi-analytical wall boundary conditions for inviscid, laminar or
turbulent flows in the meshless SPH method: UNIFIED WALL BOUNDARY
CONDITIONS IN SPH,” Int. J. Numer. Meth. Fluids, vol. 71, no. 4, pp.
446–472, Feb. 2013, doi: 10.1002/fld.3666.

The analytical formula of the gradient of gamma is non trivial.
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
    return ['g', {'name':'gradg', 'stride':3}]


def requires():
    is_mirror = False
    is_boundary = True
    is_ghost = False
    is_ghost_mirror = False
    is_boundary_shift = False

    return is_mirror, is_boundary, is_ghost, is_ghost_mirror, is_boundary_shift


def get_bc_names():
    return ['boundary', 'solid1']


class FindNearBoundaryParticlse(Equation):
    def initialize(self, d_idx, d_bid, d_proj):
        d_bid[d_idx] = -1
        d_proj[d_idx] = 10000

    def loop_all(self, d_idx, d_bid, d_x, d_y, d_z, s_x, s_y, s_z, s_normal,
                 NBRS, N_NBRS, d_proj, s_cid):
        s_idx = declare('int')
        dmin = 10000

        for i in range(N_NBRS):
            s_idx = NBRS[i]
            xij = d_x[d_idx] - s_x[s_idx]
            yij = d_y[d_idx] - s_y[s_idx]
            zij = d_z[d_idx] - s_z[s_idx]
            rij = sqrt(xij**2 + yij**2 + zij**2)
            if s_cid[s_idx] < 0:
                if rij < dmin:
                    proj_d = xij * s_normal[3 * s_idx] + yij * s_normal[
                        3 * s_idx + 1] + zij * s_normal[3 * s_idx + 2]
                    if proj_d < d_proj[d_idx]:
                        dmin = rij
                        d_proj[d_idx] = proj_d
                        d_bid[d_idx] = s_idx


class EvaluateGammaGradient(Equation):
    def initialize(self, d_idx, d_gradg):
        i = declare('int')
        for i in range(3):
            d_gradg[3 * d_idx + i] = 0.0

