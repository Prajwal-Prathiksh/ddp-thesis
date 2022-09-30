'''
Boundary condition by
H. Takeda, S. M. Miyama, and M. Sekiya, “Numerical
Simulation of Viscous Flow by Smoothed Particle Hydrodynamics,” Progress of
Theoretical Physics, vol. 92, no. 5, pp. 939–960, Nov. 1994, doi:
10.1143/ptp/92.5.939.
'''


from numpy.core.numeric import ones_like
from pysph.base.utils import get_particle_array
from pysph.sph.equation import Equation
from pysph.sph.wc.edac import SourceNumberDensity
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



class ComputeNumberDensity(SourceNumberDensity):
    pass


class SelfNumberDensity(Equation):
    r"""Evaluates the number density due to the source particles"""
    def initialize(self, d_idx, d_swij):
        d_swij[d_idx] = 0.0

    def loop(self, d_idx, d_swij, WIJ, d_wij, s_wij, s_idx):
        if d_wij[d_idx] < 1e-14:
            if s_wij[s_idx] > 1e-14:
                d_swij[d_idx] += WIJ


class EvaluateVelocity(Equation):
    def initialize(self, d_ug, d_vg, d_wg, d_idx, d_wij):
        if d_wij[d_idx] < 1e-14:
            d_ug[d_idx] = 0.0
            d_vg[d_idx] = 0.0
            d_wg[d_idx] = 0.0

    def loop(self, d_idx, WIJ, d_ug, d_vg, d_wg, s_ug, s_vg, s_wg, d_wij, s_wij, s_idx):
        if d_wij[d_idx] < 1e-14:
            if s_wij[s_idx] > 1e-14:
                d_ug[d_idx] += s_ug[s_idx] * WIJ
                d_vg[d_idx] += s_vg[s_idx] * WIJ
                d_wg[d_idx] += s_wg[s_idx] * WIJ

    def post_loop(self, d_idx, d_ug, d_vg, d_wg, d_swij, d_wij):
        if d_wij[d_idx] < 1e-14:
            if d_swij[d_idx] > 1e-14:
                d_ug[d_idx] /= d_swij[d_idx]
                d_vg[d_idx] /= d_swij[d_idx]
                d_wg[d_idx] /= d_swij[d_idx]


class FindFluidNeighbors(Equation):
    ''' For every ghost particle find the nearest fluid particle in the normal
    direction'''
    def initialize(self, d_idx, d_xf, d_yf, d_zf, d_uf, d_vf, d_wf):
        d_xf[d_idx] = 0.0
        d_yf[d_idx] = 0.0
        d_zf[d_idx] = 0.0
        d_uf[d_idx] = 0.0
        d_vf[d_idx] = 0.0
        d_wf[d_idx] = 0.0

    def loop_all(self, d_idx, d_x, d_y, d_z, s_x, s_y, s_z, s_normal,
                 NBRS, N_NBRS, d_xf, d_yf, d_zf, s_u, s_v, s_w, d_uf, d_vf, d_wf):
        s_idx = declare('int')
        dmin = 10000
        dproj = 10000

        for i in range(N_NBRS):
            s_idx = NBRS[i]
            xij = d_x[d_idx] - s_x[s_idx]
            yij = d_y[d_idx] - s_y[s_idx]
            zij = d_z[d_idx] - s_z[s_idx]
            rij = sqrt(xij**2 + yij**2 + zij**2)
            if rij < dmin:
                proj_d = xij * s_normal[3 * s_idx] + yij * s_normal[
                    3 * s_idx + 1] + zij * s_normal[3 * s_idx + 2]
                if proj_d - dproj < 1e-8:
                    dmin = rij
                    dproj = proj_d
                    d_xf[d_idx] = s_x[s_idx]
                    d_yf[d_idx] = s_y[s_idx]
                    d_zf[d_idx] = s_z[s_idx]
                    d_uf[d_idx] = s_u[s_idx]
                    d_vf[d_idx] = s_v[s_idx]
                    d_wf[d_idx] = s_w[s_idx]


class FindBoundaryNeighbors(Equation):
    ''' For every ghost particle find the nearest boundary particle in the normal
    direction'''
    def initialize(self, d_idx, d_xb, d_yb, d_zb):
        d_xb[d_idx] = 0.0
        d_yb[d_idx] = 0.0
        d_zb[d_idx] = 0.0

    def loop_all(self, d_idx, d_x, d_y, d_z, s_x, s_y, s_z, s_normal,
                 NBRS, N_NBRS, d_xb, d_yb, d_zb):
        s_idx = declare('int')
        dmin = 10000
        dproj = 10000

        for i in range(N_NBRS):
            s_idx = NBRS[i]
            xij = d_x[d_idx] - s_x[s_idx]
            yij = d_y[d_idx] - s_y[s_idx]
            zij = d_z[d_idx] - s_z[s_idx]
            rij = sqrt(xij**2 + yij**2 + zij**2)
            if rij < dmin:
                proj_d = xij * s_normal[3 * s_idx] + yij * s_normal[
                    3 * s_idx + 1] + zij * s_normal[3 * s_idx + 2]
                if proj_d - dproj < 1e-8:
                    dmin = rij
                    dproj = proj_d
                    d_xb[d_idx] = s_x[s_idx]
                    d_yb[d_idx] = s_y[s_idx]
                    d_zb[d_idx] = s_z[s_idx]


class TakedaNoSlipBoundary(Equation):
    def __init__(self, dest, sources, U=0.0, V=0.0, W=0.0):
        self.U = U
        self.V = V
        self.W = W
        super(TakedaNoSlipBoundary, self).__init__(dest, sources)

    def initialize(self, d_ug, d_vg, d_wg, d_idx):
        d_ug[d_idx] = self.U
        d_vg[d_idx] = self.V
        d_wg[d_idx] = self.W

    #Cannot perform in initialize as the values required from other two equations#
    def post_loop(self, d_idx, d_ug, d_vg, d_wg, d_xb, d_yb, d_zb,
                   d_xf, d_yf, d_zf, d_x, d_y, d_z, d_uf, d_vf, d_wf):
        r0 = sqrt((d_xb[d_idx] - d_xf[d_idx])**2 +
                  (d_yb[d_idx] - d_yf[d_idx])**2 +
                  (d_zb[d_idx] - d_zf[d_idx])**2)
        r1 = sqrt((d_x[d_idx] - d_xf[d_idx])**2 +
                  (d_y[d_idx] - d_yf[d_idx])**2 +
                  (d_z[d_idx] - d_zf[d_idx])**2)
        proj = ((d_xb[d_idx] - d_xf[d_idx]) * (d_x[d_idx] - d_xf[d_idx]) +
                (d_yb[d_idx] - d_yf[d_idx]) * (d_y[d_idx] - d_yf[d_idx]) +
                (d_zb[d_idx] - d_zf[d_idx]) * (d_z[d_idx] - d_zf[d_idx]))
        if r0 > 1e-14:
            proj = proj / (r0 * r1)
            d_ug[d_idx] = (d_uf[d_idx] - self.U) * (r1-proj)/proj + self.U
            d_vg[d_idx] = (d_vf[d_idx] - self.V) * (r1-proj)/proj + self.V
            d_wg[d_idx] = (d_wf[d_idx] - self.W) * (r1-proj)/proj + self.W


def solid_bc(bcs, fluids, rho0, p0):
    import sys
    print(bcs)
    g0 = []
    g1 = []
    for bc in bcs:
        if bc == 'u_no_slip':
            for name in bcs[bc]:
                g0.extend([
                    ComputeNumberDensity(dest='solid0', sources=['fluid']),
                    SelfNumberDensity(dest='solid0', sources=['solid0']),
                    FindFluidNeighbors(dest='solid0', sources=['fluid']),
                    FindBoundaryNeighbors(dest='solid0', sources=['boundary']),
                    TakedaNoSlipBoundary(dest='solid0', sources=None)
                ])
                g1.append(
                    EvaluateVelocity(dest='solid0', sources=['solid0'])
                )
        if bc == 'u_slip':
            print("slip bc doesn't exist")
            sys.exit(0)
        if bc == 'p_solid':
            print("pressure bc doesn't exist")
            sys.exit(0)
    return [g0, g1]