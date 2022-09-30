''' [1]M. R. Hashemi, R. Fatehi, and M. T. Manzari, “A modified SPH method
for simulating motion of rigid bodies in Newtonian fluid flows,”
International Journal of Non-Linear Mechanics, vol. 47, no. 6, pp. 626–638,
Jul. 2012, doi: 10.1016/j.ijnonlinmec.2011.10.007.
'''

from pysph.sph.equation import Equation
from compyle.api import declare
import numpy as np


def boundary_props():
    '''
    '''
    return ['dwij', {'name':'m_mat', 'stride':16}]


def get_bc_names():
    return ['boundary', 'solid1']


def requires():
    is_mirror = False
    is_boundary = True
    is_ghost = True
    is_ghost_mirror = False
    is_boundary_shift = False

    return is_mirror, is_boundary, is_ghost, is_ghost_mirror, is_boundary_shift


class PressureWallHashemi(Equation):
    def initialize(self, d_idx, d_p, d_dwij):
        d_p[d_idx] = 0.0
        d_dwij[d_idx] = 0.0

    def loop(self, d_idx, d_au, d_av, d_aw, s_p, s_m, s_rho, s_idx, d_p,
             d_dwij, DWIJ, d_normal, d_rhoc):

        tmp = s_m[s_idx] / s_rho[s_idx] /d_rhoc[d_idx]
        pval = s_p[s_idx] * (
            (tmp * DWIJ[0]) * d_normal[3 * d_idx] +
            (tmp * DWIJ[1]) * d_normal[3 * d_idx + 1] +
            (tmp * DWIJ[2]) * d_normal[3 * d_idx + 2]
        )

        d_p[d_idx] += pval
        d_dwij[d_idx] += (
            DWIJ[0] * tmp * d_normal[3 * d_idx] +
            DWIJ[1] * tmp * d_normal[3 * d_idx + 1] +
            DWIJ[2] * tmp * d_normal[3 * d_idx + 2])

    def post_loop(self, d_idx, d_p, d_dwij, d_au, d_av, d_aw, d_normal):
        proj_au = d_au[d_idx] * d_normal[3 * d_idx] + d_av[d_idx] * d_normal[3 * d_idx + 1] + d_aw[d_idx] * d_normal[3 * d_idx + 2]
        if abs(d_dwij[d_idx]) > 1e-14:
            d_p[d_idx] = (d_p[d_idx] + proj_au) / d_dwij[d_idx]


class NoSlipWallHashemi(Equation):
    def initialize(self, d_idx, d_ug, d_vg, d_wg):
        d_ug[d_idx] = 0.0
        d_vg[d_idx] = 0.0
        d_wg[d_idx] = 0.0


def solid_bc(bcs, fluids, rho0, p0):
    import sys
    print(bcs)
    g0 = []
    g1 = []
    for bc in bcs:
        if bc == 'u_no_slip':
            for name in bcs[bc]:
                g1.append(
                    NoSlipWallHashemi(dest='boundary', sources=None)
                )
            return [g1]
        if bc == 'u_slip':
            print("slip bc doesn't exist")
            sys.exit(0)
        if bc == 'p_solid':
            from pysph.sph.wc.kernel_correction import GradientCorrectionPreStep, GradientCorrection
            for name in bcs[bc]:
                g0.append(
                    GradientCorrectionPreStep(dest='boundary', sources=['fluid']),
                )
                g1.extend([
                    GradientCorrection(dest='boundary', sources=['fluid']),
                    PressureWallHashemi(dest='boundary', sources=['fluid']),]
                )
            return [g0, g1]
    return []