'''M. A. Esmaili Sikarudi and A. H. Nikseresht, “Neumann and Robin
boundary conditions for heat conduction modeling using smoothed particle
hydrodynamics,” Computer Physics Communications, vol. 198, pp. 1–11, Jan.
2016, doi: 10.1016/j.cpc.2015.07.004.
'''
from pysph.sph.equation import Equation
from compyle.api import declare
from pysph.sph.wc.linalg import gj_solve, augmented_matrix
from pysph.sph.basic_equations import SummationDensity


def boundary_props():
    '''
    bid: Boundary particle closest to fluid
    cid: greater than zero if corner particle
    cbid: Boundary particle closed to fluid particle
    '''
    return [{'name':'L', 'stride':16}, 'wij']


def get_bc_names():
    return ['solid0', 'solid1']


def requires():
    is_mirror = False
    is_boundary = False
    is_ghost = True
    is_ghost_mirror = False
    is_boundary_shift = False

    return is_mirror, is_boundary, is_ghost, is_ghost_mirror, is_boundary_shift


class LiuCorrectionEsmaili(Equation):
    def _get_helpers_(self):
        # return [gj_solve]
        return [gj_solve, augmented_matrix]

    def __init__(self, dest, sources, dim=2, tol=0.1):
        self.dim = dim
        self.tol = tol
        super(LiuCorrectionEsmaili, self).__init__(dest, sources)

    def loop(self, d_idx, d_L, WIJ, XIJ, DWIJ, HIJ, d_wij):
        i, j, n, nt = declare('int', 4)
        n = self.dim + 1
        nt = n + 1
        # Note that we allocate enough for a 3D case but may only use a
        # part of the matrix.
        temp = declare('matrix(20)')
        res = declare('matrix(4)')
        for i in range(20):
            temp[i] = 0.0
        for i in range(4):
            res[i] = 0.0

        for i in range(n):
            for j in range(n):
                temp[nt * i + j] = d_L[16 * d_idx + 4 * i + j]
            # Augmented part of matrix
            if i == 0:
                temp[nt*i + n] = WIJ
            else:
                temp[nt*i + n] = DWIJ[i-1]

        error_code = gj_solve(temp, n, 1, res)
        d_wij[d_idx] = WIJ

        # this has been added to remove the terms which are over corrected
        # due to very less neighbbors
        diff = abs(WIJ - res[0])
        if diff < self.tol:
            d_wij[d_idx] = res[0]
            for i in range(self.dim):
                DWIJ[i] = res[i+1]

class EsmailiNoSlipBC(Equation):
    def initialize(self, d_idx, d_ug, d_vg, d_wg):
        d_ug[d_idx] = 0.0
        d_vg[d_idx] = 0.0
        d_wg[d_idx] = 0.0

    def loop(self, d_idx, s_u, s_v, s_w, s_m, s_rho, d_ug, d_vg, d_wg, s_idx,
             d_wij):
        d_ug[d_idx] += -s_u[s_idx] * d_wij[d_idx] * s_m[s_idx] / s_rho[s_idx]
        d_vg[d_idx] += -s_v[s_idx] * d_wij[d_idx] * s_m[s_idx] / s_rho[s_idx]
        d_wg[d_idx] += -s_w[s_idx] * d_wij[d_idx] * s_m[s_idx] / s_rho[s_idx]


def solid_bc(bcs, fluids, rho0, p0):
    from solid_bc.marrone import MarroneSummationDensity, LiuCorrectionPreStep, LiuCorrection
    import sys
    print(bcs)
    g0, g1, g2, g3 = [], [], [], []
    g0.append(MarroneSummationDensity(dest='fluid', sources=['fluid']))
    g1.extend([
        LiuCorrectionPreStep(dest='solid0', sources=['fluid']),
    ])
    g2.extend([
        LiuCorrectionEsmaili(dest='solid0', sources=['fluid']),
    ])
    for bc in bcs:
        if bc == 'u_no_slip':
            g2.append(EsmailiNoSlipBC(dest='solid0', sources=['fluid']))
        if bc == 'u_slip':
            print("slip bc doesn't exist")
            sys.exit(0)
        if bc == 'p_solid':
            print("pressure bc doesn't exist")
            sys.exit(0)
    print(g0)
    return [g0, g1, g2]