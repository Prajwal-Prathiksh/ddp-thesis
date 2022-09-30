'''S. Marrone, M. Antuono, A. Colagrossi, G. Colicchio, D. Le Touzé,
and G. Graziani, “δ-SPH model for simulating violent impact flows,”
Computer Methods in Applied Mechanics and Engineering, vol. 200, no. 13–16,
pp. 1526–1542, Mar. 2011, doi: 10.1016/j.cma.2010.12.016.
'''

from pysph.sph.equation import Equation
from pysph.sph.wc.linalg import gj_solve, augmented_matrix
from pysph.sph.basic_equations import SummationDensity
from compyle.api import declare


def boundary_props():
    '''
    bid: Boundary particle closest to fluid
    cid: greater than zero if corner particle
    cbid: Boundary particle closed to fluid particle
    '''
    return [{'name':'L', 'stride':16}, 'wij', 'proj']


def get_bc_names():
    return ['solid0', 'solid1']


def requires():
    is_mirror = False
    is_boundary = False
    is_ghost = True
    is_ghost_mirror = True
    is_boundary_shift = False

    return is_mirror, is_boundary, is_ghost, is_ghost_mirror, is_boundary_shift


class MarroneSummationDensity(SummationDensity):
    pass


class LiuCorrectionPreStep(Equation):
    # Liu et al 2005
    def __init__(self, dest, sources, dim=2):
        self.dim = dim

        super(LiuCorrectionPreStep, self).__init__(dest, sources)

    def initialize(self, d_idx, d_L):
        i, j = declare('int', 2)

        for i in range(4):
            for j in range(4):
                d_L[16*d_idx + j+4*i] = 0.0

    def loop(self, d_idx, s_idx, d_h, s_h, s_x, s_y, s_z, d_x, d_y, d_z, s_rho,
             s_m, WIJ, XIJ, DWIJ, d_L):
        Vj = s_m[s_idx] / s_rho[s_idx]
        i16 = declare('int')
        i16 = 16*d_idx

        d_L[i16+0] += WIJ * Vj

        d_L[i16+1] += -XIJ[0] * WIJ * Vj
        d_L[i16+2] += -XIJ[1] * WIJ * Vj
        d_L[i16+3] += -XIJ[2] * WIJ * Vj

        d_L[i16+4] += DWIJ[0] * Vj
        d_L[i16+8] += DWIJ[1] * Vj
        d_L[i16+12] += DWIJ[2] * Vj

        d_L[i16+5] += -XIJ[0] * DWIJ[0] * Vj
        d_L[i16+6] += -XIJ[1] * DWIJ[0] * Vj
        d_L[i16+7] += -XIJ[2] * DWIJ[0] * Vj

        d_L[i16+9] += - XIJ[0] * DWIJ[1] * Vj
        d_L[i16+10] += -XIJ[1] * DWIJ[1] * Vj
        d_L[i16+11] += -XIJ[2] * DWIJ[1] * Vj

        d_L[i16+13] += -XIJ[0] * DWIJ[2] * Vj
        d_L[i16+14] += -XIJ[1] * DWIJ[2] * Vj
        d_L[i16+15] += -XIJ[2] * DWIJ[2] * Vj


class LiuCorrection(Equation):
    def _get_helpers_(self):
        # return [gj_solve]
        return [gj_solve, augmented_matrix]

    def __init__(self, dest, sources, dim=2, tol=0.1):
        self.dim = dim
        self.tol = tol
        super(LiuCorrection, self).__init__(dest, sources)

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

        d_wij[d_idx] = res[0]
        for i in range(self.dim):
            DWIJ[i] = res[i+1]


class MarronePressureBC(Equation):
    def initialize(self, d_idx, d_p):
        d_p[d_idx] = 0.0

    def loop(self, d_idx, s_p, s_m, s_rho, d_p, s_idx, d_wij):
        d_p[d_idx] += s_p[s_idx] * d_wij[d_idx] * s_m[s_idx] / s_rho[s_idx]


class CopyPressureMirrorToGhost(Equation):
    def initialize(self, d_idx, d_p):
        d_p[d_idx] = 0.0

    def initialize_pair(self, d_idx, d_p, s_p):
        d_p[d_idx] = s_p[d_idx]


class MarroneVelocityBC(Equation):
    def initialize(self, d_idx, d_u, d_v, d_w, d_wij):
        d_u[d_idx] = 0.0
        d_v[d_idx] = 0.0
        d_w[d_idx] = 0.0
        d_wij[d_idx] = 0.0

    def loop(self, d_idx, s_u, s_v, s_w, s_m, s_rho, d_u, d_v,
             d_w, s_idx, d_wij, WIJ):
        d_u[d_idx] += s_u[s_idx] * d_wij[d_idx] * s_m[s_idx] / s_rho[s_idx]
        d_v[d_idx] += s_v[s_idx] * d_wij[d_idx] * s_m[s_idx] / s_rho[s_idx]
        d_w[d_idx] += s_w[s_idx] * d_wij[d_idx] * s_m[s_idx] / s_rho[s_idx]


class CopySlipMirrorToGhost(Equation):
    def initialize(self, d_idx, d_ug_star, d_vg_star, d_wg_star):
        d_ug_star[d_idx] = 0.0
        d_vg_star[d_idx] = 0.0
        d_wg_star[d_idx] = 0.0

    def initialize_pair(self, d_idx, d_ug_star, d_vg_star, d_wg_star, s_u, s_v,
                        s_w, d_normal):
        un = (s_u[d_idx] * d_normal[3*d_idx] + s_v[d_idx] * d_normal[3*d_idx+1] + s_w[d_idx] * d_normal[3*d_idx+2])
        d_ug_star[d_idx] = s_u[d_idx] - 2 * un * d_normal[3*d_idx]
        d_vg_star[d_idx] = s_v[d_idx] - 2 * un * d_normal[3*d_idx + 1]
        d_wg_star[d_idx] = s_w[d_idx] - 2 * un * d_normal[3*d_idx + 2]


class CopyNoSlipMirrorToGhost(Equation):
    def initialize(self, d_idx, d_ug, d_vg, d_wg):
        d_ug[d_idx] = 0.0
        d_vg[d_idx] = 0.0
        d_wg[d_idx] = 0.0

    def initialize_pair(self, d_idx, d_ug, d_vg, d_wg, s_u, s_v,
                        s_w):
        d_ug[d_idx] = -s_u[d_idx]
        d_vg[d_idx] = -s_v[d_idx]
        d_wg[d_idx] = -s_w[d_idx]


def solid_bc(bcs, fluids, rho0, p0):
    print(bcs)
    g0, g1, g2, g3 = [], [], [], []
    g0.append(
        MarroneSummationDensity(dest='fluid', sources=['fluid'])
    )
    g1.extend([
        LiuCorrectionPreStep(dest='ghost_mirror', sources=['fluid']),
    ])
    g2.extend([
        LiuCorrection(dest='ghost_mirror', sources=['fluid']),
    ])
    for bc in bcs:
        if bc == 'u_no_slip':
            for name in bcs[bc]:
                g2.append(
                    MarroneVelocityBC(dest='ghost_mirror', sources=['fluid'])
                )
                g3.append(
                    CopyNoSlipMirrorToGhost(dest='solid0', sources=['ghost_mirror'])
                )
        if bc == 'u_slip':
            for name in bcs[bc]:
                g2.append(
                    MarroneVelocityBC(dest='ghost_mirror', sources=['fluid'])
                )
                g3.append(
                    CopySlipMirrorToGhost(dest='solid0', sources=['ghost_mirror'])
                )
        if bc == 'p_solid':
            for name in bcs[bc]:
                g2.append(
                    MarronePressureBC(dest='ghost_mirror', sources=['fluid'])
                )
                g3.append(
                    CopyPressureMirrorToGhost(dest='solid0', sources=['ghost_mirror'])
                )
    print(g0)
    return [g0, g1, g2, g3]