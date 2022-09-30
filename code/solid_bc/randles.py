''' P. W. Randles and L. D. Libersky, “Smoothed Particle Hydrodynamics:
Some recent improvements and applications,” Computer Methods in Applied
Mechanics and Engineering, vol. 139, no. 1–4, pp. 375–408, Dec. 1996, doi:
10.1016/S0045-7825(96)01090-0.
'''


from pysph.sph.equation import Equation
from compyle.api import declare
import numpy as np


def boundary_props():
    '''
    '''
    return ['wij', 'bid']


def get_bc_names():
    return ['solid0', 'solid1']


def requires():
    is_mirror = False
    is_boundary = False
    is_ghost = True
    is_ghost_mirror = False
    is_boundary_shift = False

    return is_mirror, is_boundary, is_ghost, is_ghost_mirror, is_boundary_shift


class FindBoundaryFluidParticle(Equation):
    def __init__(self, dest, sources, dx=0.01):
        self.dx = dx
        super(FindBoundaryFluidParticle, self).__init__(dest, sources)

    def initialize(self, d_idx, d_bid):
        d_bid[d_idx] = 1.0

    def loop(self, d_idx, RIJ, d_bid, d_h):
        if RIJ < d_h[d_idx]:
            d_bid[d_idx] = -1.0


class EvaluateBoundaryVelocity(Equation):
    def __init__(self, dest, sources, U=0.0, V=0.0, W=0.0):
        self.U = U
        self.V = V
        self.W = W
        super(EvaluateBoundaryVelocity, self).__init__(dest, sources)

    def initialize(self, d_idx, d_bid, d_uf, d_vf, d_wf):
        if d_bid[d_idx] < 0.0:
            d_uf[d_idx] = 0.0
            d_vf[d_idx] = 0.0
            d_wf[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_bid, s_m, s_rho, WIJ, d_uf, d_vf, d_wf, s_u, s_v, s_w, s_bid, d_wij):
        if d_bid[d_idx] < 0.0:
            d_uf[d_idx] += (s_u[s_idx] - self.U) * WIJ * s_m[s_idx] / s_rho[s_idx]
            d_vf[d_idx] += (s_v[s_idx] - self.V) * WIJ * s_m[s_idx] / s_rho[s_idx]
            d_wf[d_idx] += (s_w[s_idx] - self.W) * WIJ * s_m[s_idx] / s_rho[s_idx]

    def post_loop(self, d_idx, d_u, d_v, d_w, d_uf, d_vf, d_wf, d_bid):
        if d_bid[d_idx] < 0.0:
            d_u[d_idx] = d_uf[d_idx]
            d_v[d_idx] = d_vf[d_idx]
            d_w[d_idx] = d_wf[d_idx]


class SetGhostVelocity(Equation):
    def __init__(self, dest, sources, U=0.0, V=0.0, W=0.0):
        self.U = U
        self.V = V
        self.W = W
        super(SetGhostVelocity, self).__init__(dest, sources)

    def initialize(self, d_idx, d_ug, d_vg, d_wg, d_u, d_v, d_w):
        d_ug[d_idx] = self.U
        d_vg[d_idx] = self.V
        d_wg[d_idx] = self.W
        d_u[d_idx] = self.U
        d_v[d_idx] = self.V
        d_w[d_idx] = self.W


class EvaluateBoundaryPressure(Equation):
    def __init__(self, dest, sources, P=0.0):
        self.P = P
        super(EvaluateBoundaryVelocity, self).__init__(dest, sources)

    def initialize(self, d_idx, d_bid, d_p):
        if d_bid[d_idx] < -0.5:
            d_p[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_bid, s_m, s_rho, WIJ, d_p, s_p, s_bid, d_wij):
        if d_bid[d_idx] < -0.5:
            if s_bid[s_idx] > -0.5:
                d_p[d_idx] += (s_p[s_idx] - self.P) * WIJ * s_m[s_idx] / s_rho[s_idx]
            else:
                d_wij[d_idx] += s_m[s_idx] * WIJ / s_rho[s_idx]

    def post_loop(self, d_idx, d_p, d_wij, d_bid):
        if d_bid[d_idx] < -0.5:
            d_p[d_idx] /= (1 - d_wij[d_idx])


class SetGhostPressure(Equation):
    def __init__(self, dest, sources, P=0.0):
        self.P = P
        super(SetGhostPressure, self).__init__(dest, sources)

    def initialize(self, d_idx, d_p):
        d_p[d_idx] = self.P


def solid_bc(bcs, fluids, rho0, p0):
    import sys
    print(bcs)
    g0 = []
    g1 = []
    for bc in bcs:
        if bc == 'u_no_slip':
            for name in bcs[bc]:
                g0.extend([
                    FindBoundaryFluidParticle(dest='fluid', sources=['solid0']),
                    SetGhostVelocity(dest='solid0', sources=None)]
                )
                g1.extend([
                    EvaluateBoundaryVelocity(dest='fluid', sources=['solid0', 'fluid'])]
                )
            return [g0, g1]
        if bc == 'u_slip':
            print("slip bc doesn't exist")
            sys.exit(0)
        if bc == 'p_solid':
            print("pressure bc doesn't exist")
            sys.exit(0)
    return []