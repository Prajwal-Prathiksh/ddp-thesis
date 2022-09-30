''' A. Tafuni, J. M. Domínguez, R. Vacondio, and A. J. C. Crespo, “A
versatile algorithm for the treatment of open boundary conditions in
Smoothed particle hydrodynamics GPU models,” Computer Methods in Applied
Mechanics and Engineering, vol. 342, pp. 604–624, Dec. 2018, doi:
10.1016/j.cma.2018.08.004.

We modify this to contain the NRBC nature
'''

from pysph.sph.equation import Equation
from compyle.api import declare
from io_bc.common import (PressureBC, VelocityBC,
                          UpdateNormalsAndDisplacements,
                          CopyNormalsandDistances, LiuCorrectionPreStep,
                          LiuCorrection, PressureGradientJ, VelocityGradientJ)


class CopyTimeValuesandAverage(Equation):
    def __init__(self, dest, sources, rho=1.0, u0=1.25):
        self.rho = rho
        self.u0 = u0
        self.Imin = 0.5 * rho * u0**2

        super(CopyTimeValuesandAverage, self).__init__(dest, sources)

    def initialize(self, d_idx, d_u, d_v, d_p, d_uag, d_pag, d_uta, d_pta,
                   t, d_c0, d_rag, d_rhoc, d_rta):
        i6, i, N = declare('int', 3)
        N = 6
        i6 = N * d_idx
        N -= 1

        for i in range(N):
            d_uag[i6+(N-i)] = d_uag[i6+(N-(i+1))]
            d_pag[i6+(N-i)] = d_pag[i6+(N-(i+1))]
            d_rag[i6+(N-i)] = d_rag[i6+(N-(i+1))]

        u0 = self.u0
        fac = 1.0 / (2. * self.rho * d_c0[0])
        Imin = (0.5 * self.rho * u0**2)**2 * fac
        Eacu = d_p[d_idx] * d_p[d_idx] * fac

        if Eacu < Imin:
            d_uag[i6] = d_u[d_idx]
            d_pag[i6] = d_p[d_idx]
            d_rag[i6] = d_rhoc[d_idx]


        d_uta[d_idx] = 0.0
        d_pta[d_idx] = 0.0
        d_rta[d_idx] = 0.0

        N = 6
        for i in range(N):
            d_uta[d_idx] += d_uag[i6+i]
            d_pta[d_idx] += d_pag[i6+i]
            d_rta[d_idx] += d_rag[i6+i]

        d_uta[d_idx] /= N
        d_pta[d_idx] /= N
        d_rta[d_idx] /= N


class EvaluateCharacteristics(Equation):
    def initialize(self, d_idx, d_j1, d_j2, d_j3, d_c0, d_rhoc, d_u, d_p,
                   d_rta, d_uta, d_pta):
        co = d_c0[0]
        d_j1[d_idx] = -co**2 * (d_rhoc[d_idx] - d_rta[d_idx]) +\
             (d_p[d_idx] - d_pta[d_idx])
        d_j2[d_idx] = d_rhoc[d_idx] * co * (d_u[d_idx] - d_uta[d_idx]) +\
             (d_p[d_idx] - d_pta[d_idx])
        d_j3[d_idx] = -d_rhoc[d_idx] * co * (d_u[d_idx] - d_uta[d_idx]) +\
             (d_p[d_idx] - d_pta[d_idx])


class ExtrapolateCharacteristics(Equation):
    def initialize(self, d_idx, d_j1, d_j2, d_j3, d_wij, d_rhoc):
        d_j1[d_idx] = 0.0
        d_j2[d_idx] = 0.0
        d_j3[d_idx] = 0.0
        d_wij[d_idx] = 0.0
        d_rhoc[d_idx] = 0.0

    def loop(self, d_idx, d_j1, d_j2, d_j3, s_j1, s_j2, s_j3, WIJ, s_idx, d_wij, d_rhoc, s_rhoc):
        d_j1[d_idx] += s_j1[s_idx] * WIJ
        d_j2[d_idx] += s_j2[s_idx] * WIJ
        d_j3[d_idx] += s_j3[s_idx] * WIJ
        d_wij[d_idx] += WIJ
        d_rhoc[d_idx] += s_rhoc[s_idx] * WIJ

    def post_loop(self, d_idx, d_j1, d_j2, d_j3, d_wij, d_rhoc):
        if d_wij[d_idx] > 1e-14:
            d_j1[d_idx] /= d_wij[d_idx]
            d_j2[d_idx] /= d_wij[d_idx]
            d_j3[d_idx] /= d_wij[d_idx]
            d_rhoc[d_idx] /= d_wij[d_idx]

        if d_rhoc[d_idx] < 1e-14:
            d_rhoc[d_idx] = 1.0

class EvaluateVelocityOnGhostInlet(Equation):
    def initialize(self, d_idx, d_u, d_v, d_w):
        d_u[d_idx] = 0.0
        d_v[d_idx] = 0.0
        d_w[d_idx] = 0.0

    def initialize_pair(
        self, d_idx, d_u, d_v, d_w, s_u, s_v, s_w, s_gradv, s_disp,
        s_xn, s_yn, s_zn, d_j3, d_rhoc, d_c0
    ):

        delx = 2 * s_disp[d_idx] * s_xn[d_idx]
        dely = 2 * s_disp[d_idx] * s_yn[d_idx]
        delz = 2 * s_disp[d_idx] * s_zn[d_idx]
        d_u[d_idx] = (s_u[d_idx])

        d_v[d_idx] = (s_v[d_idx])

        d_w[d_idx] = (s_w[d_idx])


class EvaluateVelocityOnGhostOutlet(Equation):
    def initialize(self, d_idx, d_u, d_v, d_w):
        d_u[d_idx] = 0.0
        d_v[d_idx] = 0.0
        d_w[d_idx] = 0.0

    def initialize_pair(
        self, d_idx, d_u, d_v, d_w, s_u, s_v, s_w, s_gradv, s_disp,
        s_xn, s_yn, s_zn, d_j2, d_rhoc, d_c0
    ):

        delx = 2 * s_disp[d_idx] * s_xn[d_idx]
        dely = 2 * s_disp[d_idx] * s_yn[d_idx]
        delz = 2 * s_disp[d_idx] * s_zn[d_idx]
        d_u[d_idx] = (s_u[d_idx])

        d_v[d_idx] = (s_v[d_idx])

        d_w[d_idx] = (s_w[d_idx])


class EvaluatePressureOnGhostInlet(Equation):
    def initialize(self, d_idx, d_p):
        d_p[d_idx] = 0.0

    def initialize_pair(
        self, d_idx, d_p, s_p, s_gradp, s_disp, s_xn, s_yn, s_zn, d_j3
    ):

        delx = 2 * s_disp[d_idx] * s_xn[d_idx]
        dely = 2 * s_disp[d_idx] * s_yn[d_idx]
        delz = 2 * s_disp[d_idx] * s_zn[d_idx]
        d_p[d_idx] = (s_p[d_idx])


class EvaluatePressureOnGhostOutlet(Equation):
    def initialize(self, d_idx, d_p):
        d_p[d_idx] = 0.0

    def initialize_pair(
        self, d_idx, d_p, s_p, s_gradp, s_disp, s_xn, s_yn, s_zn, d_j2
    ):

        delx = 2 * s_disp[d_idx] * s_xn[d_idx]
        dely = 2 * s_disp[d_idx] * s_yn[d_idx]
        delz = 2 * s_disp[d_idx] * s_zn[d_idx]
        d_p[d_idx] = (s_p[d_idx])



def boundary_props():
    '''
    bid: Boundary particle closest to fluid
    cid: greater than zero if corner particle
    cbid: Boundary particle closed to fluid particle
    '''
    return ['uta', 'pta', 'rta', 'ioid', 'disp', 'xn', 'yn', 'zn', {'name':'L', 'stride':16},
             {'name':'pag', 'stride':6}, {'name':'uag', 'stride':6}, {'name':'rag', 'stride':6},
            'j1', 'j2', 'j3']


def get_io_names():
    return ['inlet', 'outlet']


def requires(bc):
    name = bc.split('_')[-1]
    if name == 'outlet':
        mirror_inlet = False
        mirror_outlet = True
    elif name == 'inlet':
        mirror_inlet = True
        mirror_outlet = False

    return mirror_inlet, mirror_outlet

def get_stepper(bc):
    from io_bc.common import InletStep, OutletStep, MirrorStep
    name = bc.split('_')[-1]
    if name == 'outlet':
        return {'inlet':InletStep(), 'outlet':OutletStep(), 'mirror_outlet':MirrorStep()}
    elif name == 'inlet':
        return {'inlet':InletStep(), 'outlet':OutletStep(), 'mirror_inlet':MirrorStep()}


def get_equations(arr, arr_mirror, xn=-1.0, xo=0.0, sources = ['fluid', 'wall'], u0=1.25):
    from pysph.sph.basic_equations import SummationDensity
    #FIXME no summation density on fluid alone
    g0 = [
        UpdateNormalsAndDisplacements(dest=arr, sources=None, xn=xn,
                                      yn=0.0, zn=0.0, xo=xo, yo=0.0,
                                      zo=0.0),
        CopyNormalsandDistances(dest=arr_mirror, sources=[arr]),
        CopyTimeValuesandAverage(dest='fluid', sources=None, u0=u0),
        EvaluateCharacteristics(dest='fluid', sources=None),
        ExtrapolateCharacteristics(dest=arr, sources=['fluid']),
    ]
    g0.extend([SummationDensity(dest=name, sources=sources) for name in sources])
    g1 = [LiuCorrectionPreStep(dest=arr_mirror, sources=sources)]
    g2 = [
        LiuCorrection(dest=arr_mirror, sources=sources),
    ]
    return g0, g1, g2


def velocity_eq(arr, arr_mirror, xn=-1.0, xo=0.0, sources = ['fluid', 'wall'], u0=1.25):
    g0, g1, g2 = get_equations(arr, arr_mirror, xn, xo, sources, u0)
    g2.extend([
        VelocityBC(dest=arr_mirror, sources=sources),
        VelocityGradientJ(dest=arr_mirror, sources=sources)
    ])
    return [g0, g1, g2]


def pressure_eq(arr, arr_mirror, xn=-1.0, xo=0.0, sources = ['fluid', 'wall'], u0=1.25):
    g0, g1, g2 = get_equations(arr, arr_mirror, xn, xo, sources, u0)
    g2.extend([
        PressureBC(dest=arr_mirror, sources=sources),
        PressureGradientJ(dest=arr_mirror, sources=sources, dim=2)
    ])
    return [g0, g1, g2]


def io_bc(bcs, fluids, rho0, p0):
    print(bcs)
    import sys
    #TODO run again as normals were wrong
    for bc in bcs:
        if bc == 'u_outlet':
            eqns = velocity_eq('outlet', 'mirror_outlet', xn=1.0, xo=1.0)
            eqns.append([EvaluateVelocityOnGhostOutlet(dest='outlet', sources=['mirror_outlet'])])
            return eqns
        if bc == 'p_outlet':
            eqns = pressure_eq('outlet', 'mirror_outlet', xn=1.0, xo=1.0)
            eqns.append([EvaluatePressureOnGhostOutlet(dest='outlet', sources=['mirror_outlet'])])
            return eqns
        if bc == 'u_inlet':
            eqns = velocity_eq('inlet', 'mirror_inlet', xn=-1.0, xo=0.0)
            eqns.append([EvaluateVelocityOnGhostInlet(dest='inlet', sources=['mirror_inlet'])])
            return eqns
        if bc == 'p_inlet':
            eqns = pressure_eq('inlet', 'mirror_inlet', xn=-1.0, xo=0.0)
            eqns.append([EvaluatePressureOnGhostInlet(dest='inlet', sources=['mirror_inlet'])])
            return eqns