''' M. Lastiwka, M. Basa, and N. J. Quinlan, “Permeable and non-reflecting
boundary conditions in SPH,” Int. J. Numer. Meth. Fluids, vol. 61, no. 7,
pp. 709–724, Nov. 2009, doi: 10.1002/fld.1971.
'''

from pysph.sph.equation import Equation
from compyle.api import declare


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
    mirror_inlet = False
    mirror_outlet = False

    return mirror_inlet, mirror_outlet

def get_stepper(bc):
    from io_bc.common import InletStep, OutletStep
    return {'inlet':InletStep(), 'outlet':OutletStep()}


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


class EvaluateInletVelocity(Equation):
    def initialize(self, d_u, d_j3, d_rhoc, d_c0, d_idx, d_uta):
        d_u[d_idx] = d_uta[d_idx] - d_j3[d_idx] / (2 * d_rhoc[d_idx] * d_c0[0])


class EvaluateInletPressure(Equation):
    def initialize(self, d_p, d_j3, d_idx, d_pta):
        d_p[d_idx] = d_pta[d_idx] + d_j3[d_idx] / 2


class EvaluateOutletVelocity(Equation):
    def initialize(self, d_u, d_j2, d_rhoc, d_c0, d_idx, d_uta):
        d_u[d_idx] = d_uta[d_idx] + d_j2[d_idx] / (2 * d_rhoc[d_idx] * d_c0[0])


class EvaluateOutletPressure(Equation):
    def initialize(self, d_p, d_j2, d_idx, d_pta):
        d_p[d_idx] = d_pta[d_idx] + d_j2[d_idx] / 2


def eq1(arr, u0=1.25):
    g0 = [
        CopyTimeValuesandAverage(dest='fluid', sources=None, u0=u0),
        EvaluateCharacteristics(dest='fluid', sources=None),
        ExtrapolateCharacteristics(dest=arr, sources=['fluid'])
    ]
    return g0

def io_bc(bcs, fluids, rho0, p0):
    print(bcs)
    import sys

    for bc in bcs:
        if bc == 'u_outlet':
            g0 = eq1('outlet')
            g1 = [EvaluateOutletVelocity(dest='outlet', sources=None)]
            return [g0, g1]
        if bc == 'p_outlet':
            g0 = eq1('outlet')
            g1 = [EvaluateOutletPressure(dest='outlet', sources=None)]
            return [g0, g1]
        if bc == 'u_inlet':
            g0 = eq1('inlet')
            g1 = [EvaluateInletVelocity(dest='inlet', sources=None)]
            return [g0, g1]
        if bc == 'p_inlet':
            g0 = eq1('inlet')
            g1 = [EvaluateInletPressure(dest='inlet', sources=None)]
            return [g0, g1]