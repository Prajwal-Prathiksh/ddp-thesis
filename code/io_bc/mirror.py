''' A. Tafuni, J. M. Domínguez, R. Vacondio, and A. J. C. Crespo, “A
versatile algorithm for the treatment of open boundary conditions in
Smoothed particle hydrodynamics GPU models,” Computer Methods in Applied
Mechanics and Engineering, vol. 342, pp. 604–624, Dec. 2018, doi:
10.1016/j.cma.2018.08.004.
'''

from pysph.sph.equation import Equation
from compyle.api import declare
from io_bc.common import (PressureBC, VelocityBC,
                          UpdateNormalsAndDisplacements,
                          CopyNormalsandDistances, LiuCorrectionPreStep,
                          LiuCorrection, PressureGradientJ, VelocityGradientJ)


class EvaluateVelocityOnGhost(Equation):
    def initialize(self, d_idx, d_u, d_v, d_w):
        d_u[d_idx] = 0.0
        d_v[d_idx] = 0.0
        d_w[d_idx] = 0.0

    def initialize_pair(
        self, d_idx, d_u, d_v, d_w, s_u, s_v, s_w, s_gradv, s_disp,
        s_xn, s_yn, s_zn,
    ):

        delx = 2 * s_disp[d_idx] * s_xn[d_idx]
        dely = 2 * s_disp[d_idx] * s_yn[d_idx]
        delz = 2 * s_disp[d_idx] * s_zn[d_idx]
        d_u[d_idx] = (s_u[d_idx] - delx * s_gradv[9 * d_idx + 0] -
                      dely * s_gradv[9 * d_idx + 1] -
                      delz * s_gradv[9 * d_idx + 2])

        d_v[d_idx] = (s_v[d_idx] - delx * s_gradv[9 * d_idx + 3] -
                      dely * s_gradv[9 * d_idx + 4] -
                      delz * s_gradv[9 * d_idx + 5])

        d_w[d_idx] = (s_w[d_idx] - delx * s_gradv[9 * d_idx + 6] -
                      dely * s_gradv[9 * d_idx + 7] -
                      delz * s_gradv[9 * d_idx + 8])


class EvaluatePressureOnGhost(Equation):
    def initialize(self, d_idx, d_p):
        d_p[d_idx] = 0.0

    def initialize_pair(
        self, d_idx, d_p, s_p, s_gradp, s_disp, s_xn, s_yn, s_zn,
    ):

        delx = 2 * s_disp[d_idx] * s_xn[d_idx]
        dely = 2 * s_disp[d_idx] * s_yn[d_idx]
        delz = 2 * s_disp[d_idx] * s_zn[d_idx]
        d_p[d_idx] = (s_p[d_idx] - delx * s_gradp[3 * d_idx + 0] -
                      dely * s_gradp[3 * d_idx + 1] -
                      delz * s_gradp[3 * d_idx + 2])


def boundary_props():
    '''
    bid: Boundary particle closest to fluid
    cid: greater than zero if corner particle
    cbid: Boundary particle closed to fluid particle
    '''
    return ['ioid', 'disp', 'xn', 'yn', 'zn', {'name':'L', 'stride':16}]


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


def get_equations(arr, arr_mirror, xn=-1.0, xo=0.0, sources=['fluid', 'wall']):
    from pysph.sph.basic_equations import SummationDensity
    g0 = [
        UpdateNormalsAndDisplacements(dest=arr, sources=None, xn=xn,
                                      yn=0.0, zn=0.0, xo=xo, yo=0.0,
                                      zo=0.0),
        CopyNormalsandDistances(dest=arr_mirror, sources=[arr]),
    ]
    g0.extend([SummationDensity(dest=name, sources=sources) for name in sources])
    g1 = [LiuCorrectionPreStep(dest=arr_mirror, sources=sources)]
    g2 = [
        LiuCorrection(dest=arr_mirror, sources=sources),
    ]
    return g0, g1, g2


def velocity_eq(arr, arr_mirror, xn=-1.0, xo=0.0, sources=['fluid', 'wall']):
    g0, g1, g2 = get_equations(arr, arr_mirror, xn, xo, sources)
    g2.extend([
        VelocityBC(dest=arr_mirror, sources=sources),
        VelocityGradientJ(dest=arr_mirror, sources=sources)
    ])
    g3 = [EvaluateVelocityOnGhost(dest=arr, sources=[arr_mirror])]
    return [g0, g1, g2, g3]


def pressure_eq(arr, arr_mirror, xn=-1.0, xo=0.0, sources=['fluid', 'wall']):
    g0, g1, g2 = get_equations(arr, arr_mirror, xn, xo, sources)
    g2.extend([
        PressureBC(dest=arr_mirror, sources=sources),
        PressureGradientJ(dest=arr_mirror, sources=sources, dim=2)
    ])
    g3 = [EvaluatePressureOnGhost(dest=arr, sources=[arr_mirror])]
    return [g0, g1, g2, g3]


def io_bc(bcs, fluids, rho0, p0):
    print(bcs)
    import sys
    for bc in bcs:
        if bc == 'u_outlet':
            return velocity_eq('outlet', 'mirror_outlet', xn=1.0, xo=1.0)
        if bc == 'p_outlet':
            return pressure_eq('outlet', 'mirror_outlet', xn=1.0, xo=1.0)
        if bc == 'u_inlet':
            return velocity_eq('inlet', 'mirror_inlet', xn=-1.0, xo=0.0)
        if bc == 'p_inlet':
            return pressure_eq('inlet', 'mirror_inlet', xn=-1.0, xo=0.0)