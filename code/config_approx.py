from pysph.base.utils import get_particle_array
from pysph.sph.integrator_step import EulerStep
from pysph.sph.equation import Group, Equation

from pysph.sph.basic_equations import SummationDensity
from solid_bc.marrone import LiuCorrection, LiuCorrectionPreStep

from config_solid_bc import get_iterative

from compyle.api import declare

import numpy as np

def create_equations(bctype='adami', bc={}):
    eqns = []
    eqs = None
    print(bctype, bc)
    sources = ['fluid', 'solid1']
    solid_name = ''
    if bctype == 'adami':
        from solid_bc.adami import solid_bc
        eqs = solid_bc(bc, ['fluid'], 1.0, 1.0)
        solid_name = 'solid0'
    elif bctype == 'marrone':
        from solid_bc.marrone import solid_bc
        eqs = solid_bc(bc, ['fluid'], 1.0, 1.0)
        solid_name = 'solid0'
    elif bctype == 'takeda':
        from solid_bc.takeda import solid_bc
        eqs = solid_bc(bc, ['fluid'], 1.0, 1.0)
        solid_name = 'solid0'
    elif bctype == 'colagrossi':
        from solid_bc.colagrossi import solid_bc
        eqs = solid_bc(bc, ['fluid'], 1.0, 1.0)
        solid_name = 'mirror'
    elif bctype == 'hashemi':
        from solid_bc.hashemi import solid_bc
        eqs = solid_bc(bc, ['fluid'], 1.0, 1.0)
        solid_name = 'solid0'
    elif bctype == 'marongiu':
        from solid_bc.marongiu import solid_bc
        eqs = solid_bc(bc, ['fluid'], 1.0, 1.0)
        solid_name = 'boundary_shift'
    elif bctype == 'new':
        from solid_bc.new_bc import solid_bc
        eqs = solid_bc(bc, ['fluid'], 1.0, 1.0)
        solid_name = 'solid0'
    elif bctype == 'new_marrone':
        from solid_bc.new_marrone import solid_bc
        eqs = solid_bc(bc, ['fluid'], 1.0, 1.0)
        solid_name = 'solid0'

    sources.append(solid_name)


    groups = get_iterative(bctype, bc)
    iterative_added = False
    if eqs is not None:
        if len(eqs) > 0:
            # Add others equations as subsequent groups
            for i in range(len(eqs)):
                if (i in groups):
                    if iterative_added:
                        continue
                    iterative_added = True
                    iterate_eqs = []
                    for eq in groups:
                        iterate_eqs.append(Group(equations=eqs[eq]))
                    eqns.append(
                        Group(equations=iterate_eqs,
                              iterate=True,
                              min_iterations=10,
                              max_iterations=20))
                else:
                    eqns.append(Group(equations=eqs[i]))

    print(eqns)
    return eqns


class EulerStepDummy(EulerStep):
    def stage1(self):
        pass


class VelocityDotNormal(Equation):
    def initialize(self, d_idx, d_vdotn):
        d_vdotn[d_idx] = 0.0

    def loop(self, d_idx, d_vdotn, s_u, s_v, s_w, s_normal, s_idx):
        d_vdotn[d_idx] += (s_u[s_idx] * s_normal[3 * s_idx] +
                           s_v[s_idx] * s_normal[3 * s_idx + 1] +
                           s_w[s_idx] * s_normal[3 * s_idx + 2])


class VelocityDotNormalSolid(Equation):
    def initialize(self, d_idx, d_vdotn):
        d_vdotn[d_idx] = 0.0

    def loop(self, d_idx, d_vdotn, s_ug_star, s_vg_star, s_wg_star, s_normal,
             s_idx):
        d_vdotn[d_idx] += (s_ug_star[s_idx] * s_normal[3 * s_idx] +
                           s_vg_star[s_idx] * s_normal[3 * s_idx + 1] +
                           s_wg_star[s_idx] * s_normal[3 * s_idx + 2])


class PressureGradDotNormal(Equation):
    def initialize(self, d_idx, d_pgraddotn, d_gradp, d_normal):
        d_pgraddotn[d_idx] = (
            d_gradp[3 * d_idx] * d_normal[3 * d_idx] +
            d_gradp[3 * d_idx + 1] * d_normal[3 * d_idx + 1] +
            d_gradp[3 * d_idx + 2] * d_normal[3 * d_idx + 2])


class VelocityGradDotNormal(Equation):
    def initialize(self, d_idx, d_vgraddotn, d_gradv, d_normal):
        d_vgraddotn[3 * d_idx] = (
            d_gradv[9 * d_idx] * d_normal[3 * d_idx] +
            d_gradv[9 * d_idx + 1] * d_normal[3 * d_idx + 1] +
            d_gradv[9 * d_idx + 2] * d_normal[3 * d_idx + 2])

        d_vgraddotn[3 * d_idx + 1] = (
            d_gradv[9 * d_idx + 3] * d_normal[3 * d_idx] +
            d_gradv[9 * d_idx + 4] * d_normal[3 * d_idx + 1] +
            d_gradv[9 * d_idx + 5] * d_normal[3 * d_idx + 2])

        d_vgraddotn[3 * d_idx + 2] = (
            d_gradv[9 * d_idx + 6] * d_normal[3 * d_idx] +
            d_gradv[9 * d_idx + 7] * d_normal[3 * d_idx + 1] +
            d_gradv[9 * d_idx + 8] * d_normal[3 * d_idx + 2])


class VelocityGradientDestSoild(Equation):
    def __init__(self, dest, sources, dim=1, rho0=1.0):
        self.dim = dim
        self.rho0 = rho0

        super(VelocityGradientDestSoild, self).__init__(dest, sources)

    def initialize(self, d_idx, d_gradv):
        i = declare('int')
        for i in range(9):
            d_gradv[9*d_idx+i] = 0.0

    def loop(self, d_idx, s_idx, s_m, s_rho,
             d_gradv, DWIJ, s_ug, s_vg, s_wg, d_u, d_v, d_w):

        i, j = declare('int', 2)
        uji = declare('matrix(3)')
        tmp = s_m[s_idx]/s_rho[s_idx]
        uji[0] = s_ug[s_idx]
        uji[1] = s_vg[s_idx]
        uji[2] = s_wg[s_idx]
        for i in range(3):
            for j in range(3):
                d_gradv[9*d_idx+3*i+j] += tmp * uji[i] * DWIJ[j]


class VelocityGradient(Equation):
    def __init__(self, dest, sources, dim=1, rho0=1.0):
        self.dim = dim
        self.rho0 = rho0

        super(VelocityGradient, self).__init__(dest, sources)

    def initialize(self, d_idx, d_gradv):
        i = declare('int')
        for i in range(9):
            d_gradv[9*d_idx+i] = 0.0

    def loop(self, d_idx, s_idx, s_m, s_rho,
             d_gradv, DWIJ, s_u, s_v, s_w):

        i, j = declare('int', 2)
        uji = declare('matrix(3)')
        tmp = s_m[s_idx]/s_rho[s_idx]
        uji[0] = s_u[s_idx]
        uji[1] = s_v[s_idx]
        uji[2] = s_w[s_idx]
        for i in range(3):
            for j in range(3):
                d_gradv[9*d_idx+3*i+j] += tmp * uji[i] * DWIJ[j]


class PressureGradient(Equation):
    def __init__(self, dest, sources, dim=1, rho0=1.0):
        self.dim = dim
        self.rho0 = rho0

        super(PressureGradient, self).__init__(dest, sources)

    def initialize(self, d_idx, d_gradp):
        i = declare('int')
        for i in range(3):
            d_gradp[3*d_idx+i] = 0.0

    def loop(self, d_idx, s_idx, s_m, s_rho, s_p,
             d_gradp, DWIJ, d_p, d_wij):

        i = declare('int')
        tmp = s_m[s_idx]/s_rho[s_idx]
        rij = s_p[s_idx]
        d_p[d_idx] += tmp * rij * d_wij[d_idx]
        for i in range(3):
            d_gradp[3*d_idx+i] += tmp * rij * DWIJ[i]