# *** THIS IS AN AUTOGENERAED FILE ***
# ***         DO NOT EDIT          ***

from pysph.sph.equation import Equation
from math import exp, pi, sin, cos, sqrt, atan
import numpy as np
from compyle.api import declare


def get_props(x, y, z, t, c0):
    from numpy import sin, cos, exp, log, sqrt
    from numpy import arctan2 as atan2
    u = (${formula['u']}) * np.ones_like(x)
    v = (${formula['v']}) * np.ones_like(x)
    w = (${formula['w']}) * np.ones_like(x)
    p = (${formula['p']}) * np.ones_like(x)
    rhoc = p/c0**2 + 1.0

    return u, v, w, rhoc, p


class AddMomentumSourceTerm(Equation):
    def initialize(self, d_au, d_av, d_aw, d_idx):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def post_loop(self, d_au, d_av, d_aw, d_idx, d_x, d_y, d_z, d_u, d_v, d_w,
                  d_rho, d_rhoc, d_p, t, dt, d_c0):

        x = d_x[d_idx]
        y = d_y[d_idx]
        z = d_z[d_idx]
        c0 = d_c0[0]
        rhoc0 = 1.0
        rho0 = 1.0

        u = ${formula['u']}
        v = ${formula['v']}
        w = ${formula['w']}
        p = ${formula['p']}
        rhoc = p/c0**2 + rhoc0
        rho = d_rho[d_idx]

        d_au[d_idx] += ${formula['su']}
        d_av[d_idx] += ${formula['sv']}
        d_aw[d_idx] += ${formula['sw']}


class SetValuesonSolid(Equation):
    def __init__(self, dest, sources, bc='mms'):
        self.bc = bc
        super(SetValuesonSolid, self).__init__(dest, sources)

    def initialize(
        self, d_idx, d_u, d_v, d_w, d_rhoc, d_p, d_gradv, t, d_x, d_y, d_z,
        d_c0, d_ug_star, d_vg_star, d_wg_star, d_ug, d_vg, d_wg, d_rho):

        idx9 = declare('int')
        idx9 = d_idx * 9

        c0 = d_c0[0]
        x = d_x[d_idx]
        y = d_y[d_idx]
        z = d_z[d_idx]
        rhoc0 = 1.0
        rho0 = 1.0

        u = ${formula['u']}
        v = ${formula['v']}
        w = ${formula['w']}
        p = ${formula['p']}
        rhoc = p/c0**2 + rhoc0
        rho = d_rho[d_idx]

        if not ((self.bc == 'u_inlet') or (self.bc == 'u_outlet')):
            d_u[d_idx] = u
            d_v[d_idx] = v
            d_w[d_idx] = w
        if not self.bc == 'u_slip':
            d_ug_star[d_idx] = u
            d_vg_star[d_idx] = v
            d_wg_star[d_idx] = w
        if not self.bc == 'u_no_slip':
            d_ug[d_idx] = u
            d_vg[d_idx] = v
            d_wg[d_idx] = w
        if not ((self.bc == 'p_solid') or (self.bc == 'p_inlet') or (self.bc == 'p_outlet')):
            d_p[d_idx] = p
        d_rhoc[d_idx] = d_p[d_idx]/c0**2 + rhoc0

        d_gradv[idx9] = ${formula['gradv0']}
        d_gradv[idx9 + 1] = ${formula['gradv1']}
        d_gradv[idx9 + 2] = ${formula['gradv2']}
        d_gradv[idx9 + 3] = ${formula['gradv3']}
        d_gradv[idx9 + 4] = ${formula['gradv4']}
        d_gradv[idx9 + 5] = ${formula['gradv5']}
        d_gradv[idx9 + 6] = ${formula['gradv6']}
        d_gradv[idx9 + 7] = ${formula['gradv7']}
        d_gradv[idx9 + 8] = ${formula['gradv8']}


class AddContinuitySourceTerm(Equation):
    def initialize(self, d_arho, d_idx):
        d_arho[d_idx] = 0.0

    def post_loop(self, d_arho, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_rho,
                  d_rhoc, d_p, t, dt, d_c0, d_h):

        x = d_x[d_idx]
        y = d_y[d_idx]
        z = d_z[d_idx]
        h = d_h[d_idx]
        c0 = d_c0[0]
        rhoc0 = 1.0
        rho0 = 1.0

        u = ${formula['u']}
        v = ${formula['v']}
        w = ${formula['w']}
        p = ${formula['p']}
        rhoc = p/c0**2 + rhoc0
        rho = d_rho[d_idx]

        d_arho[d_idx] += ${formula['srho']}


class AddPressureEvolutionSourceTerm(Equation):
    def initialize(self, d_ap, d_idx):
        d_ap[d_idx] = 0.0

    def post_loop(self, d_ap, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_rho,
                  d_rhoc, d_p, t, dt, d_c0, d_h):

        x = d_x[d_idx]
        y = d_y[d_idx]
        z = d_z[d_idx]
        h = d_h[d_idx]
        c0 = d_c0[0]
        rhoc0 = 1.0
        rho0 = 1.0

        u = ${formula['u']}
        v = ${formula['v']}
        w = ${formula['w']}
        p = ${formula['p']}
        rhoc = p/c0**2 + rhoc0
        rho = d_rho[d_idx]

        d_ap[d_idx] += ${formula['spp']}


def config_eq(eqns, bc):
    for group in eqns:
        for i, equation in enumerate(group.equations):
            kclass = equation.__class__.__name__
            if kclass == 'SetValuesonSolid':
                dest = equation.dest
                _bc = bc
                # do mms on outlet if BC is inlet
                if (dest == 'outlet' and bc.split('_')[-1] == 'inlet'):
                    _bc = 'mms'
                elif (dest == 'inlet' and bc.split('_')[-1] == 'outlet'):
                    _bc = 'mms'
                elif (bc.split('_')[-1].endswith('let') and (not(dest.endswith('let')))):
                    _bc = 'mms'

                if (dest == 'solid1'):
                    _bc = 'mms'
                group.equations[i] = SetValuesonSolid(dest=dest,
                sources=None, bc=_bc)
            elif kclass == 'AddContinuitySourceTerm':
                dest = equation.dest
                group.equations[i] = AddContinuitySourceTerm(dest=dest, sources=None)
            elif kclass == 'AddMomentumSourceTerm':
                dest = equation.dest
                group.equations[i] = AddMomentumSourceTerm(dest=dest, sources=None)
            elif kclass == 'AddPressureEvolutionSourceTerm':
                dest = equation.dest
                group.equations[i] = AddPressureEvolutionSourceTerm(dest=dest, sources=None)
    return eqns