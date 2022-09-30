# *** THIS IS AN AUTOGENERAED FILE ***
# ***         DO NOT EDIT          ***

from pysph.sph.equation import Equation
from math import exp, pi, sin, cos, sqrt, atan
import numpy as np
from compyle.api import declare


def get_props(x, y, z, t, c0):
    from numpy import sin, cos, exp, log, sqrt
    from numpy import arctan2 as atan2
    u = (y*(y - 1)*exp(-10*t)*cos(2*pi*y) + 1) * np.ones_like(x)
    v = (-x**2*(x - 1)**2*exp(-10*t)*sin(2*pi*y)) * np.ones_like(x)
    w = (0.0) * np.ones_like(x)
    p = ((cos(4*pi*x) + cos(4*pi*y))*exp(-10*t)) * np.ones_like(x)
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

        u = y*(y - 1)*exp(-10*t)*cos(2*pi*y) + 1
        v = -x**2*(x - 1)**2*exp(-10*t)*sin(2*pi*y)
        w = 0.0
        p = (cos(4*pi*x) + cos(4*pi*y))*exp(-10*t)
        rhoc = p/c0**2 + rhoc0
        rho = d_rho[d_idx]

        d_au[d_idx] += v*(-2*pi*y*(y - 1)*exp(-10*t)*sin(2*pi*y) + y*exp(-10*t)*cos(2*pi*y) + (y - 1)*exp(-10*t)*cos(2*pi*y)) - 10*y*(y - 1)*exp(-10*t)*cos(2*pi*y) + 1.0*pi**2*y*(y - 1)*exp(-10*t)*cos(2*pi*y) + 1.0*pi*y*exp(-10*t)*sin(2*pi*y) + 1.0*pi*(y - 1)*exp(-10*t)*sin(2*pi*y) - 0.5*exp(-10*t)*cos(2*pi*y) - 4*pi*exp(-10*t)*sin(4*pi*x)/rhoc
        d_av[d_idx] += u*(-x**2*(2*x - 2)*exp(-10*t)*sin(2*pi*y) - 2*x*(x - 1)**2*exp(-10*t)*sin(2*pi*y)) - 2*pi*v*x**2*(x - 1)**2*exp(-10*t)*cos(2*pi*y) - 1.0*pi**2*x**2*(x - 1)**2*exp(-10*t)*sin(2*pi*y) + 10*x**2*(x - 1)**2*exp(-10*t)*sin(2*pi*y) + 0.5*x**2*exp(-10*t)*sin(2*pi*y) + 1.0*x*(2*x - 2)*exp(-10*t)*sin(2*pi*y) + 0.5*(x - 1)**2*exp(-10*t)*sin(2*pi*y) - 4*pi*exp(-10*t)*sin(4*pi*y)/rhoc
        d_aw[d_idx] += 0.0


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

        u = y*(y - 1)*exp(-10*t)*cos(2*pi*y) + 1
        v = -x**2*(x - 1)**2*exp(-10*t)*sin(2*pi*y)
        w = 0.0
        p = (cos(4*pi*x) + cos(4*pi*y))*exp(-10*t)
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

        d_gradv[idx9] = 0
        d_gradv[idx9 + 1] = -2*pi*y*(y - 1)*exp(-10*t)*sin(2*pi*y) + y*exp(-10*t)*cos(2*pi*y) + (y - 1)*exp(-10*t)*cos(2*pi*y)
        d_gradv[idx9 + 2] = 0
        d_gradv[idx9 + 3] = -x**2*(2*x - 2)*exp(-10*t)*sin(2*pi*y) - 2*x*(x - 1)**2*exp(-10*t)*sin(2*pi*y)
        d_gradv[idx9 + 4] = -2*pi*x**2*(x - 1)**2*exp(-10*t)*cos(2*pi*y)
        d_gradv[idx9 + 5] = 0
        d_gradv[idx9 + 6] = 0
        d_gradv[idx9 + 7] = 0
        d_gradv[idx9 + 8] = 0


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

        u = y*(y - 1)*exp(-10*t)*cos(2*pi*y) + 1
        v = -x**2*(x - 1)**2*exp(-10*t)*sin(2*pi*y)
        w = 0.0
        p = (cos(4*pi*x) + cos(4*pi*y))*exp(-10*t)
        rhoc = p/c0**2 + rhoc0
        rho = d_rho[d_idx]

        d_arho[d_idx] += -2*pi*rhoc*x**2*(x - 1)**2*exp(-10*t)*cos(2*pi*y) - 4*pi*u*exp(-10*t)*sin(4*pi*x)/c0**2 - 4*pi*v*exp(-10*t)*sin(4*pi*y)/c0**2 - 10*(cos(4*pi*x) + cos(4*pi*y))*exp(-10*t)/c0**2


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

        u = y*(y - 1)*exp(-10*t)*cos(2*pi*y) + 1
        v = -x**2*(x - 1)**2*exp(-10*t)*sin(2*pi*y)
        w = 0.0
        p = (cos(4*pi*x) + cos(4*pi*y))*exp(-10*t)
        rhoc = p/c0**2 + rhoc0
        rho = d_rho[d_idx]

        d_ap[d_idx] += -800*pi*rhoc*x**2*(x - 1)**2*exp(-10*t)*cos(2*pi*y) - 4*pi*u*exp(-10*t)*sin(4*pi*x) - 4*pi*v*exp(-10*t)*sin(4*pi*y) - 10*(cos(4*pi*x) + cos(4*pi*y))*exp(-10*t)


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