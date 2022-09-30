from pysph.sph.equation import Equation, Group
from math import exp, pi, sin, cos
import numpy as np
from compyle.api import declare, Elementwise, annotate, get_config, wrap
from pysph.tools.sph_evaluator import SPHEvaluator


class SetPressureBC(Equation):
    def initialize(self, d_idx, d_p):
        d_p[d_idx] = 0.0


class SetNoSlipWallVelocity(Equation):
    def initialize(self, d_idx, d_uf, d_vf, d_wf):
        d_uf[d_idx] = 0.0
        d_vf[d_idx] = 0.0
        d_wf[d_idx] = 0.0


class SetSlipWallVelocity(Equation):
    def initialize(self, d_idx, d_ug_star, d_vg_star, d_wg_star):
        d_ug_star[d_idx] = 0.0
        d_vg_star[d_idx] = 0.0
        d_wg_star[d_idx] = 0.0


class SetValuesonSolid(Equation):
    def initialize(self, d_idx, d_u):
        d_u[d_idx] = 0.0


class AddMomentumSourceTerm(Equation):
    def initialize(self, d_au, d_av, d_aw, d_idx):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0


class AddContinuitySourceTerm(Equation):
    def initialize(self, d_arho, d_idx):
        d_arho[d_idx] = 0.0


class AddPressureEvolutionSourceTerm(Equation):
    def initialize(self, d_ap, d_idx):
        d_ap[d_idx] = 0.0


class SetValuesonSolidEDAC(Equation):
    def initialize(self, d_idx, d_u, d_v, d_w):
        d_u[d_idx] = 0.0
        d_v[d_idx] = 0.0
        d_w[d_idx] = 0.0


class XSPHCorrectionDummy(Equation):
    def initialize(self, d_idx, d_ax, d_ay, d_az):
        d_ax[d_idx] = 0.0
        d_ay[d_idx] = 0.0
        d_az[d_idx] = 0.0

    def post_loop(self, d_idx, d_ax, d_ay, d_az, d_u, d_v, d_w):
        d_ax[d_idx] = d_u[d_idx]
        d_ay[d_idx] = d_v[d_idx]
        d_az[d_idx] = d_w[d_idx]


def get_props(x, y, z, t, c0, mms='mms_pres_d1'):
    if mms == 'mms_pres_d1':
        from mms_pres_d1 import get_props
        return get_props(x, y, z, t, c0)
    elif mms == 'mms_noslip_d1':
        from mms_noslip_d1 import get_props
        return get_props(x, y, z, t, c0)
    elif mms == 'mms_pres_d5':
        from mms_pres_d5 import get_props
        return get_props(x, y, z, t, c0)
    elif mms == 'mms_noslip_d5':
        from mms_noslip_d5 import get_props
        return get_props(x, y, z, t, c0)
    elif mms == 'mms_noslip_d6':
        from mms_noslip_d6 import get_props
        return get_props(x, y, z, t, c0)
    elif mms == 'mms_io_vel':
        from mms_io_vel import get_props
        return get_props(x, y, z, t, c0)
    elif mms == 'mms_io_pres':
        from mms_io_pres import get_props
        return get_props(x, y, z, t, c0)
    elif mms == 'mms_slip_d1':
        from mms_slip_d1 import get_props
        return get_props(x, y, z, t, c0)
    elif mms == 'mms_slip_d5':
        from mms_slip_d5 import get_props
        return get_props(x, y, z, t, c0)
    elif mms == 'mms_in_vel_wave':
        from mms_in_vel_wave import get_props
        return get_props(x, y, z, t, c0)
    elif mms == 'mms_out_vel_wave':
        from mms_out_vel_wave import get_props
        return get_props(x, y, z, t, c0)
    elif mms == 'mms_inlet_pres':
        from mms_inlet_pres import get_props
        return get_props(x, y, z, t, c0)
    elif mms == 'mms_outlet_pres':
        from mms_outlet_pres import get_props
        return get_props(x, y, z, t, c0)
    elif mms == 'mmsc1':
        from mms_custom import get_props_p
        return get_props_p(x, y, z, t, c0)
    elif mms == 'mmsc2':
        from mms_custom import get_props_u_slip
        return get_props_u_slip(x, y, z, t, c0)


def config_eq(eqns, mms='mms1', bc='mms'):
    if mms == 'mms_pres_d1':
        from mms_pres_d1 import config_eq
        return config_eq(eqns, bc)
    elif mms == 'mms_noslip_d1':
        from mms_noslip_d1 import config_eq
        return config_eq(eqns, bc)
    elif mms == 'mms_pres_d5':
        from mms_pres_d5 import config_eq
        return config_eq(eqns, bc)
    elif mms == 'mms_noslip_d5':
        from mms_noslip_d5 import config_eq
        return config_eq(eqns, bc)
    elif mms == 'mms_noslip_d6':
        from mms_noslip_d6 import config_eq
        return config_eq(eqns, bc)
    elif mms == 'mms_io_vel':
        from mms_io_vel import config_eq
        return config_eq(eqns, bc)
    elif mms == 'mms_io_pres':
        from mms_io_pres import config_eq
        return config_eq(eqns, bc)
    elif mms == 'mms_slip_d1':
        from mms_slip_d1 import config_eq
        return config_eq(eqns, bc)
    elif mms == 'mms_slip_d5':
        from mms_slip_d5 import config_eq
        return config_eq(eqns, bc)
    elif mms == 'mms_in_vel_wave':
        from mms_in_vel_wave import config_eq
        return config_eq(eqns, bc)
    elif mms == 'mms_out_vel_wave':
        from mms_out_vel_wave import config_eq
        return config_eq(eqns, bc)
    elif mms == 'mms_inlet_pres':
        from mms_inlet_pres import config_eq
        return config_eq(eqns, bc)
    elif mms == 'mms_outlet_pres':
        from mms_outlet_pres import config_eq
        return config_eq(eqns, bc)


def create_packed_fluid(dx, L, nl):
    import os
    filename = os.path.join('preprocess', '_%.4f.npz'%dx)
    print(filename)
    data = np.load(filename)
    xf = data['xf']
    yf = data['yf']

    xs0, ys0 = np.mgrid[dx / 2 - nl:L + nl:dx, dx / 2 - nl:1.5 * L + nl:dx]
    cond0 = ~((xs0>0) & (xs0<L) & (ys0>0) & (ys0<1.5*L))
    xs0 = np.ravel(xs0[cond0])
    ys0 = np.ravel(ys0[cond0])

    xs = np.concatenate((xf, xs0))
    ys = np.concatenate((yf, ys0))

    return xs, ys, xf, yf


def create_scheme(app, rho0, p0, c0):
    from pysph.sph.scheme import SchemeChooser
    from tsph_with_pst import TSPHScheme

    from pysph.sph.wc.edac import EDACScheme
    tsph = TSPHScheme(
        ['fluid'], ['solid0', 'solid1'], dim=2, rho0=rho0,
        hdx=None, nu=None, gamma=7.0, kernel_corr=True
    )

    s = SchemeChooser(
        default='tsph', tsph=tsph
    )

    return s


def create_scheme_io(app, rho0, p0, c0):
    from pysph.sph.scheme import SchemeChooser
    from tsph_with_pst import TSPHScheme

    from pysph.sph.wc.edac import EDACScheme
    tsph = TSPHScheme(
        ['fluid'], ['wall'], ios=['inlet', 'outlet'], dim=2, rho0=rho0,
        hdx=None, nu=None, gamma=7.0, kernel_corr=True
    )

    s = SchemeChooser(
        default='tsph', tsph=tsph
    )

    return s


def configure_scheme(app, rho0, p0, c0):
    from pysph.base.kernels import QuinticSpline
    from config_solid_bc import set_solid_names, get_stepper
    kernel = QuinticSpline(dim=2)
    h = app.dx * app.hdx

    scheme = app.scheme.scheme
    set_solid_names(app.bctype, scheme)
    scheme_split = app.options.scheme.split('_')

    if scheme_split[0] == 'tsph':
        app.scheme.configure(hdx=app.hdx, nu=app.nu)

    extra_steppers = get_stepper(app.bctype)

    app.scheme.configure_solver(kernel=kernel,
                                tf=app.tf,
                                dt=app.dt,
                                extra_steppers=extra_steppers)


def configure_scheme_io(app, rho0, p0, c0):
    from pysph.base.kernels import QuinticSpline
    from config_io import set_fluid_names, get_stepper
    from config_io import get_inlet_outlet_manager
    kernel = QuinticSpline(dim=2)
    h = app.dx * app.hdx

    scheme = app.scheme.scheme
    set_fluid_names(app.bctype, scheme)
    iom = get_inlet_outlet_manager(app)

    app.iom = iom

    app.scheme.configure(hdx=app.hdx, nu=app.nu)

    extra_steppers = get_stepper(app.bctype, app.bc)
    print(extra_steppers)

    app.scheme.configure_solver(kernel=kernel,
                                tf=app.tf,
                                dt=app.dt,
                                extra_steppers=extra_steppers)


def create_equations(app, rho0, p0):
    from config_solid_bc import config_solid_bc
    eqns = app.scheme.get_equations()
    bc = app.bc
    if app.bctype == 'mms':
        bc = 'mms'
    eqns = config_eq(eqns, mms=app.mms, bc=bc)
    scheme = app.scheme.scheme
    eqns = config_solid_bc(eqns, app.bctype, app.bcs, scheme.fluids, rho0, p0)
    print(eqns)
    return eqns


def create_equations_io(app, rho0, p0):
    from config_io import config_io_bc
    eqns = app.scheme.get_equations()
    bc = app.bc
    if app.bctype == 'mms':
        bc = 'mms'
    eqns = config_eq(eqns, mms=app.mms, bc=bc)
    scheme = app.scheme.scheme
    eqns = config_io_bc(eqns, app.bctype, app.bcs, scheme.fluids, rho0, p0)
    print(eqns)
    return eqns