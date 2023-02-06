from pysph.sph.scheme import Scheme
from pysph.sph.equation import Equation
from pysph.sph.integrator import Integrator, IntegratorStep
from pst import IterativePST, ModifiedFickian, DeltaPlusSPHPST
from pysph.sph.wc.linalg import gj_solve
from compyle.api import declare

from tsph_with_pst import DensityGradient, LinearEOS, MomentumEquationSecondOrder


class RK2Integrator(Integrator):
    def one_timestep(self, t, dt):
        # Initialise `U^{n}`
        self.initialize()

        # Stage 1 - Compute and store `U^{1}`
        self.compute_accelerations()
        self.stage1()
        # Call any post-stage functions
        self.do_post_stage(0.5*dt, 1)

        # Stage 2 - Compute and store `U^{n+1}`
        self.compute_accelerations()
        self.stage2()
        self.update_domain()
        # Call any post-stage functions
        self.do_post_stage(dt, 2)

### Runge-Kutta Second-Order Integrator Step-------------------------------
class RK2Stepper(IntegratorStep):
    def initialize(
        self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
        d_u0, d_v0, d_w0, d_u, d_v, d_w, d_rho, d_rho0
    ):
        d_u0[d_idx] = d_u[d_idx]
        d_v0[d_idx] = d_v[d_idx]
        d_w0[d_idx] = d_w[d_idx]

        d_rho0[d_idx] = d_rho[d_idx]

    def stage1(
        self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
        d_u0, d_v0, d_w0, d_u, d_v, d_w, d_au, d_av, d_aw,
        dt, d_rho, d_arho, d_rho0
    ):
        dtb2 = 0.5*dt

        d_u[d_idx] = d_u0[d_idx] + dtb2*d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dtb2*d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dtb2*d_aw[d_idx]

        d_rho[d_idx] = d_rho0[d_idx] + dtb2*d_arho[d_idx]

    def stage2(
        self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
        d_u0, d_v0, d_w0, d_u, d_v, d_w, d_au, d_av, d_aw,
        dt, d_rho, d_arho, d_rho0
    ):
        d_u[d_idx] = d_u0[d_idx] + dt*d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dt*d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dt*d_aw[d_idx]

        d_rho[d_idx] = d_rho0[d_idx] + dt*d_arho[d_idx]


class RK2StepperRhoc(IntegratorStep):
    def initialize(
        self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
        d_u0, d_v0, d_w0, d_u, d_v, d_w, d_rhoc, d_rhoc0
    ):
        d_u0[d_idx] = d_u[d_idx]
        d_v0[d_idx] = d_v[d_idx]
        d_w0[d_idx] = d_w[d_idx]

        d_rhoc0[d_idx] = d_rhoc[d_idx]

    def stage1(
        self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
        d_u0, d_v0, d_w0, d_u, d_v, d_w, d_au, d_av, d_aw,
        dt, d_rhoc, d_arho, d_rhoc0
    ):
        dtb2 = 0.5*dt
        d_u[d_idx] = d_u0[d_idx] + dtb2*d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dtb2*d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dtb2*d_aw[d_idx]

        d_rhoc[d_idx] = d_rhoc0[d_idx] + dtb2*d_arho[d_idx]

    def stage2(
        self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
        d_u0, d_v0, d_w0, d_u, d_v, d_w, d_au, d_av, d_aw,
        dt, d_rhoc, d_arho, d_rhoc0
    ):

        d_u[d_idx] = d_u0[d_idx] + dt*d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dt*d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dt*d_aw[d_idx]

        d_rhoc[d_idx] = d_rhoc0[d_idx] + dt*d_arho[d_idx]


class TaitEOSOrg(Equation):
    def __init__(self, dest, sources, rho0, gamma, p0=0.0):
        self.rho0 = rho0
        self.rho01 = 1.0/rho0
        self.gamma = gamma
        self.gamma1 = 0.5*(gamma - 1.0)
        self.p0 = p0

        super(TaitEOSOrg, self).__init__(dest, sources)

    def initialize(self, d_idx, d_rho, d_p, d_c0):
        ratio = d_rho[d_idx] * self.rho01
        tmp = pow(ratio, self.gamma)

        c0 = d_c0[0]
        B = self.rho0*c0*c0/self.gamma
        d_p[d_idx] = self.p0 + B * (tmp - 1.0)


class ViscosityClearySOC(Equation):
    def __init__(self, dest, sources, nu, rho0):
        r"""
        Parameters
        ----------
        nu : float
            kinematic viscosity
        """

        self.nu = nu
        self.rho0 = rho0
        super(ViscosityClearySOC, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, s_rho, d_m,
             d_au, d_av, d_aw, s_m,
             R2IJ, EPS, DWIJ, VIJ, XIJ, RIJ, s_rhoc, d_rhoc):

        if RIJ > 1e-14:
            xijdotdwij = XIJ[0]*DWIJ[0] + XIJ[1]*DWIJ[1] + XIJ[2]*DWIJ[2]
            tmp = s_m[s_idx] / (s_rho[s_idx])
            tmp = tmp * (2 * self.nu) * (xijdotdwij/R2IJ)

            d_au[d_idx] += tmp * VIJ[0]
            d_av[d_idx] += tmp * VIJ[1]
            d_aw[d_idx] += tmp * VIJ[2]


class ContinuityEquationOrg(Equation):
    def initialize(self, d_idx, d_arho):
        d_arho[d_idx] = 0.0

    def loop(self, d_idx, d_arho, s_idx, s_m, DWIJ, VIJ, d_rho, s_rho):
        vijdotdwij = DWIJ[0]*VIJ[0] + DWIJ[1]*VIJ[1] + DWIJ[2]*VIJ[2]
        d_arho[d_idx] += s_m[s_idx]*vijdotdwij #/s_rho[s_idx] * d_rho[d_idx]


class ContinuityEquationRhoc(Equation):
    def initialize(self, d_idx, d_arho, d_gradrc):
        d_arho[d_idx] = 0.0
        d_gradrc[3 * d_idx] = 0.0
        d_gradrc[3 * d_idx + 1] = 0.0
        d_gradrc[3 * d_idx + 2] = 0.0

    def loop(self, d_idx, s_idx, s_m, DWIJ, VIJ, d_rho, s_rho, d_rhoc,
             s_rhoc, d_u, d_v, d_w, d_gradrc):
        vj = s_m[s_idx] / s_rho[s_idx]
        rhocij = s_rhoc[s_idx] - d_rhoc[d_idx]
        d_gradrc[3 * d_idx] += rhocij * DWIJ[0] * vj
        d_gradrc[3 * d_idx + 1] += rhocij * DWIJ[1] * vj
        d_gradrc[3 * d_idx + 2] += rhocij * DWIJ[2] * vj

    def post_loop(self, d_idx, d_arho, d_gradrc, d_u, d_v, d_w):
        d_arho[d_idx] -= (d_u[d_idx] * d_gradrc[3 * d_idx] +
                          d_v[d_idx] * d_gradrc[3 * d_idx + 1] +
                          d_w[d_idx] * d_gradrc[3 * d_idx + 2])


class MomentumEquationPressureGradient(Equation):
    def __init__(self, dest, sources, gx=0., gy=0., gz=0.,
                 tdamp=0.0):

        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.tdamp = tdamp
        super(MomentumEquationPressureGradient, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_m, d_rho, s_rho,
             d_au, d_av, d_aw, d_p, s_p, DWIJ):

        # averaged pressure Eq. (7)
        rhoi = d_rho[d_idx]
        rhoj = s_rho[s_idx]
        p_i = d_p[d_idx]
        p_j = s_p[s_idx]

        tmp = -d_m[d_idx] * (p_i/rhoi**2 + p_j/rhoj**2)

        d_au[d_idx] += tmp * DWIJ[0]
        d_av[d_idx] += tmp * DWIJ[1]
        d_aw[d_idx] += tmp * DWIJ[2]


class MomentumEquationSymm(Equation):
    # violeu p 5.130
    def __init__(self, dest, sources,
                 gx=0.0, gy=0.0, gz=0.0):

        self.gx = gx
        self.gy = gy
        self.gz = gz
        super(MomentumEquationSymm, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw, d_dt_cfl):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, d_cs,
             d_p, d_au, d_av, d_aw, s_m,
             s_rho, s_cs, s_p, VIJ,
             XIJ, HIJ, R2IJ, RHOIJ1, EPS,
             DWIJ, WIJ, WDP, d_dt_cfl, d_m, d_rhoc):

        tmp = (s_p[s_idx] + d_p[d_idx])/(d_rhoc[d_idx] * s_rho[s_idx])

        d_au[d_idx] += -s_m[s_idx] * (tmp) * DWIJ[0]
        d_av[d_idx] += -s_m[s_idx] * (tmp) * DWIJ[1]
        d_aw[d_idx] += -s_m[s_idx] * (tmp) * DWIJ[2]

    def post_loop(self, d_idx, d_au, d_av, d_aw, d_dt_force):
        d_au[d_idx] += self.gx
        d_av[d_idx] += self.gy
        d_aw[d_idx] += self.gz


class AdvectionAcceleration(Equation):
    def __init__(self, dest, sources, dim=1, rho0=1.0):
        self.dim = dim
        self.rho0 = rho0

        super(AdvectionAcceleration, self).__init__(dest, sources)

    def initialize(self, d_idx, d_gradv, d_au, d_av, d_aw):
        i = declare('int')
        for i in range(9):
            d_gradv[9*d_idx+i] = 0.0
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, s_rho,
             d_gradv, DWIJ, VIJ):

        i, j = declare('int', 2)
        tmp = s_m[s_idx]/s_rho[s_idx]
        for i in range(3):
            for j in range(3):
                d_gradv[9*d_idx+3*i+j] += tmp * -VIJ[i] * DWIJ[j]

    def post_loop(self, d_idx, d_u, d_v, d_w, d_gradv, d_au, d_av, d_aw):
        d_au[d_idx] += -(d_u[d_idx] * d_gradv[d_idx * 9] +
                        d_v[d_idx] * d_gradv[d_idx * 9 + 1] +
                        d_w[d_idx] * d_gradv[d_idx * 9 + 2])

        d_av[d_idx] += -(d_u[d_idx] * d_gradv[d_idx * 9 + 3] +
                        d_v[d_idx] * d_gradv[d_idx * 9 + 4] +
                        d_w[d_idx] * d_gradv[d_idx * 9 + 5])

        d_aw[d_idx] += -(d_u[d_idx] * d_gradv[d_idx * 9 + 6] +
                        d_v[d_idx] * d_gradv[d_idx * 9 + 7] +
                        d_w[d_idx] * d_gradv[d_idx * 9 + 8])


class AdvectionAccelerationSOC(Equation):
    def __init__(self, dest, sources, dim=1, rho0=1.0):
        self.dim = dim
        self.rho0 = rho0

        super(AdvectionAccelerationSOC, self).__init__(dest, sources)

    def initialize(self, d_idx, d_gradv, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def post_loop(self, d_idx, d_u, d_v, d_w, d_gradv, d_au, d_av, d_aw):
        d_au[d_idx] += -(d_u[d_idx] * d_gradv[d_idx * 9] +
                        d_v[d_idx] * d_gradv[d_idx * 9 + 1] +
                        d_w[d_idx] * d_gradv[d_idx * 9 + 2])

        d_av[d_idx] += -(d_u[d_idx] * d_gradv[d_idx * 9 + 3] +
                        d_v[d_idx] * d_gradv[d_idx * 9 + 4] +
                        d_w[d_idx] * d_gradv[d_idx * 9 + 5])

        d_aw[d_idx] += -(d_u[d_idx] * d_gradv[d_idx * 9 + 6] +
                        d_v[d_idx] * d_gradv[d_idx * 9 + 7] +
                        d_w[d_idx] * d_gradv[d_idx * 9 + 8])


class EWCSPHScheme(Scheme):
    def __init__(self, fluids, solids, dim, rho0, c0, h0, hdx, nu=0.0, gamma=7.0, kernel_corr=False, method='soc', damp_pre=False, periodic=True):
        """Parameters
        ----------

        fluids: list
            List of names of fluid particle arrays.
        solids: list
            List of names of solid particle arrays (or boundaries).
        dim: int
            Dimensionality of the problem.
        rho0: float
            Reference density.
        c0: float
            Reference speed of sound.
        gamma: float
            Gamma for the equation of state.
        h0: float
            Reference smoothing length.
        hdx: float
            Ratio of h/dx.
        nu: float
            Dynamic Viscosity

        """
        self.fluids = fluids
        self.solids = solids
        self.solver = None
        self.rho0 = rho0
        self.c0 = c0
        self.gamma = gamma
        self.dim = dim
        self.h0 = h0
        self.hdx = hdx
        self.nu = nu
        self.kernel_corr = kernel_corr
        self.damp_pre = damp_pre
        self.method = 'soc'
        self.periodic = periodic


    def add_user_options(self, group):
        from pysph.sph.scheme import add_bool_argument
        add_bool_argument(
            group, "kernel-corr", dest="kernel_corr",
            help="Use this if kernel correction is required",
            default=None
        )
        add_bool_argument(
            group, "damp-pre", dest="damp_pre",
            help="if True then apply pressure damping",
            default=None
        )
        group.add_argument(
            '--method', action='store', dest='method', default=None,
            help="method for pressure eval 'soc' or 'no_soc'"
        )

    def consume_user_options(self, options):
        vars = ["kernel_corr", "method", "damp_pre"]

        data = dict((var, self._smart_getattr(options, var))
                    for var in vars)
        self.configure(**data)

    def configure_solver(self, kernel=None, integrator_cls=None,
                         extra_steppers=None, **kw):
        from pysph.base.kernels import CubicSpline
        if kernel is None:
            kernel = CubicSpline(dim=self.dim)

        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        cls = integrator_cls if integrator_cls is not None else RK2Integrator
        step_cls = RK2Stepper
        if self.method == 'soc':
            step_cls = RK2StepperRhoc
        for name in self.fluids + self.solids:
            if name not in steppers:
                steppers[name] = step_cls()

        integrator = cls(**steppers)

        from pysph.solver.solver import Solver
        self.solver = Solver(
            dim=self.dim, integrator=integrator, kernel=kernel, **kw
        )

    def get_equations(self):
        if self.method == 'soc':
            return self.get_equations_soc()
        else:
            return self.get_equations_ewcsph()

    def get_equations_soc(self):
        from pysph.sph.equation import Group
        from pysph.sph.basic_equations import \
            (SummationDensity)
        from tsph_with_pst import (
            CopyPropsToGhost, TaitEOS, ContinuityEquation,
            VelocityGradient, DivGrad, CopyGradVToGhost,
            CopyGradVToGhost, CopyGradRhoToGhost, DensityDamping)

        from scheme_equation import ViscosityCleary
        from pysph.sph.wc.kernel_correction import GradientCorrectionPreStep, GradientCorrection

        equations = []
        g1 = []
        all = self.fluids + self.solids

        g0 = []
        for name in all:
            g0.append(SummationDensity(dest=name, sources=all))
            g0.append(LinearEOS(dest=name, sources=None, rho0=self.rho0, gamma=self.gamma))
        equations.append(Group(equations=g0))

        g1 = []
        if self.periodic:
            for name in all:
                g1.append(CopyPropsToGhost(dest=name, sources=None))
            equations.append(Group(equations=g1, real=False))

        g1 = []
        for name in all:
            g1.append(GradientCorrectionPreStep(dest=name, sources=all, dim=self.dim))
        equations.append(Group(equations=g1))

        g2 = []
        if self.kernel_corr:
            for name in all:
                g2.append(GradientCorrection(dest=name, sources=all))
        for name in self.fluids:
            g2.append(VelocityGradient(dest=name, sources=all, dim=self.dim))
        equations.append(Group(equations=g2))

        if self.nu > 1e-14:
            if self.periodic:
                g1 = []
                for name in all:
                    g1.append(CopyGradVToGhost(dest=name, sources=None))
                equations.append(Group(equations=g1, real=False))

        if self.damp_pre:
            g2 = []
            for name in all:
                g2.append(GradientCorrection(dest=name, sources=all))
            for name in self.fluids:
                g2.append(DensityGradient(dest=name, sources=all))
            equations.append(Group(equations=g2))

            if self.periodic:
                g1 = []
                for name in all:
                    g1.append(CopyGradRhoToGhost(dest=name, sources=None))
                equations.append(Group(equations=g1, real=False))


        g2 = []
        for name in self.fluids:
            g2.append(GradientCorrection(dest=name, sources=all))
            g2.append(ContinuityEquation(dest=name, sources=all))
            g2.append(ContinuityEquationRhoc(dest=name, sources=all))
            g2.append(AdvectionAccelerationSOC(dest=name, sources=all))
            g2.append(MomentumEquationSymm(dest=name, sources=all))
            if self.damp_pre:
                g2.append(DensityDamping(dest=name, sources=all, gamma=0.1))
            if self.nu > 1e-14:
                g2.append(DivGrad(dest=name, sources=all, nu=self.nu, rho0=self.rho0))

        equations.append(Group(equations=g2))

        return equations

    def get_equations_ewcsph(self):
        from pysph.sph.equation import Group
        from pysph.sph.basic_equations import \
            (SummationDensity)
        from tsph_with_pst import CopyPropsToGhost
        from scheme_equation import ViscosityCleary
        from pysph.sph.wc.kernel_correction import GradientCorrectionPreStep, GradientCorrection

        equations = []
        g1 = []
        all = self.fluids + self.solids

        g0 = []
        for name in all:
            g0.append(TaitEOSOrg(dest=name, sources=None, rho0=self.rho0, gamma=self.gamma))
        equations.append(Group(equations=g0))

        if self.periodic:
            g1 = []
            for name in all:
                g1.append(CopyPropsToGhost(dest=name, sources=None))
            equations.append(Group(equations=g1, real=False))

        g1 = []
        for name in all:
            g1.append(GradientCorrectionPreStep(dest=name, sources=all, dim=self.dim))
        equations.append(Group(equations=g1))

        g2 = []
        for name in self.fluids:
            g2.append(GradientCorrection(dest=name, sources=all))
            g2.append(ContinuityEquationOrg(dest=name, sources=all))
            g2.append(AdvectionAcceleration(dest=name, sources=all))
            g2.append(MomentumEquationSymm(dest=name, sources=all))
            if self.nu > 1e-14:
                g2.append(ViscosityCleary(dest=name, sources=all, nu=self.nu, rho0=self.rho0))

        equations.append(Group(equations=g2))

        return equations

    def setup_properties(self, particles, clean=True):
        from pysph.base.utils import get_particle_array_wcsph
        dummy = get_particle_array_wcsph(name='junk')
        props = list(dummy.properties.keys()) + [
            'V0'
        ]
        props += ['vmax', {'name': 'dpos', 'stride': 3}, {'name': 'gradrc', 'stride': 3}, {'name': 'gradp', 'stride': 3}, 'ki', 'ki0', 'rhoc', 'rhoc0', 'ap', 'p0']
        props += ['xi', 'yi', 'zi', 'ui', 'vi', 'wi', 'rhoi']
        output_props = ['x', 'y', 'z', 'u', 'v', 'w', 'rho', 'm', 'h',
                        'pid', 'gid', 'tag', 'p', 'rhoc']
        if self.kernel_corr:
            delta_sph_props = [
                {'name': 'm_mat', 'stride': 9},
                {'name': 'gradv', 'stride': 9},
            ]
            props += delta_sph_props

        for pa in particles:
            self._ensure_properties(pa, props, clean)
            pa.set_output_arrays(output_props)
            pa.add_constant('maxki0', 0.0)
            pa.add_constant('maxki', 0.0)