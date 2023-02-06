# Standard Delta plus SPH scheme
# Sun et al 2017 and 2019
from pysph.sph.scheme import Scheme
from pysph.sph.equation import Equation
from pysph.sph.integrator_step import IntegratorStep

from pysph.base.reduce_array import serial_reduce_array, parallel_reduce_array
from math import sqrt


class DeltaPlusSPHStep(IntegratorStep):
    def initialize(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
                   d_u0, d_v0, d_w0, d_u, d_v, d_w, d_rho0, d_rho):
        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]
        d_z0[d_idx] = d_z[d_idx]

        d_u0[d_idx] = d_u[d_idx]
        d_v0[d_idx] = d_v[d_idx]
        d_w0[d_idx] = d_w[d_idx]

        d_rho0[d_idx] = d_rho[d_idx]

    def stage1(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
                   d_u0, d_v0, d_w0, d_u, d_v, d_w, d_rho0, d_rho, d_au, d_av,
                   d_aw, d_arho, dt):
        dtb2 = 0.5*dt
        d_u[d_idx] = d_u0[d_idx] + dtb2*d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dtb2*d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dtb2*d_aw[d_idx]

        d_x[d_idx] = d_x0[d_idx] + dtb2 * d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dtb2 * d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dtb2 * d_w[d_idx]

        # Update densities and smoothing lengths from the accelerations
        d_rho[d_idx] = d_rho0[d_idx] + dtb2 * d_arho[d_idx]

    def stage2(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
                   d_u0, d_v0, d_w0, d_u, d_v, d_w, d_rho0, d_rho, d_au, d_av,
                   d_aw, d_arho, dt):

        d_u[d_idx] = d_u0[d_idx] + dt*d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dt*d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dt*d_aw[d_idx]

        d_x[d_idx] = d_x0[d_idx] + dt * d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dt * d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dt * d_w[d_idx]

        # Update densities and smoothing lengths from the accelerations
        d_rho[d_idx] = d_rho0[d_idx] + dt * d_arho[d_idx]


class DeltaPlusPSTSPHStep(IntegratorStep):
    def initialize(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
                   d_u0, d_v0, d_w0, d_u, d_v, d_w, d_rho0, d_rho):
        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]
        d_z0[d_idx] = d_z[d_idx]

        d_u0[d_idx] = d_u[d_idx]
        d_v0[d_idx] = d_v[d_idx]
        d_w0[d_idx] = d_w[d_idx]

        d_rho0[d_idx] = d_rho[d_idx]

    def stage1(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
                   d_u0, d_v0, d_w0, d_u, d_v, d_w, d_rho0, d_rho, d_au, d_av,
                   d_aw, d_arho, d_du, d_dv, d_dw, dt):
        dtb2 = 0.5*dt
        d_u[d_idx] = d_u0[d_idx] + dtb2*d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dtb2*d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dtb2*d_aw[d_idx]

        d_x[d_idx] = d_x0[d_idx] + dtb2 * (d_u[d_idx] + d_du[d_idx])
        d_y[d_idx] = d_y0[d_idx] + dtb2 * (d_v[d_idx] + d_dv[d_idx])
        d_z[d_idx] = d_z0[d_idx] + dtb2 * (d_w[d_idx] + d_dw[d_idx])

        # Update densities and smoothing lengths from the accelerations
        d_rho[d_idx] = d_rho0[d_idx] + dtb2 * d_arho[d_idx]

    def stage2(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
                   d_u0, d_v0, d_w0, d_u, d_v, d_w, d_rho0, d_rho, d_au, d_av,
                   d_aw, d_arho, d_du, d_dv, d_dw, dt):

        d_u[d_idx] = d_u0[d_idx] + dt*d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dt*d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dt*d_aw[d_idx]

        d_x[d_idx] = d_x0[d_idx] + dt * (d_u[d_idx] + d_du[d_idx])
        d_y[d_idx] = d_y0[d_idx] + dt * (d_v[d_idx] + d_dv[d_idx])
        d_z[d_idx] = d_z0[d_idx] + dt * (d_w[d_idx] + d_dw[d_idx])

        # Update densities and smoothing lengths from the accelerations
        d_rho[d_idx] = d_rho0[d_idx] + dt * d_arho[d_idx]


class CopyPropsToGhost(Equation):
    def initialize(self, d_idx, d_tag, d_gid, d_p, d_gradrho, d_rho):
        idx = declare('int')
        if d_tag[d_idx] == 2:
            idx = d_gid[d_idx]
            d_p[d_idx] = d_p[idx]
            d_gradrho[3*d_idx] = d_gradrho[3*idx] 
            d_gradrho[3*d_idx + 1] = d_gradrho[3*idx + 1] 
            d_gradrho[3*d_idx + 2] = d_gradrho[3*idx + 2] 
            d_rho[d_idx] = d_rho[idx]

class LinearEOS(Equation):
    def __init__(self, dest, sources, c0, rho0):
        self.c0 = c0
        self.rho0 = rho0

        super(LinearEOS, self).__init__(dest, sources)

    def initialize(self, d_p, d_idx, d_rho):
        d_p[d_idx] = self.c0**2 * (d_rho[d_idx] - self.rho0)


class ContinuityDeltaPlusSPH(Equation):
    def initialize(self, d_idx, d_arho):
        d_arho[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_u, d_v, d_w, d_du, d_dv, d_dw, DWIJ, d_arho, s_m, s_rho, s_u, s_v, s_w, s_du, s_dv, s_dw, d_rho):
        uhati = d_u[d_idx] + d_du[d_idx]
        vhati = d_v[d_idx] + d_dv[d_idx]
        whati = d_w[d_idx] + d_dw[d_idx]

        uhatj = s_u[s_idx] + s_du[s_idx]
        vhatj = s_v[s_idx] + s_dv[s_idx]
        whatj = s_w[s_idx] + s_dw[s_idx]

        duhatij = uhatj - uhati
        dvhatij = vhatj - vhati
        dwhatij = whatj - whati

        rhoi = d_rho[d_idx]
        vj = s_m[s_idx] / s_rho[s_idx]

        d_arho[d_idx] += -rhoi * (duhatij * DWIJ[0] + dvhatij * DWIJ[1] + dwhatij * DWIJ[2]) * vj

        rhodui = d_rho[d_idx] * d_du[d_idx]
        rhodvi = d_rho[d_idx] * d_dv[d_idx]
        rhodwi = d_rho[d_idx] * d_dw[d_idx]

        rhoduj = s_rho[s_idx] * s_du[s_idx]
        rhodvj = s_rho[s_idx] * s_dv[s_idx]
        rhodwj = s_rho[s_idx] * s_dw[s_idx]

        rhoduij = rhoduj + rhodui
        rhodvij = rhodvj + rhodvi
        rhodwij = rhodwj + rhodwi

        d_arho[d_idx] += (rhoduij * DWIJ[0] + rhodvij * DWIJ[1] + rhodwij * DWIJ[2]) * vj


class MomentumEquationArtificialStress(Equation):
    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_u, d_v, d_w, d_du, d_dv, d_dw, s_m, s_rho,
             s_u, s_v, s_w, s_du, s_dv, s_dw, d_au, d_av, d_aw, DWIJ):

        Volj = s_m[s_idx] / s_rho[s_idx]
        ui = d_u[d_idx]
        vi = d_v[d_idx]
        wi = d_w[d_idx]
        dui = d_du[d_idx]
        dvi = d_dv[d_idx]
        dwi = d_dw[d_idx]

        uj = s_u[s_idx]
        vj = s_v[s_idx]
        wj = s_w[s_idx]
        duj = s_du[s_idx]
        dvj = s_dv[s_idx]
        dwj = s_dw[s_idx]

        d_au[d_idx] += ((uj * duj + ui * dui) * DWIJ[0] +
                        (uj * dvj + ui * dvi) * DWIJ[1] +
                        (uj * dwj + ui * dwi) * DWIJ[2]) * Volj
        d_av[d_idx] += ((vj * duj + vi * dui) * DWIJ[0] +
                        (vj * dvj + vi * dvi) * DWIJ[1] +
                        (vj * dwj + vi * dwi) * DWIJ[2]) * Volj
        d_aw[d_idx] += ((wj * duj + wi * dui) * DWIJ[0] +
                        (wj * dvj + wi * dvi) * DWIJ[1] +
                        (wj * dwj + wi * dwi) * DWIJ[2]) * Volj


class MomentumEquationShiftingAcceleration(Equation):
    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_du, d_dv, d_dw, d_u, d_v, d_w, s_du, s_dv,
             s_dw, s_m, s_rho, DWIJ, d_au, d_av, d_aw):

        vj = s_m[s_idx] / s_rho[s_idx]
        dudwij = -((s_du[s_idx] - d_du[d_idx]) * DWIJ[0] +
                  (s_dv[s_idx] - d_dv[d_idx]) * DWIJ[1] +
                  (s_dw[s_idx] - d_dw[d_idx]) * DWIJ[2]) * vj

        d_au[d_idx] += d_u[d_idx] * dudwij
        d_av[d_idx] += d_v[d_idx] * dudwij
        d_aw[d_idx] += d_w[d_idx] * dudwij


class ComputeShiftVelocity(Equation):
    def __init__(self, dest, sources, hdx, vmax, variable=False,
                 tensile_correction=True):
        self.hdx = hdx
        self.vmax = vmax
        self.variable = variable
        self.tensile_correction = tensile_correction

        super(ComputeShiftVelocity, self).__init__(dest, sources)

    def initialize(self, d_idx, d_du, d_dv, d_dw):
        d_du[d_idx] = 0.0
        d_dv[d_idx] = 0.0
        d_dw[d_idx] = 0.0

    def py_initialize(self, dst, t, dt):
        if self.variable:
            from numpy import sqrt
            vmag = sqrt(dst.u**2 + dst.v**2 + dst.w**2)
            dst.vmax[0] = serial_reduce_array(vmag, 'max')
            # dst.vmax[:] = parallel_reduce_array(dst.vmax, 'max')

    def loop(self, d_idx, d_h, s_idx, s_m, s_rho, dt, d_du, d_dv, d_dw, DWIJ, WIJ,
             SPH_KERNEL, d_vmax):
        vmax = self.vmax
        if self.variable:
            vmax = d_vmax[0]
        hi = d_h[d_idx]
        dx = hi / self.hdx
        fij = 0.0
        wdx = SPH_KERNEL.kernel([0, 0, 0], dx, d_h[d_idx])

        Vj = s_m[s_idx] / s_rho[s_idx]

        if self.tensile_correction:
            R = 0.24
            n = 4
            fij = R * (WIJ / wdx)**n

        fac = -Vj * (1 + fij) * 2 * hi * vmax 
        d_du[d_idx] += fac * DWIJ[0]
        d_dv[d_idx] += fac * DWIJ[1]
        d_dw[d_idx] += fac * DWIJ[2]

    # def post_loop(self, d_idx, d_du, d_dv, d_dw, d_vmax):
    #     dumag = sqrt(d_du[d_idx]**2 + d_dv[d_idx]**2 + d_dw[d_idx]**2)
    #     if dumag > d_vmax[0] * 0.05:
    #         fac = 0.05 * d_vmax[0]/dumag
    #         d_du[d_idx] *= fac
    #         d_dv[d_idx] *= fac
    #         d_dw[d_idx] *= fac


class MomentumEquationSimple(Equation):
    def __init__(self, dest, sources, gx=0.0, gy=0.0, gz=0.0):
        self.gx = gx
        self.gy = gy
        self.gz = gz

        super(MomentumEquationSimple, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw, d_dt_cfl):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, d_p, d_au, d_av, d_aw, s_m, s_rho, s_p,
             DWIJ):
        drho = -1 / d_rho[d_idx]
        vj = s_m[s_idx] / s_rho[s_idx]
        pij = d_p[d_idx] + s_p[s_idx]
        fac = pij * drho * vj

        d_au[d_idx] += fac * DWIJ[0]
        d_av[d_idx] += fac * DWIJ[1]
        d_aw[d_idx] += fac * DWIJ[2]

    def post_loop(self, d_idx, d_au, d_av, d_aw, d_dt_force):
        d_au[d_idx] += self.gx
        d_av[d_idx] += self.gy
        d_aw[d_idx] += self.gz


class MomentumEquationTensileCorrected(Equation):
    def __init__(self, dest, sources, gx=0.0, gy=0.0, gz=0.0):
        self.gx = gx
        self.gy = gy
        self.gz = gz

        super(MomentumEquationTensileCorrected, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw, d_dt_cfl):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, d_p, d_au, d_av, d_aw, s_m, s_rho, s_p,
             DWIJ):
        drho = -1 / d_rho[d_idx]
        vj = s_m[s_idx] / s_rho[s_idx]
        pi = d_p[d_idx]
        pij = 0.0
        if pi > 1e-14:
            pij = d_p[d_idx] + s_p[s_idx]
        else:
            pij = s_p[s_idx] - d_p[d_idx]
        fac = pij * drho * vj

        d_au[d_idx] += fac * DWIJ[0]
        d_av[d_idx] += fac * DWIJ[1]
        d_aw[d_idx] += fac * DWIJ[2]

    def post_loop(self, d_idx, d_au, d_av, d_aw, d_dt_force):
        d_au[d_idx] += self.gx
        d_av[d_idx] += self.gy
        d_aw[d_idx] += self.gz


class DeltaPlusSPHScheme(Scheme):
    def __init__(self, fluids, solids, dim, rho0, c0, h0, hdx, correction,
                 gamma=0.0, gx=0.0, gy=0.0, gz=0.0, delta=0.1,
                 nu=0.0, tensile_correction=False, velocity_correction=False, variable_umax=False):
        # sun et al 2017
        # sun at al 2019 velocity correction true
        self.fluids = fluids
        self.solids = solids
        self.solver = None
        self.rho0 = rho0
        self.c0 = c0
        self.gamma = gamma
        self.dim = dim
        self.h0 = h0
        self.hdx = hdx
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.delta = delta
        self.nu = nu
        self.tensile_correction = tensile_correction
        self.correction = correction
        self.velocity_correction = velocity_correction
        self.variable_umax = variable_umax

    def add_user_options(self, group):
        from pysph.sph.scheme import add_bool_argument
        group.add_argument(
            "--delta", action="store", type=float, dest="delta",
            default=None,
            help="Delta for the delta-SPH."
        )
        group.add_argument(
            "--gamma", action="store", type=float, dest="gamma",
            default=None,
            help="Gamma for the state equation."
        )
        add_bool_argument(
            group, 'tensile-correction', dest='tensile_correction',
            help="Use tensile instability correction.",
            default=None
        )
        add_bool_argument(
            group, 'velocity-correction', dest='velocity_correction',
            help="Use velocity correction.",
            default=None
        )
        add_bool_argument(
            group, 'variable-umax', dest='variable_umax',
            help="Use variable umax.",
            default=None
        )

    def consume_user_options(self, options):
        vars = ['gamma', 'tensile_correction',
                'delta', 'velocity_correction', 'variable_umax']

        data = dict((var, self._smart_getattr(options, var))
                    for var in vars)
        self.configure(**data)

    def get_timestep(self, cfl=0.5):
        return cfl*self.h0/self.c0

    def configure_solver(self, kernel=None, integrator_cls=None,
                         extra_steppers=None, **kw):
        from pysph.base.kernels import CubicSpline
        if kernel is None:
            kernel = CubicSpline(dim=self.dim)

        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        from pysph.sph.integrator import PECIntegrator

        cls = integrator_cls if integrator_cls is not None else PECIntegrator
        step_cls = DeltaPlusSPHStep
        if self.velocity_correction:
            step_cls = DeltaPlusPSTSPHStep
        for name in self.fluids + self.solids:
            if name not in steppers:
                steppers[name] = step_cls()

        integrator = cls(**steppers)

        from pysph.solver.solver import Solver
        if 'dt' not in kw:
            kw['dt'] = self.get_timestep()
        self.solver = Solver(
            dim=self.dim, integrator=integrator, kernel=kernel, **kw
        )

    def get_equations(self):
        from pysph.sph.equation import Group

        from pysph.sph.wc.basic import (ContinuityEquationDeltaSPH,
                                        ContinuityEquationDeltaSPHPreStep)
        from pysph.sph.wc.basic import (MomentumEquation, TaitEOS,
                                        MomentumEquationDeltaSPH)
        from pysph.sph.basic_equations import \
            (ContinuityEquation)
        from pysph.sph.wc.viscosity import (LaminarViscosity,
                                            LaminarViscosityDeltaSPH)
        from pysph.sph.wc.kernel_correction import (
            GradientCorrectionPreStep, MixedKernelCorrectionPreStep,
            MixedGradientCorrection, GradientCorrection)
        from kgf_sph import (
            KGFPreStep, KGFCorrection, FirstOrderPreStep, FirstOrderCorrection)
        from pysph.sph.wc.crksph import CRKSPHPreStep, CRKSPH

        equations = []
        g1 = []
        all = self.fluids + self.solids

        # This correction applies only to solids.
        for name in self.fluids:
            g1.append(
                LinearEOS(dest=name, sources=None, c0=self.c0, rho0=self.rho0))
        if self.velocity_correction:
            for name in self.fluids:
                g1.extend([
                    ComputeShiftVelocity(dest=name,
                                         sources=[name],
                                         hdx=self.hdx,
                                         vmax=self.c0,
                                         variable=self.variable_umax)
                ])
        equations.append(Group(equations=g1, real=False))

        eq2_pre = []
        for name in self.fluids:
            if self.correction == 'gradient':
                eq2_pre.append(
                    GradientCorrectionPreStep(dest=name,
                                              sources=[name],
                                              dim=self.dim))
            elif self.correction == 'kgf':
                eq2_pre.append(
                    KGFPreStep(dest=name, sources=[name], dim=self.dim))
            elif self.correction == 'order1':
                eq2_pre.append(
                    FirstOrderPreStep(dest=name, sources=[name], dim=self.dim))
            elif self.correction == 'crksph':
                eq2_pre.append(
                    CRKSPHPreStep(dest=name, sources=[name], dim=self.dim))
                

        equations.append(Group(equations=eq2_pre, real=False))

        eq2 = []
        for name in self.fluids:
            if self.correction == 'gradient':
                eq2.extend([
                    GradientCorrection(dest=name, sources=[name], dim=self.dim),
                ])
            elif self.correction == 'kgf':
                eq2.extend([
                    KGFCorrection(dest=name, sources=[name], dim=self.dim),
                ])
            elif self.correction == 'order1':
                eq2.extend([
                    FirstOrderCorrection(dest=name, sources=[name], dim=self.dim),
                ])
            elif self.correction == 'crksph':
                eq2.extend([
                    CRKSPH(dest=name, sources=[name], dim=self.dim),
                ])
            eq2.extend(
                [ContinuityEquationDeltaSPHPreStep(dest=name, sources=[name])])

        equations.append(Group(equations=eq2, real=False))

        g2 = []
        for name in self.fluids:
            g2.append(
                CopyPropsToGhost(dest=name, sources=None))
        equations.append(Group(equations=g2, real=False))

        g2 = []
        for name in self.fluids:
            g2.append(
                ContinuityEquationDeltaSPH(dest=name,
                                           sources=[name],
                                           c0=self.c0,
                                           delta=self.delta))
            if self.velocity_correction:
                g2.extend([
                    ContinuityDeltaPlusSPH(
                        dest=name,
                        sources=[name],
                    ),
                    MomentumEquationArtificialStress(
                        dest=name,
                        sources=[name],
                    ),
                    MomentumEquationShiftingAcceleration(
                        dest=name,
                        sources=[name],
                    )
                ])
            else:
                g2.append(ContinuityEquation(dest=name, sources=all))

            if not self.tensile_correction:
                g2.append(
                    MomentumEquationSimple(dest=name,
                                           sources=all,
                                           gx=self.gx,
                                           gy=self.gy,
                                           gz=self.gz))
            else:
                g2.append(
                    MomentumEquationTensileCorrected(dest=name,
                                                     sources=all,
                                                     gx=self.gx,
                                                     gy=self.gy,
                                                     gz=self.gz))

            if abs(self.nu) > 1e-14:
                eq = LaminarViscosity(dest=name,
                                      sources=all,
                                      nu=self.nu)
                g2.insert(-1, eq)
            # else:
            #     alpha = 0.1
            #     g2.append(
            #         MomentumEquationDeltaSPH(dest=name,
            #                                  sources=[name],
            #                                  rho0=self.rho0,
            #                                  c0=self.c0,
            #                                  alpha=alpha))
        equations.append(Group(equations=g2))



        return equations

    def setup_properties(self, particles, clean=True):
        from pysph.base.utils import get_particle_array_wcsph
        dummy = get_particle_array_wcsph(name='junk')
        props = list(dummy.properties.keys())
        output_props = ['x', 'y', 'z', 'u', 'v', 'w', 'rho', 'm', 'h',
                        'pid', 'gid', 'tag', 'p']
        delta_sph_props = [
            {'name': 'm_mat', 'stride': 9},
            {'name': 'gradrho', 'stride': 3},
        ]
        props += delta_sph_props
        if self.velocity_correction:
            props += ['du', 'dv', 'dw']

        for pa in particles:
            self._ensure_properties(pa, props, clean)
            if self.velocity_correction:
                pa.add_constant('vmax', [0.0])
            pa.set_output_arrays(output_props)