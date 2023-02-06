from pysph.sph.scheme import Scheme
from pysph.sph.equation import Equation
from pysph.sph.integrator import Integrator, IntegratorStep
from pst import IterativePST
from pysph.sph.wc.linalg import gj_solve
from compyle.api import declare

from pysph.base.reduce_array import serial_reduce_array, parallel_reduce_array

### Runge-Kutta Second-Order Integrator Step-------------------------------
class RK2Stepper(IntegratorStep):
    def initialize(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
                   d_u0, d_v0, d_w0, d_u, d_v, d_w, d_rhoc0, d_rhoc):
        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]
        d_z0[d_idx] = d_z[d_idx]

        d_u0[d_idx] = d_u[d_idx]
        d_v0[d_idx] = d_v[d_idx]
        d_w0[d_idx] = d_w[d_idx]

        d_rhoc0[d_idx] = d_rhoc[d_idx]

    def stage1(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
                   d_u0, d_v0, d_w0, d_u, d_v, d_w, d_rhoc0, d_rhoc, d_au, d_av,
                   d_aw, d_arho, d_du, d_dv, d_dw, dt):
        dtb2 = 0.5*dt
        d_x[d_idx] = d_x0[d_idx] + dtb2 * (d_u[d_idx] + d_du[d_idx])
        d_y[d_idx] = d_y0[d_idx] + dtb2 * (d_v[d_idx] + d_dv[d_idx])
        d_z[d_idx] = d_z0[d_idx] + dtb2 * (d_w[d_idx] + d_dw[d_idx])

        # Update densities and smoothing lengths from the accelerations
        d_rhoc[d_idx] = d_rhoc0[d_idx] + dtb2 * d_arho[d_idx]

        d_u[d_idx] = d_u0[d_idx] + dtb2*d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dtb2*d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dtb2*d_aw[d_idx]


    def stage2(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
                   d_u0, d_v0, d_w0, d_u, d_v, d_w, d_rhoc0, d_rhoc, d_au, d_av,
                   d_aw, d_arho, d_du, d_dv, d_dw, dt):

        d_x[d_idx] = d_x0[d_idx] + dt * (d_u[d_idx] + d_du[d_idx])
        d_y[d_idx] = d_y0[d_idx] + dt * (d_v[d_idx] + d_dv[d_idx])
        d_z[d_idx] = d_z0[d_idx] + dt * (d_w[d_idx] + d_dw[d_idx])

        # Update densities and smoothing lengths from the accelerations
        d_rhoc[d_idx] = d_rhoc0[d_idx] + dt * d_arho[d_idx]

        d_u[d_idx] = d_u0[d_idx] + dt*d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dt*d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dt*d_aw[d_idx]


class VelGradDotShifting(Equation):
    def initialize(self, d_au, d_av, d_aw, d_idx):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def post_loop(self, d_au, d_av, d_aw, d_idx, d_du, d_dv, d_dw, d_gradv, d_rho):
        idx9 = declare('int')
        idx9 = d_idx * 9
        d_au[d_idx] += (d_gradv[idx9] * d_du[d_idx] +
                        d_gradv[idx9 + 1] * d_dv[d_idx] +
                        d_gradv[idx9 + 2] * d_dw[d_idx])

        d_av[d_idx] += (d_gradv[idx9 + 3] * d_du[d_idx] +
                        d_gradv[idx9 + 4] * d_dv[d_idx] +
                        d_gradv[idx9 + 5] * d_dw[d_idx])

        d_aw[d_idx] += (d_gradv[idx9 + 6] * d_du[d_idx] +
                        d_gradv[idx9 + 7] * d_dv[d_idx] +
                        d_gradv[idx9 + 8] * d_dw[d_idx])


class RhoGradDotShifting(Equation):
    def initialize(self, d_arho, d_idx):
        d_arho[d_idx] = 0.0

    def post_loop(self, d_arho, d_idx, d_du, d_dv, d_dw, d_gradrc):
        idx3 = declare('int')
        idx3 = d_idx * 3
        d_arho[d_idx] += d_gradrc[idx3] * d_du[d_idx] +\
            d_gradrc[idx3 + 1] * d_dv[d_idx] + \
            d_gradrc[idx3 + 2] * d_dw[d_idx]

class CopyShiftVToGhost(Equation):
    def initialize(self, d_idx, d_tag, d_gid, d_du, d_dv, d_dw):
        idx = declare('int')
        if d_tag[d_idx] == 2:
            idx = d_gid[d_idx]
            d_du[d_idx] = d_du[idx]
            d_dv[d_idx] = d_dv[idx]
            d_dw[d_idx] = d_dw[idx]

class CopyGradToGhost(Equation):
    def initialize(self, d_idx, d_tag, d_gid, d_gradv, d_gradrc):
        idx, idx3, didx, didx3, i = declare('int', 5)
        if d_tag[d_idx] == 2:
            idx = d_gid[d_idx] * 9
            idx3 = d_gid[d_idx] * 3
            didx = d_idx * 9
            didx3 = d_idx * 3
            for i in range(9):
                d_gradv[didx + i] = d_gradv[idx + i]
            for i in range(3):
                d_gradrc[didx3 + i] = d_gradrc[idx3 + i]

class ContinuityDeltaPlusSPH(Equation):
    def initialize(self, d_idx, d_arho):
        d_arho[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_u, d_v, d_w, d_du, d_dv, d_dw, DWIJ, d_arho, s_m, s_rho, s_u, s_v, s_w, s_du, s_dv, s_dw, d_rhoc, s_rhoc):
        uhati = d_du[d_idx]
        vhati = d_dv[d_idx]
        whati = d_dw[d_idx]

        uhatj = s_du[s_idx]
        vhatj = s_dv[s_idx]
        whatj = s_dw[s_idx]

        duhatij = uhatj - uhati
        dvhatij = vhatj - vhati
        dwhatij = whatj - whati

        vj = s_m[s_idx] / s_rho[s_idx]

        d_arho[d_idx] += -s_rhoc[s_idx] * (duhatij * DWIJ[0] + dvhatij * DWIJ[1] + dwhatij * DWIJ[2]) * vj

        rhodui = d_rhoc[d_idx] * d_du[d_idx]
        rhodvi = d_rhoc[d_idx] * d_dv[d_idx]
        rhodwi = d_rhoc[d_idx] * d_dw[d_idx]

        rhoduj = s_rhoc[s_idx] * s_du[s_idx]
        rhodvj = s_rhoc[s_idx] * s_dv[s_idx]
        rhodwj = s_rhoc[s_idx] * s_dw[s_idx]

        rhoduij = rhoduj - rhodui
        rhodvij = rhodvj - rhodvi
        rhodwij = rhodwj - rhodwi

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

        d_au[d_idx] += ((uj * duj - ui * dui) * DWIJ[0] +
                        (uj * dvj - ui * dvi) * DWIJ[1] +
                        (uj * dwj - ui * dwi) * DWIJ[2]) * Volj

        d_av[d_idx] += ((vj * duj - vi * dui) * DWIJ[0] +
                        (vj * dvj - vi * dvi) * DWIJ[1] +
                        (vj * dwj - vi * dwi) * DWIJ[2]) * Volj

        d_aw[d_idx] += ((wj * duj - wi * dui) * DWIJ[0] +
                        (wj * dvj - wi * dvi) * DWIJ[1] +
                        (wj * dwj - wi * dwi) * DWIJ[2]) * Volj


class MomentumEquationShiftingAcceleration(Equation):
    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_du, d_dv, d_dw, d_u, d_v, d_w, s_du, s_dv,
             s_dw, s_m, s_rho, DWIJ, d_au, d_av, d_aw, s_u, s_v, s_w):

        vj = s_m[s_idx] / s_rho[s_idx]
        dudwij = -((s_du[s_idx] - d_du[d_idx]) * DWIJ[0] +
                  (s_dv[s_idx] - d_dv[d_idx]) * DWIJ[1] +
                  (s_dw[s_idx] - d_dw[d_idx]) * DWIJ[2]) * vj

        #this change is due to dervation from the taylor series form d_idx -> s_idx
        d_au[d_idx] += s_u[s_idx] * dudwij
        d_av[d_idx] += s_v[s_idx] * dudwij
        d_aw[d_idx] += s_w[s_idx] * dudwij


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


class TSPHWithDSPHScheme(Scheme):
    def __init__(self, fluids, solids, dim, rho0, c0, h0, hdx, nu=0.0, gamma=7.0, kernel_corr=False, split_cal=False):
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
        self.split_cal = split_cal
        self.shifter = None


    def add_user_options(self, group):
        from pysph.sph.scheme import add_bool_argument
        add_bool_argument(
            group, "kernel-corr", dest="kernel_corr",
            help="Use this if kernel correction is required",
            default=None
        )
        add_bool_argument(
            group, "split-cal", dest="split_cal",
            help="use derived calculation for shifting velocity",
            default=None
        )

    def consume_user_options(self, options):
        vars = ["kernel_corr", "split_cal"]

        data = dict((var, self._smart_getattr(options, var))
                    for var in vars)
        self.configure(**data)

    def configure_solver(self, kernel=None, integrator_cls=None,
                         extra_steppers=None, **kw):
        from pysph.base.kernels import CubicSpline
        from tsph_with_pst import RK2Integrator
        if kernel is None:
            kernel = CubicSpline(dim=self.dim)

        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        cls = integrator_cls if integrator_cls is not None else RK2Integrator
        step_cls = RK2Stepper
        for name in self.fluids + self.solids:
            if name not in steppers:
                steppers[name] = step_cls()

        integrator = cls(**steppers)

        from pysph.solver.solver import Solver
        self.solver = Solver(
            dim=self.dim, integrator=integrator, kernel=kernel, **kw
        )

    def get_equations(self):
        from pysph.sph.equation import Group
        from pysph.sph.basic_equations import \
            (SummationDensity)

        from tsph_with_pst import (VelocityGradient, DivGrad, MomentumEquationSecondOrder, DensityGradient, ContinuityEquation, LinearEOS, CopyPropsToGhost, TaitEOS)

        from pysph.sph.wc.kernel_correction import GradientCorrectionPreStep, GradientCorrection

        equations = []
        g1 = []
        all = self.fluids + self.solids

        g0 = []
        for name in all:
            g0.append(SummationDensity(dest=name, sources=all))
            g0.append(LinearEOS(dest=name, sources=None, rho0=self.rho0))
        equations.append(Group(equations=g0))

        g1 = []
        for name in all:
            g1.append(CopyPropsToGhost(dest=name, sources=None))
        equations.append(Group(equations=g1, real=False))

        g0=[]
        for name in self.fluids:
            g0.append(ComputeShiftVelocity(dest=name, sources=all, hdx=self.hdx, vmax=1.0))
        if self.kernel_corr:
            for name in all:
                g0.append(GradientCorrectionPreStep(dest=name, sources=all, dim=self.dim))
        equations.append(Group(equations=g0))

        g1 = []
        for name in all:
            g1.append(CopyShiftVToGhost(dest=name, sources=None))
        equations.append(Group(equations=g1, real=False))

        g2 = []
        if self.kernel_corr:
            for name in all:
                g2.append(GradientCorrection(dest=name, sources=all))
        for name in self.fluids:
            g2.append(VelocityGradient(dest=name, sources=all, dim=self.dim))
            if not self.split_cal:
                g2.append(DensityGradient(dest=name, sources=all, dim=self.dim))
        equations.append(Group(equations=g2))

        g1 = []
        for name in all:
            g1.append(CopyGradToGhost(dest=name, sources=None))
        equations.append(Group(equations=g1, real=False))

        g3 = []
        if self.kernel_corr:
            for name in all:
                g3.append(GradientCorrection(dest=name, sources=all))
        for name in all:
            g3.append(ContinuityEquation(dest=name, sources=all))
            if self.split_cal:
                g3.append(ContinuityDeltaPlusSPH(dest=name, sources=all))
        for name in self.fluids:
            g3.append(DivGrad(dest=name, sources=all, nu=self.nu, rho0=self.rho0))
            g3.append(MomentumEquationSecondOrder(dest=name, sources=all))
            if self.split_cal:
                g3.append(MomentumEquationArtificialStress(dest=name, sources=all))
                g3.append(MomentumEquationShiftingAcceleration(dest=name, sources=all))
            else:
                g3.append(VelGradDotShifting(dest=name, sources=all))
                g3.append(RhoGradDotShifting(dest=name, sources=all))

        equations.append(Group(equations=g3))

        return equations

    def setup_properties(self, particles, clean=True):
        from pysph.base.utils import get_particle_array_wcsph
        dummy = get_particle_array_wcsph(name='junk')
        props = list(dummy.properties.keys()) + [
            'V0'
        ]
        props += ['vmax', 'du', 'dv', 'dw', 'rhoc', 'rhoc0']
        output_props = ['x', 'y', 'z', 'u', 'v', 'w', 'rho', 'm', 'h',
                        'pid', 'gid', 'tag', 'p', 'rhoc']
        if self.kernel_corr:
            delta_sph_props = [
                {'name': 'm_mat', 'stride': 9},
                {'name': 'gradv', 'stride': 9},
                {'name': 'gradrc', 'stride': 3},
            ]
            props += delta_sph_props
        for pa in particles:
            self._ensure_properties(pa, props, clean)
            pa.set_output_arrays(output_props)
            pa.add_constant('maxki0', 0.0)
            pa.add_constant('maxki', 0.0)
            # if pa.name in self.solids:
            #     # This is the load balancing weight for the solid particles.
            #     # They do less work so we reduce the weight.
            #     if 'lb_weight' not in pa.constants:
            #         pa.add_constant('lb_weight', 0.1)
