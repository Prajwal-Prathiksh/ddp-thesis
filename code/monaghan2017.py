#NOQA
r"""
SPH-ϵ Scheme
Author: K T Prajwal Prathiksh
###
References
-----------
    .. [Monaghan2017] J. J. Monaghan, “SPH-ϵ simulation of 2D turbulence driven by a moving cylinder,” Eur. J. Mech. B/Fluids, vol. 65, pp. 486–493, 2017, doi: 10.1016/j.euromechflu.2017.03.011.
"""
###########################################################################
# Import
###########################################################################
from pst import NumberDensityMoment
from pysph.base.kernels import WendlandQuinticC4
from pysph.base.utils import get_particle_array_wcsph
from pysph.solver.solver import Solver
from pysph.sph.basic_equations import (ContinuityEquation, SummationDensity,
                                       XSPHCorrection)
from pysph.sph.equation import Equation, Group
from pysph.sph.integrator import PECIntegrator
from pysph.sph.integrator_step import IntegratorStep
from pysph.sph.scheme import Scheme
from pysph.sph.wc.basic import TaitEOS
from pysph.sph.wc.kernel_correction import (GradientCorrection,
                                            GradientCorrectionPreStep)
from pysph.tools.sph_evaluator import SPHEvaluator
from tsph_with_pst import (CopyRhoToGhost, DensityGradient, IterativePSTNew,
                           SaveInitialdistances, UpdateDensity, UpdateVelocity,
                           VelocityGradient, GradientCorrectionPreStepNew)


###########################################################################
# Equations
###########################################################################
class LinearEOS(Equation):
    def __init__(self, dest, sources, c0, rho0):
        self.c0 = c0
        self.rho0 = rho0

        super(LinearEOS, self).__init__(dest, sources)

    def initialize(self, d_p, d_idx, d_rho):
        d_p[d_idx] = self.c0**2 * (d_rho[d_idx] - self.rho0)

# Momentum Equation------------------------------------------------------
class MonaghanSPHEpsilonMomentumEquation(Equation):
    def __init__(self, dest, sources, rho0, c0, alpha=1.0, eps=0.5):
        self.rho0 = rho0
        self.c0 = c0
        self.alpha = alpha
        self.eps = eps

        self.fac = self.alpha * self.c0
        self.eps_fac = self.eps/(2.0*self.rho0)
        super().__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0
    
    def loop(
        self, d_idx, s_idx, d_au, d_av, d_aw, d_rho, d_p,
        s_m, s_rho, s_p, DWIJ, VIJ, XIJ, RIJ, EPS
    ):
        p_i = d_p[d_idx]
        p_j = s_p[s_idx]
        rho_i = d_rho[d_idx]
        rho_j = s_rho[s_idx]
        m_j = s_m[s_idx]

        # Compute the pressure force
        p_force = p_i/(rho_i**2 + EPS) + p_j/(rho_j**2 + EPS)

        # Compute the viscosity force
        rhoavg = 0.5*(rho_i + rho_j)
        vijdotxij = VIJ[0]*XIJ[0] + VIJ[1]*XIJ[1] + VIJ[2]*XIJ[2]
        visc_force = - self.fac * vijdotxij / (rhoavg * (RIJ + EPS))

        # Compute correction force
        v2ij = VIJ[0]**2 + VIJ[1]**2 + VIJ[2]**2
        corr_force = - self.eps_fac * v2ij

        # Compute the total force
        force = m_j*(p_force + visc_force + corr_force)

        # Update the acceleration
        d_au[d_idx] += - force * DWIJ[0]
        d_av[d_idx] += - force * DWIJ[1]
        d_aw[d_idx] += - force * DWIJ[2]

###########################################################################
# Steppers
###########################################################################
class WCSPHStep(IntegratorStep):
    """Standard Predictor Corrector integrator for the WCSPH formulation

    Use this integrator for WCSPH formulations. In the predictor step,
    the particles are advanced to `t + dt/2`. The particles are then
    advanced with the new force computed at this position.

    This integrator can be used in PEC or EPEC mode.

    The same integrator can be used for other problems. Like for
    example solid mechanics (see SolidMechStep)

    """
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
                   d_aw, d_ax, d_ay, d_az, d_arho, dt):
        dtb2 = 0.5*dt

        d_u[d_idx] = d_u0[d_idx] + dtb2*d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dtb2*d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dtb2*d_aw[d_idx]


        d_x[d_idx] = d_x0[d_idx] + dtb2 * d_ax[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dtb2 * d_ay[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dtb2 * d_az[d_idx]

        # Update densities and smoothing lengths from the accelerations
        d_rho[d_idx] = d_rho0[d_idx] + dtb2 * d_arho[d_idx]

    def stage2(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
                   d_u0, d_v0, d_w0, d_u, d_v, d_w, d_rho0, d_rho, d_au, d_av,
                   d_aw, d_ax, d_ay, d_az, d_arho, dt):

        d_u[d_idx] = d_u0[d_idx] + dt*d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dt*d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dt*d_aw[d_idx]

        d_x[d_idx] = d_x0[d_idx] + dt * d_ax[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dt * d_ay[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dt * d_az[d_idx]

        # Update densities and smoothing lengths from the accelerations
        d_rho[d_idx] = d_rho0[d_idx] + dt * d_arho[d_idx]

###########################################################################
# Scheme
###########################################################################
class Monaghan2017Scheme(Scheme):
    def __init__(
        self, fluids, solids, dim, rho0, c0, h0, nu, eps=0.5, gamma=7.,
        pst_freq=10, periodic=True, kernel_corr=True, eos='tait'
    ):
        self.fluids = fluids
        self.solids = solids
        self.dim = dim
        self.rho0 = rho0
        self.c0 = c0
        self.h0 = h0
        self.nu = nu
        self.eps = eps
        self.gamma = gamma
        self.pst_freq = pst_freq
        self.periodic = periodic
        self.kernel_corr = kernel_corr
        self.eos = eos

        self.shifter = None
        self.solver = None

    def add_user_options(self, group):
        group.add_argument(
            "--mon2017-eps", action="store", type=float, dest="eps",
            default=0.25, help="Epsilon for SPH-ϵ scheme"
        )
        group.add_argument(
            '--pst-freq', action='store', type=int, dest='pst_freq',
            default=10, help='PST frequency'
        )
        group.add_argument(
            '--eos', type=str, action='store', dest='eos', default='tait',
            choices=['tait', 'linear'], help='Equation of state to use.'
        )
        group.add_argument(
            '--mon-kernel-corr', action='store', type=str,
            choices=['yes', 'no'], dest='kernel_corr', default='no', 
            help='Whether to use kernel correction or not for Monaghan 2017 scheme.'
        )
    
    def consume_user_options(self, options):
        vars = ['eps', 'pst_freq', 'eos', 'kernel_corr']
        
        data = dict((var, self._smart_getattr(options, var))
                    for var in vars)

        _bool = lambda x: True if x == 'yes' else False
        data['kernel_corr'] = _bool(data['kernel_corr'])

        self.configure(**data)

    def get_timestep(self, cfl=0.5):
        return cfl*self.h0/self.c0

    def configure_solver(
        self, kernel=None, integrator_cls=None, extra_steppers=None, **kw
    ):
        if kernel is None:
            kernel = WendlandQuinticC4(dim=self.dim)

        integrator = PECIntegrator(fluid=WCSPHStep())

        if 'dt' not in kw:
            kw['dt'] = self.get_timestep()

        self.solver = Solver(
            dim=self.dim, integrator=integrator, kernel=kernel, **kw
        )
    
    def get_equations(self):
        self.alpha = 8.*self.nu/(self.c0*self.h0)
        equations = []

        all = self.fluids + self.solids

        # Add equation of state
        g0 = []
        for name in self.fluids:
            if self.eos == 'linear':
                g0.append(LinearEOS(
                    dest=name, sources=None, rho0=self.rho0, c0=self.c0
                ))
            elif self.eos == 'tait':
                g0.append(TaitEOS(
                    dest=name, sources=None, rho0=self.rho0, gamma=self.gamma,
                    c0=self.c0
                ))
        equations.append(Group(equations=g0))

        # Add equation to compute pre-step of Bonet-Lok correction
        if self.kernel_corr:
            g2 = []
            for name in all:
                g2.append(GradientCorrectionPreStepNew(
                    dest=name, sources=all, dim=self.dim
                ))
            equations.append(Group(equations=g2))

        # Add Bonet-Lok correction
        g3 = []
        if self.kernel_corr:
            for name in all:
                g3.append(GradientCorrection(
                    dest=name, sources=all, dim=self.dim
                ))
            equations.append(Group(equations=g3))
        
        # Add continuity equation
        g4 = []
        for name in self.fluids:
            g4.append(ContinuityEquation(
                dest=name, sources=self.fluids
            ))
        equations.append(Group(equations=g4))

        # Add momentum equation
        g5 = []
        if self.kernel_corr:
            g5.append(GradientCorrection(
                dest=name, sources=all, dim=self.dim
            ))
        for name in self.fluids:
            g5.append(MonaghanSPHEpsilonMomentumEquation(
                dest=name, sources=self.fluids,
                rho0=self.rho0, c0=self.c0, alpha=self.alpha, eps=self.eps
            ))
        equations.append(Group(equations=g5))

        # Add XSPH correction
        g6 = []
        for name in self.fluids:
            g6.append(XSPHCorrection(
                dest=name, sources=self.fluids, eps=self.eps
            ))
        equations.append(Group(equations=g6))
        return equations

    def post_step(self, pa_arr, domain):
        if self.shifter is None:
            equations = []
            all = self.fluids + self.solids

            g0 = []
            for name in all:
                g0.append(SummationDensity(dest=name, sources=all))
            equations.append(Group(equations=g0))


            g0 = []
            if self.periodic:
                for name in all:
                    g0.append(CopyRhoToGhost(dest=name, sources=all))
                equations.append(Group(equations=g0, real=False))

            g1 = []
            for name in self.fluids:
                g1.append(GradientCorrectionPreStep(dest=name, sources=all, dim=self.dim))
                g1.append(SaveInitialdistances(dest=name, sources=None))
            equations.append(Group(equations=g1))


            g2 = []
            for name in self.fluids:
                g2.append(GradientCorrection(dest=name, sources=all, dim=self.dim))
                g2.append(VelocityGradient(dest=name, sources=all, dim=self.dim))
                g2.append(DensityGradient(dest=name, sources=all, dim=self.dim))
            equations.append(Group(equations=g2))

            g3 = []
            for name in self.fluids:
                g3.append(IterativePSTNew(dest=name, sources=all))
                g3.append(NumberDensityMoment(dest=name, sources=all, debug=False))
            equations.append(Group(equations=g3, iterate=True, min_iterations=1, max_iterations=10, real=False))

            g4 = []
            for name in self.fluids:
                g4.append(UpdateVelocity(dest=name, sources=None))
                g4.append(UpdateDensity(dest=name, sources=None))
            equations.append(Group(equations=g4))
            print(equations, pa_arr)

            self.shifter = SPHEvaluator(
                arrays=pa_arr, equations=equations, dim=self.dim,
                kernel=self.solver.kernel, backend='cython'
            )

        if self.pst_freq > 0 and self.solver.count % self.pst_freq == 0:
            self.shifter.update()
            self.shifter.evaluate(t=self.solver.t, dt=self.solver.dt)

    def setup_properties(self, particles, clean=False):
        
        dummy = get_particle_array_wcsph(name='junk')
        
        props = list(dummy.properties.keys())
        props += [
            'vmax', 'V0',
            dict(name='dpos', stride=3),
            dict(name='gradrc', stride=3),
            dict(name='gradp', stride=3),
            'ki', 'ki0', 'rhoc', 'rhoc0'
        ]

        output_props = [
            'x', 'y', 'z', 'u', 'v', 'w', 'rho', 'm', 'h', 'pid', 'gid', 'tag',
            'p', 'rhoc', 'gradv',
        ]

        delta_sph_props = [
            dict(name='m_mat', stride=9),
            dict(name='gradv', stride=9),
        ]
        props += delta_sph_props

        for pa in particles:
            self._ensure_properties(pa, props, clean)
            pa.set_output_arrays(output_props)
            pa.add_constant('maxki0', 0.0)
            pa.add_constant('maxki', 0.0)
