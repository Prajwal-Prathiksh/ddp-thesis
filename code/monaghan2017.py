#NOQA
r"""
SPH-ϵ Model
#####################
References
-----------
    .. [Monaghan2017] J. J. Monaghan, “SPH-ϵ simulation of 2D turbulence driven by a moving cylinder,” Eur. J. Mech. B/Fluids, vol. 65, pp. 486–493, 2017, doi: 10.1016/j.euromechflu.2017.03.011.
"""
###########################################################################
# Import
###########################################################################
from compyle.api import declare
from math import sqrt
from pysph.sph.equation import Equation
from pysph.base.utils import get_particle_array
from pysph.sph.integrator_step import IntegratorStep
from pysph.sph.scheme import Scheme

###########################################################################
# Equations
###########################################################################
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
        p_force = p_i/rho_i**2 + p_j/rho_j**2

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
# Scheme
###########################################################################
class Monaghan2017Scheme(Scheme):
    def __init__(
        self, fluids, solids, dim, rho0, c0, h0, alpha=1.0, eps=0.5, gamma=7.
    ):
        self.fluids = fluids
        self.solids = solids
        self.dim = dim
        self.rho0 = rho0
        self.c0 = c0
        self.h0 = h0
        self.alpha = alpha
        self.eps = eps
        self.gamma = gamma

    def get_timestep(self, cfl=0.5):
        return cfl*self.h0/self.c0

    def configure_solver(
        self, kernel=None, integrator_cls=None, extra_steppers=None, **kw
    ):
        from pysph.base.kernels import WendlandQuinticC4
        if kernel is None:
            kernel = WendlandQuinticC4(dim=self.dim)

        from pysph.sph.integrator import PECIntegrator
        from pysph.sph.integrator_step import WCSPHStep

        integrator = PECIntegrator(fluid=WCSPHStep())

        from pysph.solver.solver import Solver
        if 'dt' not in kw:
            kw['dt'] = self.get_timestep()

        self.solver = Solver(
            dim=self.dim, integrator=integrator, kernel=kernel, **kw
        )
    
    def get_equations(self):
        from pysph.sph.equation import Group
        from pysph.sph.wc.basic import TaitEOS
        from pysph.sph.basic_equations import (
            ContinuityEquation, XSPHCorrection
        )
        equations = [
            Group(equations=[
                TaitEOS(
                    dest='fluid', sources=None,
                    rho0=self.rho0,
                    c0=self.c0, gamma=self.gamma, p0=0.
                )
            ], real=False
            ),
            Group(equations=[
                ContinuityEquation(dest='fluid', sources=['fluid']),
                MonaghanSPHEpsilonMomentumEquation(
                    dest='fluid', sources=['fluid'],
                    rho0=self.rho0, c0=self.c0,
                    alpha=self.alpha, eps=self.eps
                ),
                XSPHCorrection(dest='fluid', sources=['fluid'], eps=self.eps)
            ], real=True
            )
        ]

        return equations

    def setup_properties(self, particles, clean=False):
        from pysph.base.utils import get_particle_array_wcsph
        dummy = get_particle_array_wcsph(name='junk')
        props = list(dummy.properties.keys())
        output_props = dummy.output_property_arrays
        for pa in particles:
            self._ensure_properties(pa, props, clean)
            pa.set_output_arrays(output_props)
