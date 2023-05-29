r"""
:math:`k-\varepsilon` Model
Author: K T Prajwal Prathiksh
###
References
-----------
    .. [Shao2006] S. Shao, “Incompressible SPH simulation of wave breaking and overtopping with turbulence modelling,” no. May 2005, pp. 597–621, 2006.
"""
# TODO: Reduce the number of CopyToGhost equations
###########################################################################
# Import
###########################################################################
import inspect
from pysph.sph.equation import Equation
from pysph.sph.integrator_step import IntegratorStep
from pysph.sph.scheme import Scheme
from compyle.api import declare
from math import sqrt

def get_grp_name(grp_var):
    """
    Gets the name of group. Does it from the out most frame inner-wards.
    """
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    name = [
        var_name for var_name, var_val in callers_local_vars if var_val is grp_var
    ]
    grp_name = name[0].replace('g', 'Group_')
    return grp_name


###########################################################################
# Equations
###########################################################################
class StrainRateTensor(Equation):
    def initialize(self, d_idx, d_gradv, d_S):
        didx9 = declare('int')
        didx9 = 9*d_idx

        i, j = declare('int', 2)
        for i in range(9):
            # Calculate index of transpose
            j = i//3 + (i%3)*3
            d_S[didx9+i] = 0.5*(d_gradv[didx9+i] + d_gradv[didx9+j])

class ReynoldsStressTensor(Equation):
    def __init__(self, dest, sources, Cd=0.09):
        self.Cd = Cd
        super().__init__(dest, sources)

    def initialize(self, d_idx, d_rhoc, d_S, d_tau, d_k, d_eps):
        didx9 = declare('int')
        didx9 = 9*d_idx

        rho, k, eps = d_rhoc[d_idx], d_k[d_idx], d_eps[d_idx]
        fac_1 = 2.0*self.Cd*rho*k**2/eps
        fac_2 = 2.0*rho*k/3.0

        i = declare('int')
        for i in range(9):
            d_tau[didx9+i] = fac_1*d_S[didx9+i]
            
        for i in range(0, 9, 4):
            d_tau[didx9+i] -= fac_2

class CopyTensorsToGhost(Equation):
    def initialize(self, d_idx, d_tag, d_gid, d_S, d_tau):
        idx, idx9, i = declare('int', 3)
        if d_tag[d_idx] == 2:
            idx = d_gid[d_idx] * 9
            idx9 = d_idx * 9
            for i in range(9):
                d_S[idx9 + i] = d_S[idx + i]
                d_tau[idx9 + i] = d_tau[idx + i]

class GradKEpsilon(Equation):
    def initialize(self, d_idx, d_gradk, d_gradeps):
        didx3 = declare('int')
        didx3 = 3*d_idx
        for i in range(3):
            d_gradk[didx3+i] = 0.0
            d_gradeps[didx3+i] = 0.0

    def loop(
        self, d_idx, s_idx, d_gradk, d_gradeps, 
        d_k, d_eps, s_k, s_eps, s_m, s_rho, DWIJ
    ):
        didx3 = declare('int')
        didx3 = 3*d_idx

        omega_j = s_m[s_idx]/s_rho[s_idx]
        tmp_k = (s_k[s_idx] - d_k[d_idx])*omega_j
        tmp_eps = (s_eps[s_idx] - d_eps[d_idx])*omega_j
        for i in range(3):
            d_gradk[didx3+i] += tmp_k*DWIJ[i]
            d_gradeps[didx3+i] += tmp_eps*DWIJ[i]

class CopyGradKEpsilonToGhost(Equation):
    def initialize(self, d_idx, d_tag, d_gid, d_gradk, d_gradeps):
        idx, idx3, i = declare('int', 3)
        if d_tag[d_idx] == 2:
            idx = d_gid[d_idx] * 3
            idx3 = d_idx * 3
            for i in range(3):
                d_gradk[idx3 + i] = d_gradk[idx + i]
                d_gradeps[idx3 + i] = d_gradeps[idx + i]

class LaplacianKEpsilon(Equation):
    def initialize(self, d_idx, d_lapk, d_lapeps):
        d_lapk[d_idx] = 0.0
        d_lapeps[d_idx] = 0.0

    def loop(
        self, d_idx, s_idx, d_lapk, d_lapeps, d_gradk, d_gradeps,
        s_gradk, s_gradeps, s_m, s_rho, DWIJ
    ):
        didx3 = declare('int')
        didx3 = 3*d_idx
        omega_j = s_m[s_idx]/s_rho[s_idx]

        gradkdotdw = 0.0
        gradepsdotdw = 0.0
        for i in range(3):
            gradkdotdw += (s_gradk[3*s_idx+i] - d_gradk[didx3+i])*DWIJ[i]
            gradepsdotdw += (s_gradeps[3*s_idx+i] - d_gradeps[didx3+i])*DWIJ[i]
        
        d_lapk[d_idx] += omega_j*gradkdotdw
        d_lapeps[d_idx] += omega_j*gradepsdotdw

class ModifiedLaplacianKEpsilon(Equation):
    def initialize(self, d_idx, d_lapk, d_lapeps):
        d_lapk[d_idx] = 0.0
        d_lapeps[d_idx] = 0.0

    def loop(
        self, d_idx, s_idx, d_k, d_eps, d_lapk, d_lapeps, d_gradk, d_gradeps,
        s_k, s_eps, s_gradk, s_gradeps, s_m, s_rho, DWIJ
    ):
        didx3 = declare('int')
        didx3 = 3*d_idx
        omega_j = s_m[s_idx]/s_rho[s_idx]

        modgradkdotdw = 0.0
        modgradepsdotdw = 0.0
        for i in range(3):
            s_fac = s_k[s_idx]**2/s_eps[s_idx]
            d_fac = d_k[d_idx]**2/d_eps[d_idx]
            modgradkdotdw += (
                s_fac*s_gradk[3*s_idx+i] - d_fac*d_gradk[didx3+i]*DWIJ[i]
            )
            modgradepsdotdw += (
                s_fac*s_gradeps[3*s_idx+i] - d_fac*d_gradeps[didx3+i]*DWIJ[i]
            )
        
        d_lapk[d_idx] += omega_j*modgradkdotdw
        d_lapeps[d_idx] += omega_j*modgradepsdotdw

class CopyLaplacianKEpsilonToGhost(Equation):
    def initialize(self, d_idx, d_tag, d_gid, d_lapk, d_lapeps):
        idx = declare('int')
        if d_tag[d_idx] == 2:
            idx = d_gid[d_idx]
            d_lapk[d_idx] = d_lapk[idx]
            d_lapeps[d_idx] = d_lapeps[idx]
    
class KTransportEquation(Equation):
    def __init__(self, dest, sources, Cd=0.09, sigma_k=1.0):
        self.fac_mod_lap = Cd/sigma_k
        self.fac_Pk = Cd * sqrt(2.0)
        super().__init__(dest, sources)
    
    def initialize(self, d_idx, d_ak):
        d_ak[d_idx] = 0.0

    def loop(self, d_idx, d_ak, d_eps, d_lapk, d_S):
        didx9 = declare('int')
        didx9 = 9*d_idx

        eps = d_eps[d_idx]
        modlapk = d_lapk[d_idx]

        # Calculate Frobenius norm of strain rate tensor
        Ssq = 0.0
        for i in range(9):
            Ssq += d_S[didx9+i]*d_S[didx9+i]
        Pk = self.fac_Pk*sqrt(Ssq)

        d_ak[d_idx] += (self.fac_mod_lap*modlapk) + Pk - eps

class KTransportEquationExpanded(Equation):
    def __init__(self, dest, sources, Cd=0.09, sigma_k=1.0):
        self.Cd = Cd
        self.sigma_k = sigma_k
        super().__init__(dest, sources)
    
    def initialize(self, d_idx, d_ak):
        d_ak[d_idx] = 0.0
    
    def loop(
        self, d_idx, d_ak, d_k, d_eps, d_gradk, d_gradeps, d_lapk, d_S
    ):
        didx3, didx9 = declare('int', 2)
        didx3 = 3*d_idx
        didx9 = 9*d_idx

        k = d_k[d_idx]
        eps = d_eps[d_idx]
        gradkdotgradeps = 0.0
        gradksq = 0.0
        for i in range(3):
            gradkdotgradeps += d_gradk[didx3+i]*d_gradeps[didx3+i]
            gradksq += d_gradk[didx3+i]*d_gradk[didx3+i]
        lapk = d_lapk[d_idx]
        
        # Calculate Frobenius norm of strain rate tensor
        Ssq = 0.0
        for i in range(9):
            Ssq += d_S[didx9+i]*d_S[didx9+i]
        Pk = self.Cd*sqrt(2.0*Ssq)

        div_term = (k**2*lapk + 2*k*gradksq)/eps
        div_term -= k**2*gradkdotgradeps/eps**2
        d_ak[d_idx] += (self.Cd/self.sigma_k)*div_term + Pk - eps

class EpsilonTransportEquation(Equation):
    def __init__(
        self, dest, sources, Cd=0.09, sigma_eps=1.3, C1eps=1.44, C2eps=1.92
    ):
        self.Cd = Cd
        self.sigma_eps = sigma_eps
        self.C1eps = C1eps
        self.C2eps = C2eps
        self.fac_mod_lap = Cd/sigma_eps
        self.fac_Pk = Cd * sqrt(2.0)
        super().__init__(dest, sources)
    
    def initialize(self, d_idx, d_aeps):
        d_aeps[d_idx] = 0.0
    
    def loop(self, d_idx, d_aeps, d_k, d_eps, d_lapeps, d_S):
        didx9 = declare('int')
        didx9 = 9*d_idx

        k = d_k[d_idx]
        eps = d_eps[d_idx]
        modlapeps = d_lapeps[d_idx]
        
        # Calculate Frobenius norm of strain rate tensor
        Ssq = 0.0
        for i in range(9):
            Ssq += d_S[didx9+i]*d_S[didx9+i]
        Pk = self.fac_Pk*sqrt(Ssq)
    
        prod_term = self.C1eps*eps*Pk/k
        decay_term = self.C2eps*eps*eps/k
        d_aeps[d_idx] += self.fac_mod_lap*modlapeps + prod_term - decay_term

class EpsilonTransportEquationExpanded(Equation):
    def __init__(
        self, dest, sources, Cd=0.09, sigma_eps=1.3, C1eps=1.44, C2eps=1.92
    ):
        self.Cd = Cd
        self.sigma_eps = sigma_eps
        self.C1eps = C1eps
        self.C2eps = C2eps
        super().__init__(dest, sources)
    
    def initialize(self, d_idx, d_aeps):
        d_aeps[d_idx] = 0.0
    
    def loop(
        self, d_idx, d_aeps, d_k, d_eps, d_gradk, d_gradeps, d_lapeps, d_S
    ):
        didx3, didx9 = declare('int', 2)
        didx3 = 3*d_idx
        didx9 = 9*d_idx

        k = d_k[d_idx]
        eps = d_eps[d_idx]
        gradkdotgradeps = 0.0
        gradepssq = 0.0
        for i in range(3):
            gradkdotgradeps += d_gradk[didx3+i]*d_gradeps[didx3+i]
            gradepssq += d_gradeps[didx3+i]*d_gradeps[didx3+i]
        lapeps = d_lapeps[d_idx]
        
        # Calculate Frobenius norm of strain rate tensor
        Ssq = 0.0
        for i in range(9):
            Ssq += d_S[didx9+i]*d_S[didx9+i]
        Pk = self.Cd*sqrt(2.0*Ssq)
    
        div_term = (k**2*lapeps + 2*k*gradkdotgradeps)/eps
        div_term -= k**2*gradepssq/eps**2
        d_aeps[d_idx] += (self.Cd/self.sigma_eps)*div_term
        d_aeps[d_idx] += self.C1eps*eps*Pk/k - self.C2eps*eps**2/k

class KEpsilonMomentumEquation(Equation):
    def __init__(
        self, dest, sources, gx=0.0, gy=0.0, gz=0.0, rho0=1.0
    ):
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.rho0 = rho0
        super().__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(
        self, d_idx, s_idx, d_rhoc, d_p, d_au, d_av, d_aw, s_m, s_rho, s_p, d_tau, s_tau, DWIJ
    ):

        didx9, sidx9 = declare('int', 2)
        didx9 = 9*d_idx
        sidx9 = 9*s_idx

        # Pressure gradient term
        omega_j = s_m[s_idx]/s_rho[s_idx]
        rho_i = d_rhoc[d_idx]
        gradp_term = omega_j*(s_p[s_idx] - d_p[d_idx])/rho_i

        # Divergence of Reynolds stress tensor
        div_tau = declare('matrix(3)')
        ij = declare('int')
        for i in range(3):
            div_tau[i] = 0.0
            for j in range(3):
                ij = i*3 + j
                div_tau[i] += (s_tau[sidx9+ij] - d_tau[didx9+ij])*DWIJ[j]
            div_tau[i] *= omega_j/rho_i
        
        d_au[d_idx] += -gradp_term * DWIJ[0] #+ div_tau[0]
        d_av[d_idx] += -gradp_term * DWIJ[1] #+ div_tau[1]
        d_aw[d_idx] += -gradp_term * DWIJ[2] #+ div_tau[2]

    def post_loop(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] += self.gx
        d_av[d_idx] += self.gy
        d_aw[d_idx] += self.gz


###########################################################################
# Integrator Step
###########################################################################
class KEpsilonRK2Step(IntegratorStep):
    def initialize(
        self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
        d_u0, d_v0, d_w0, d_u, d_v, d_w,
        d_rhoc0, d_rhoc,
        d_k0, d_k, d_eps0, d_eps,
    ):
        # Initialise `U^{n}`
        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]
        d_z0[d_idx] = d_z[d_idx]

        d_u0[d_idx] = d_u[d_idx]
        d_v0[d_idx] = d_v[d_idx]
        d_w0[d_idx] = d_w[d_idx]

        d_rhoc0[d_idx] = d_rhoc[d_idx]

        d_k0[d_idx] = d_k[d_idx]
        d_eps0[d_idx] = d_eps[d_idx]

    def stage1(
        self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
        d_u0, d_v0, d_w0, d_u, d_v, d_w, d_au, d_av, d_aw,
        d_rhoc0, d_rhoc, d_arho,
        d_k0, d_k, d_ak, d_eps0, d_eps, d_aeps,
        dt
    ):
        dtb2 = 0.5*dt

        # Compute `U^{1}`
        d_x[d_idx] = d_x0[d_idx] + dtb2*d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dtb2*d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dtb2*d_w[d_idx]

        d_u[d_idx] = d_u0[d_idx] + dtb2*d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dtb2*d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dtb2*d_aw[d_idx]

        d_rhoc[d_idx] = d_rhoc0[d_idx] + dtb2*d_arho[d_idx]

        d_k[d_idx] = d_k0[d_idx] + dtb2*d_ak[d_idx]
        d_eps[d_idx] = d_eps0[d_idx] + dtb2*d_aeps[d_idx]

    def stage2(
        self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
        d_u0, d_v0, d_w0, d_u, d_v, d_w, d_au, d_av, d_aw,
        d_rhoc0, d_rhoc, d_arho,
        d_k0, d_k, d_ak, d_eps0, d_eps, d_aeps,
        dt
    ):
        # Compute `U^{n+1}`
        d_x[d_idx] = d_x0[d_idx] + dt*d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dt*d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dt*d_w[d_idx]

        d_u[d_idx] = d_u0[d_idx] + dt*d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dt*d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dt*d_aw[d_idx]

        d_rhoc[d_idx] = d_rhoc0[d_idx] + dt*d_arho[d_idx]

        d_k[d_idx] = d_k0[d_idx] + dt*d_ak[d_idx]
        d_eps[d_idx] = d_eps0[d_idx] + dt*d_aeps[d_idx]


###########################################################################
# Scheme
###########################################################################
class KEpsilonScheme(Scheme):
    def __init__(
        self, fluids, solids, dim, rho0, c0, h0, hdx, gx=0.0, gy=0.0, gz=0.0,
        nu=0.0, gamma=7.0, kernel_corr=False, periodic=True, k_eps_expand=False
    ):
        self.fluids = fluids
        self.solids = solids
        self.dim = dim
        self.rho0 = rho0
        self.c0 = c0
        self.h0 = h0
        self.hdx = hdx
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.nu = nu
        self.gamma = gamma
        self.kernel_corr = kernel_corr
        self.periodic = periodic
        self.k_eps_expand = k_eps_expand

    def add_user_options(self, group):
        group.add_argument(
            "--k-eps-expand", action="store_true", dest="k_eps_expand",
            help="Use the expanded form of k-epsilon transport equations."
        )
    
    def consume_user_options(self, options):
        vars = ["k_eps_expand"]

        data = dict((var, self._smart_getattr(options, var))
                    for var in vars)
        self.configure(**data)

    def get_timestep(self, cfl=0.5):
        return cfl*self.h0/self.c0

    def configure_solver(
        self, kernel=None, integrator_cls=None, extra_steppers=None, **kw
    ):
        from pysph.base.kernels import WendlandQuinticC4
        if kernel is None:
            kernel = WendlandQuinticC4(dim=self.dim)

        from pysph.sph.integrator import PECIntegrator

        integrator = PECIntegrator(fluid=KEpsilonRK2Step())

        from pysph.solver.solver import Solver
        if 'dt' not in kw:
            kw['dt'] = self.get_timestep()

        self.solver = Solver(
            dim=self.dim, integrator=integrator, kernel=kernel, **kw
        )

    def get_equations(self):
        from pysph.sph.equation import Group
        from pysph.sph.wc.basic import TaitEOS
        from pysph.sph.basic_equations import SummationDensity

        equations = []
        all = self.fluids + self.solids

        # Add equation of state
        g0 = []
        for name in self.fluids:
            g0.append(TaitEOS(
                dest=name, sources=None, rho0=self.rho0, c0=self.c0, gamma=self.gamma
            ))
        equations.append(Group(equations=g0, name=get_grp_name(g0)))

        
        # Add summation density equation
        g1_1 = []
        for name in all:
            g1_1.append(SummationDensity(dest=name, sources=all))
        equations.append(Group(equations=g1_1, name=get_grp_name(g1_1)))


        # Add equation to copy properties to ghost particles
        from tsph_with_pst import CopyPropsToGhost
        g1_2 = []
        if self.periodic:
            for name in self.fluids:
                g1_2.append(CopyPropsToGhost(dest=name, sources=None))
            equations.append(Group(
                equations=g1_2, real=False, name=get_grp_name(g1_2)))

        
        # Add equation to compute pre-step of Bonet-Lok correction
        from tsph_with_pst import GradientCorrectionPreStepNew
        if self.kernel_corr:
            g2 = []
            for name in all:
                g2.append(GradientCorrectionPreStepNew(
                    dest=name, sources=all, dim=self.dim
                ))
            equations.append(Group(equations=g2, name=get_grp_name(g2)))

        
        # Add equation to compute Bonet-Lok correction and subsequently
        # compute the velocity gradient, and copy it to ghost particles
        from pysph.sph.wc.kernel_correction import GradientCorrection
        from tsph_with_pst import VelocityGradient
        g3_1 = []
        if self.nu > 1e-14:
            if self.kernel_corr:
                for name in all:
                    g3_1.append(GradientCorrection(
                        dest=name, sources=all, dim=self.dim
                    ))
            for name in self.fluids:
                g3_1.append(VelocityGradient(
                    dest=name, sources=self.fluids, dim=self.dim
                ))
            equations.append(Group(equations=g3_1, name=get_grp_name(g3_1)))

            from tsph_with_pst import CopyGradVToGhost
            g3_2 = []
            if self.periodic:
                for name in all:
                    g3_2.append(CopyGradVToGhost(dest=name, sources=None))
                equations.append(Group(
                    equations=g3_2, real=False, name=get_grp_name(g3_2)))

        
        # Add equation to compute the strain rate tensor and Reynolds stress
        # tensor, and subsequently copy them to ghost particles
        g4_1 = []
        for name in self.fluids:
            g4_1.extend([
                StrainRateTensor(dest=name, sources=[name]),
                ReynoldsStressTensor(dest=name, sources=[name]),
            ])
        equations.append(Group(equations=g4_1, name=get_grp_name(g4_1)))
        g4_2 = []
        for name in self.fluids:
            g4_2.append(CopyTensorsToGhost(dest=name, sources=None))
        equations.append(Group(
            equations=g4_2, real=False, name=get_grp_name(g4_2)))

        
        # Add equation to compute the gradient of k and epsilon, and
        # subsequently copy them to ghost particles
        g5_1 = []
        for name in self.fluids:
            g5_1.append(GradKEpsilon(dest=name, sources=[name]))
        equations.append(Group(equations=g5_1, name=get_grp_name(g5_1)))
        g5_2 = []
        for name in self.fluids:
            g5_2.append(CopyGradKEpsilonToGhost(dest=name, sources=None))
        equations.append(Group(
            equations=g5_2, real=False, name=get_grp_name(g5_2)))

        
        # Add equation to compute the Laplacian of k and epsilon, and
        # subsequently copy them to ghost particles
        g6_1 = []
        for name in self.fluids:
            if self.k_eps_expand:
                _eq = LaplacianKEpsilon(dest=name, sources=[name])
            else:
                _eq = ModifiedLaplacianKEpsilon(dest=name, sources=[name])
            g6_1.append(_eq)
        equations.append(Group(equations=g6_1, name=get_grp_name(g6_1)))
        g6_2 = []
        for name in self.fluids:
            g6_2.append(CopyLaplacianKEpsilonToGhost(dest=name, sources=None))
        equations.append(Group(
            equations=g6_2, real=False, name=get_grp_name(g6_2)))

        
        # Add equations to compute the continuity equation, velocity laplacian,
        # and transport equations for momentum, k and epsilon
        from tsph_with_pst import ContinuityEquation
        g7 = []
        if self.kernel_corr:
            for name in self.fluids:
                g7.append(GradientCorrection(
                    dest=name, sources=all, dim=self.dim
                ))
        for name in self.fluids:
            g7.append(ContinuityEquation(dest=name, sources=self.fluids))

        from tsph_with_pst import DivGrad
        for name in self.fluids:
            if self.nu > 1e-14:
                g7.append(DivGrad(
                    dest=name, sources=all, nu=self.nu, rho0=self.rho0
                ))
            g7.append(KEpsilonMomentumEquation(
                dest=name, sources=all, gx=self.gx, gy=self.gy, gz=self.gz
            ))

            if self.k_eps_expand:
                _eqs = [
                    KTransportEquationExpanded(dest=name, sources=[name]),
                    EpsilonTransportEquationExpanded(
                        dest=name, sources=[name]
                    ),
                ]
            else:
                _eqs = [
                    KTransportEquation(dest=name, sources=[name]),
                    EpsilonTransportEquation(dest=name, sources=[name]),
                ]
            g7.extend(_eqs)
        equations.append(Group(equations=g7, name=get_grp_name(g7)))
        
        return equations

    def setup_properties(self, particles, clean=False):
        from pysph.base.utils import get_particle_array_wcsph
        dummy = get_particle_array_wcsph(name='junk')
        output_props = [
            'x', 'y', 'z', 'u', 'v', 'w', 'rho', 'm', 'h', 'pid', 'gid', 'tag',
            'p', 'rhoc', 'gradv', 'k', 'eps', 'lapk', 'lapeps', 'gradk', 
            'gradeps',
        ]

        props = list(dummy.properties.keys()) + [
            'V0'
        ]
        props += [
            'vmax', {
                'name': 'dpos',
                'stride': 3
            }, {
                'name': 'gradrc',
                'stride': 3
            }, {
                'name': 'gradp',
                'stride': 3
            }, 'ki', 'ki0', 'rhoc', 'rhoc0', 'ap', 'p0'
        ]
        props += [
            'k', 'eps',
            dict(name='S', stride=9),
            dict(name='tau', stride=9),
            dict(name='gradk', stride=3),
            dict(name='gradeps', stride=3),
            'lapk', 'lapeps',
            'ak', 'aeps',
            'k0', 'eps0',
        ]
        if len(self.solids) > 0:
            props += ['wf', 'wg', 'ug', 'vf', 'uf', 'vg', 'wij', 'vg_star', 'wg_star', 'ug_star']

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