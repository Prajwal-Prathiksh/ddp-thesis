#NOQA
r"""
:math:`\delta` LES-SPH
Author: K T Prajwal Prathiksh
###
References
-----------
    .. [Colagrossi2021] A. Colagrossi, “Smoothed particle hydrodynamics method
    from a large eddy simulation perspective . Generalization to a
    quasi-Lagrangian model Smoothed particle hydrodynamics method from a
    large eddy simulation perspective . Generalization to a quasi-Lagrangian
    model,” vol. 015102, no. December 2020, 2021, doi: 10.1063/5.0034568.
"""
###########################################################################
# Import
###########################################################################
from math import sqrt

from compyle.api import declare
from k_eps import get_grp_name
from pst import NumberDensityMoment
from pysph.base.kernels import WendlandQuinticC4
from pysph.base.utils import get_particle_array_wcsph
from pysph.solver.solver import Solver
from pysph.sph.basic_equations import SummationDensity
from pysph.sph.equation import Equation, Group
from pysph.sph.integrator_step import IntegratorStep
from pysph.sph.scheme import Scheme
from pysph.sph.wc.kernel_correction import (GradientCorrection,
                                            GradientCorrectionPreStep)
from pysph.tools.sph_evaluator import SPHEvaluator
from sph_integrators import RK2Integrator
from tsph_with_pst import (CopyRhoToGhost, DensityGradient,
                           GradientCorrectionPreStepNew, IterativePSTNew,
                           SaveInitialdistances, UpdateDensity, UpdateVelocity, CopyPropsToGhost)


###########################################################################
# Equations
###########################################################################
class LinearEOS(Equation):
    def __init__(self, dest, sources, c0, rho0):
        self.c0 = c0
        self.rho0 = rho0

        super().__init__(dest, sources)

    def initialize(self, d_p, d_idx, d_rho):
        d_p[d_idx] = self.c0**2 * (d_rho[d_idx] - self.rho0)

class CalculateShiftVelocity(Equation):
    def __init__(
        self, dest, sources, hdx, prob_l, Ma, c0, Umax, xhi=0.2,
        shiftvel_exp=4.
    ):
        self.hdx = hdx
        self.xhi = xhi
        self.shiftvel_exp = shiftvel_exp
        self.fac = prob_l*Ma*c0
        self.Umax = Umax

        super().__init__(dest, sources)
    
    def initialize(self, d_idx, d_du, d_dv, d_dw):
        d_du[d_idx] = 0.0
        d_dv[d_idx] = 0.0
        d_dw[d_idx] = 0.0
    
    def loop(
        self, d_idx, s_idx,
        d_h, d_du, d_dv, d_dw,
        s_m, s_rho,
        DWIJ, WIJ, SPH_KERNEL
    ):
        hi = d_h[d_idx]
        dx = hi / self.hdx
        wdx = SPH_KERNEL.kernel([0, 0, 0], dx, d_h[d_idx])
        Vj = s_m[s_idx] / s_rho[s_idx]
        
        tmp = (WIJ / wdx)**self.shiftvel_exp
        fac = 1. + self.xhi * tmp
        
        d_du[d_idx] += fac * Vj * DWIJ[0]
        d_dv[d_idx] += fac * Vj * DWIJ[1]
        d_dw[d_idx] += fac * Vj * DWIJ[2]

    def post_loop(self, d_idx, d_du, d_dv, d_dw):
        d_du[d_idx] *= - self.fac
        d_dv[d_idx] *= - self.fac
        d_dw[d_idx] *= - self.fac

        du, dv, dw = d_du[d_idx], d_dv[d_idx], d_dw[d_idx]
        dvmag = (du**2 + dv**2 + dw**2)**0.5
        
        fac = min(dvmag, self.Umax*0.5)/dvmag

        d_du[d_idx] = fac * du
        d_dv[d_idx] = fac * dv
        d_dw[d_idx] = fac * dw

class CopydUsToGhost(Equation):
    def initialize(self, d_idx, d_tag, d_gid, d_du, d_dv, d_dw):
        idx = declare('int')
        if d_tag[d_idx] == 2:
            idx = d_gid[d_idx]
            d_du[d_idx] = d_du[idx]
            d_dv[d_idx] = d_dv[idx]
            d_dw[d_idx] = d_dw[idx]

class VelocityGradient(Equation):
    def initialize(self, d_idx, d_gradv):
        i = declare('int')
        for i in range(9):
            d_gradv[9*d_idx+i] = 0.0

    def loop(self, d_idx, s_idx, d_gradv, s_m, s_rho, DWIJ, VIJ):

        i, j = declare('int', 2)
        tmp = s_m[s_idx]/s_rho[s_idx]
        for i in range(3):
            for j in range(3):
                d_gradv[9*d_idx+3*i+j] += tmp * -VIJ[i] * DWIJ[j]

class StrainRateTensor(Equation):
    def initialize(self, d_idx, d_gradv, d_S):
        didx9 = declare('int')
        didx9 = 9*d_idx

        i, j = declare('int', 2)
        for i in range(9):
            # Calculate index of transpose
            j = i//3 + (i%3)*3
            d_S[didx9+i] = 0.5*(d_gradv[didx9+i] + d_gradv[didx9+j])

class CopyTensorsToGhost(Equation):
    def initialize(self, d_idx, d_tag, d_gid, d_S):
        idx, idx9, i = declare('int', 3)
        if d_tag[d_idx] == 2:
            idx = d_gid[d_idx] * 9
            idx9 = d_idx * 9
            for i in range(9):
                d_S[idx9 + i] = d_S[idx + i]

class CalculatePressureGradient(Equation):
    def initialize(self, d_idx, d_gradp):
        i, didx3 = declare('int', 2)
        didx3 = 3 * d_idx
        for i in range(3):
            d_gradp[didx3 + i] = 0.0
    
    def loop(self, d_idx, s_idx, d_rhoc, d_gradp, s_m, s_rho, DWIJ):
        Vj = s_m[s_idx] / s_rho[s_idx]
        fac = Vj * (s_rho[s_idx] - d_rhoc[d_idx])

        i = declare('int')
        for i in range(3):
            d_gradp[3*d_idx + i] += fac * DWIJ[i]

class CopyGradVAandPToGhost(Equation):
    def initialize(self, d_idx, d_tag, d_gid, d_gradv, d_gradp):
        ridx, idx9, idx3, i = declare('int', 4)
        if d_tag[d_idx] == 2:
            ridx = d_gid[d_idx] * 9
            idx9 = d_idx * 9
            for i in range(9):
                d_gradv[idx9 + i] = d_gradv[ridx + i]

            ridx = d_gid[d_idx] * 3
            idx3 = d_idx * 3
            for i in range(3):
                d_gradp[idx3 + i] = d_gradp[ridx + i]

class ContinuityEquationLES(Equation):
    def __init__(self, dest, sources, prob_l, C_delta=6.0):
        self.prob_l = prob_l
        self.C_delta = C_delta
        self.nu_fac = (C_delta*prob_l)**2

        super().__init__(dest, sources)

    def initialize(self, d_idx, d_arho):
        d_arho[d_idx] = 0.0
    
    def loop(
        self, d_idx, s_idx,
        d_u, d_v, d_w, d_du, d_dv, d_dw, d_rhoc, d_S, d_gradp, d_arho,
        s_u, s_v, s_w, s_du, s_dv, s_dw, s_rhoc, s_S, s_gradp, s_m, s_rho,  
        DWIJ, R2IJ, XIJ, EPS
    ):
        uhati = d_u[d_idx] + d_du[d_idx]
        vhati = d_v[d_idx] + d_dv[d_idx]
        whati = d_w[d_idx] + d_dw[d_idx]

        uhatj = s_u[s_idx] + s_du[s_idx]
        vhatj = s_v[s_idx] + s_dv[s_idx]
        whatj = s_w[s_idx] + s_dw[s_idx]

        duhatji = uhatj - uhati
        dvhatji = vhatj - vhati
        dwhatji = whatj - whati

        dui = d_du[d_idx]
        dvi = d_dv[d_idx]
        dwi = d_dw[d_idx]

        duj = s_du[s_idx]
        dvj = s_dv[s_idx]
        dwj = s_dw[s_idx]

        rhoi = d_rhoc[d_idx]
        rhoj = s_rhoc[s_idx]
        rhoji = rhoj - rhoi
        Vj = s_m[s_idx] / s_rho[s_idx]

        # Calculate Frobenius norm of strain rate tensors
        i, didx9 = declare('int', 2)
        didx9 = 9*d_idx
        Smagi, Smagj = declare('double', 2)
        Smagj = 0.0
        Smagi = 0.0

        for i in range(9):
            Smagi += d_S[didx9+i]*d_S[didx9+i]
            Smagj += s_S[didx9+i]*s_S[didx9+i]
        Smagi = sqrt(2*Smagi)
        Smagj = sqrt(2*Smagj)

        nui = self.nu_fac * Smagi
        nuj = self.nu_fac * Smagj
        deltaij = 2.0 * nui * nuj / (nui + nuj)

        gradpdotxij = 0.0
        for i in range(3):
            temp = d_gradp[3*d_idx + i] + s_gradp[3*s_idx + i]
            gradpdotxij += temp * -XIJ[i]
        psi_fac = (rhoji - 0.5 * gradpdotxij)/(R2IJ + EPS)
        psiij = declare('matrix(3)')
        for i in range(3):
            psiij[i] = psi_fac * -XIJ[i]

        # Calculate the change in density
        duhatdotdwij = duhatji*DWIJ[0] + dvhatji*DWIJ[1] + dwhatji*DWIJ[2]
        term_1 = -rhoi * duhatdotdwij * Vj

        rhodudotdwij = (rhoj*duj + rhoi*dui)*DWIJ[0] +\
            (rhoj*dvj + rhoi*dvi)*DWIJ[1] + (rhoj*dwj + rhoi*dwi)*DWIJ[2]
        term_2 = rhodudotdwij * Vj

        psiijdotdwij = psiij[0]*DWIJ[0] + psiij[1]*DWIJ[1] + psiij[2]*DWIJ[2]
        term_3 = deltaij * psiijdotdwij * Vj

        d_arho[d_idx] += term_1 + term_2 + term_3

class MomentumEquationLES(Equation):
    def __init__(
        self, dest, sources, dim, rho0, mu, prob_l, C_S=0.12,
        tensile_correction=True
    ):
        self.dim = dim
        self.K = 2 * (dim + 2)
        self.rho0 = rho0
        self.mu = mu
        self.alpha_term = mu / rho0
        self.prob_l = prob_l
        self.C_S = C_S
        self.nu_fac = (C_S*prob_l)**2
        self.tensile_correction = tensile_correction

        super().__init__(dest, sources)
        
    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0
    
    def loop(
        self, d_idx, s_idx,
        d_u, d_v, d_w, d_du, d_dv, d_dw, d_rhoc, d_S, d_p, d_au, d_av, d_aw,
        s_u, s_v, s_w, s_du, s_dv, s_dw, s_rhoc, s_S, s_p, s_m, s_rho, 
        DWIJ, R2IJ, XIJ, VIJ, EPS
    ):  
        ui = d_u[d_idx]
        vi = d_v[d_idx]
        wi = d_w[d_idx]

        uj = s_u[s_idx]
        vj = s_v[s_idx]
        wj = s_w[s_idx]

        dui = d_du[d_idx]
        dvi = d_dv[d_idx]
        dwi = d_dw[d_idx]

        duj = s_du[s_idx]
        dvj = s_dv[s_idx]
        dwj = s_dw[s_idx]

        rhoi = d_rhoc[d_idx]
        rhoj = s_rhoc[s_idx]
        rhoji = rhoj - rhoi
        Vj = s_m[s_idx] / s_rho[s_idx]

        Pi = d_p[d_idx]
        Pj = s_p[s_idx]

        Pij = 0.0
        if self.tensile_correction and Pi <= 1e-14:
            Pij = Pj - Pi
        else:
            Pij = Pj + Pi

        # Calculate Frobenius norm of strain rate tensors
        i, didx9 = declare('int', 2)
        didx9 = 9*d_idx
        Smagi, Smagj = declare('double', 2)
        Smagj = 0.0
        Smagi = 0.0

        for i in range(9):
            Smagi += d_S[didx9+i]*d_S[didx9+i]
            Smagj += s_S[didx9+i]*s_S[didx9+i]
        Smagi = sqrt(2*Smagi)
        Smagj = sqrt(2*Smagj)

        nui = self.nu_fac * Smagi
        nuj = self.nu_fac * Smagj
        alphaij = self.alpha_term + (2 * nui * nuj / (nui + nuj))
        vjidotxji = VIJ[0]*XIJ[0] + VIJ[1]*XIJ[1] + VIJ[2]*XIJ[2]
        piij = vjidotxji / (R2IJ + EPS)

        tensordotdwij = declare('matrix(3)')
        tensordotdwij[0] = (uj*duj + ui*dui)*DWIJ[0] +\
            (uj*dvj + ui*dvi)*DWIJ[1] + (uj*dwj + ui*dwi)*DWIJ[2]
        tensordotdwij[1] = (vj*duj + vi*dui)*DWIJ[0] +\
            (vj*dvj + vi*dvi)*DWIJ[1] + (vj*dwj + vi*dwi)*DWIJ[2]
        tensordotdwij[2] = (wj*duj + wi*dui)*DWIJ[0] +\
            (wj*dvj + wi*dvi)*DWIJ[1] + (wj*dwj + wi*dwi)*DWIJ[2]
        
        dujidotdwij = (duj - dui)*DWIJ[0] + (dvj - dvi)*DWIJ[1] +\
            (dwj - dwi)*DWIJ[2]

        # Calculate accelerations
        term_1 = declare('matrix(3)')
        term_2 = declare('matrix(3)')
        term_3 = declare('matrix(3)')
        term_4 = declare('matrix(3)')
        
        rho0i = self.rho0 / rhoi

        term_1_fac = -(1/rhoi) * Pij * Vj
        for i in range(3):
            term_1[i] = term_1_fac * DWIJ[i]
        
        term_2_fac = rho0i * self.K * alphaij * piij * Vj
        for i in range(3):
            term_2[i] = term_2_fac * DWIJ[i]

        term_3_fac = rho0i * Vj
        for i in range(3):
            term_3[i] = term_3_fac * tensordotdwij[i]
        
        term_4_fac = -rho0i * dujidotdwij * Vj
        term_4[0] = term_4_fac * ui
        term_4[1] = term_4_fac * vi
        term_4[2] = term_4_fac * wi

        d_au[d_idx] += term_1[0] + term_2[0] + term_3[0] + term_4[0]
        d_av[d_idx] += term_1[1] + term_2[1] + term_3[1] + term_4[1]
        d_aw[d_idx] += term_1[2] + term_2[2] + term_3[2] + term_4[2]

###########################################################################
# Integrator Step
###########################################################################
class DeltaLESRK2Step(IntegratorStep):
    def initialize(
        self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
        d_u0, d_v0, d_w0, d_u, d_v, d_w,
        d_rhoc, d_rhoc0
    ):
        # Initialise `U^{n}`
        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]
        d_z0[d_idx] = d_z[d_idx]

        d_u0[d_idx] = d_u[d_idx]
        d_v0[d_idx] = d_v[d_idx]
        d_w0[d_idx] = d_w[d_idx]

        d_rhoc0[d_idx] = d_rhoc[d_idx]

    def stage1(
        self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
        d_u0, d_v0, d_w0, d_u, d_v, d_w, d_du, d_dv, d_dw, d_au, d_av, d_aw, 
        d_rhoc, d_arho, d_rhoc0,
        dt
    ):
        dtb2 = 0.5*dt

        # Compute `U^{1}`
        d_x[d_idx] = d_x0[d_idx] + dtb2*(d_u[d_idx] + d_du[d_idx])
        d_y[d_idx] = d_y0[d_idx] + dtb2*(d_v[d_idx] + d_dv[d_idx])
        d_z[d_idx] = d_z0[d_idx] + dtb2*(d_w[d_idx] + d_dw[d_idx])

        d_u[d_idx] = d_u0[d_idx] + dtb2*d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dtb2*d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dtb2*d_aw[d_idx]

        d_rhoc[d_idx] = d_rhoc0[d_idx] + dtb2*d_arho[d_idx]

    def stage2(
        self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
        d_u0, d_v0, d_w0, d_u, d_v, d_w, d_du, d_dv, d_dw, d_au, d_av, d_aw,
        d_rhoc, d_arho, d_rhoc0,
        dt
    ):
        # Compute `U^{n+1}`
        d_x[d_idx] = d_x0[d_idx] + dt*(d_u[d_idx] + d_du[d_idx])
        d_y[d_idx] = d_y0[d_idx] + dt*(d_v[d_idx] + d_dv[d_idx])
        d_z[d_idx] = d_z0[d_idx] + dt*(d_w[d_idx] + d_dw[d_idx])

        d_u[d_idx] = d_u0[d_idx] + dt*d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dt*d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dt*d_aw[d_idx]

        d_rhoc[d_idx] = d_rhoc0[d_idx] + dt*d_arho[d_idx]


###########################################################################
# Scheme
###########################################################################
class DeltaLESScheme(Scheme):
    def __init__(
        self, fluids, solids,
        dim, rho0, c0, h0, hdx, prob_l, Ma, Umax, nu,
        xhi=0.2, shiftvel_exp=4., C_delta=6., C_S=0.12,
        tensile_correction=True, pst=True, pst_freq=10
    ):
        self.fluids = fluids
        self.solids = solids
        
        self.dim = dim
        self.rho0 = rho0
        self.c0 = c0
        self.h0 = h0
        self.hdx = hdx
        self.prob_l = prob_l
        self.Ma = Ma
        self.Umax = Umax
        self.nu = nu
        self.xhi = xhi
        self.shiftvel_exp = shiftvel_exp
        self.C_delta = C_delta
        self.C_S = C_S
        self.tensile_correction = tensile_correction
        self.pst = pst
        self.pst_freq = pst_freq
        self.shifter = None

    def add_user_options(self, group):
        group.add_argument(
            '--les-xhi', action='store', type=float, dest='xhi',
            default=0.2, help='LES parameter xhi'
        )
        group.add_argument(
            '--les-shiftvel-exp', action='store', type=float,
            dest='shiftvel_exp', default=4.,
            help='LES parameter shiftvel_exp'
        )
        group.add_argument(
            '--les-cdelta', action='store', type=float, dest='C_delta',
            default=6., help='LES parameter C_delta'
        )
        group.add_argument(
            '--les-cs', action='store', type=float, dest='C_S',
            default=0.12, help='LES parameter C_S'
        )
        group.add_argument(
            '--les-no-tc', action='store_false',
            dest='tensile_correction', help='Disable tensile correction'
        )
        group.add_argument(
            '--les-no-pst', action='store_false', dest='pst',
            help='Disable PST'
        )
        group.add_argument(
            '--pst-freq', action='store', type=int, dest='pst_freq',
            default=10, help='PST frequency'
        )
    
    def consume_user_options(self, options):
        vars = [
            'xhi', 'shiftvel_exp', 'C_delta', 'C_S', 'tensile_correction',
            'pst', 'pst_freq'
        ]
        data = dict(
            (var, self._smart_getattr(options, var)) for var in vars
        )
        self.configure(**data)

    def get_timestep(self, cfl=1.5):
        return cfl*self.h0/self.c0
    
    def configure_solver(
        self, kernel=None, integrator_cls=None, extra_steppers=None, **kw
    ):
        if kernel is None:
            kernel = WendlandQuinticC4(dim=self.dim)

        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        cls = integrator_cls if integrator_cls is not None else RK2Integrator
        step_cls = DeltaLESRK2Step
        for name in self.fluids + self.solids:
            if name not in steppers:
                steppers[name] = step_cls()
        
        integrator = cls(**steppers)


        if 'dt' not in kw:
            kw['dt'] = self.get_timestep()
        self.solver = Solver(
            dim=self.dim, integrator=integrator, kernel=kernel, **kw
        )

    def get_equations(self):
        self.mu = self.nu * self.rho0
        equations = []
        all = self.fluids + self.solids

        # Add equation of state 
        g0 = []
        for name in self.fluids:
            g0.append(
                LinearEOS(dest=name, sources=None, c0=self.c0, rho0=self.rho0)
            )
        equations.append(Group(equations=g0, name=get_grp_name(g0)))


        # Add summation density equation and copy properties to ghost particles
        g1_1 = []
        for name in all:
            g1_1.append(SummationDensity(dest=name, sources=all))
        equations.append(Group(equations=g1_1, name=get_grp_name(g1_1)))

        g1_2 = []
        for name in self.fluids:
            g1_2.append(CopyPropsToGhost(dest=name, sources=None))
        equations.append(Group(
            equations=g1_2, real=False, name=get_grp_name(g1_2)))


        # Add equation to compute the shifted velocities and copy them to ghost
        # particles
        g2_1 = []
        for name in self.fluids:
            g2_1.append(
                CalculateShiftVelocity(
                    dest=name, sources=self.fluids, hdx=self.hdx,
                    prob_l=self.prob_l, Ma=self.Ma, c0=self.c0, Umax=self.Umax, xhi=self.xhi, shiftvel_exp=self.shiftvel_exp
                )
            )
        equations.append(Group(equations=g2_1, name=get_grp_name(g2_1)))

        g2_2 = []
        for name in self.fluids:
            g2_2.append(CopydUsToGhost(dest=name, sources=None))
        equations.append(Group(
            equations=g2_2, real=False, name=get_grp_name(g2_2)))

        
        # Add equation to compute pre-step of Bonet-Lok correction
        g2 = []
        for name in all:
            g2.append(GradientCorrectionPreStepNew(
                dest=name, sources=all, dim=self.dim
            ))
        equations.append(Group(equations=g2, name=get_grp_name(g2)))


        # Add equation to compute Bonet-Lok correction, and subsequnetly 
        # compute the velocity gradient and pressure gradient
        g3_1 = []
        for name in all:
            g3_1.append(GradientCorrection(
                dest=name, sources=all, dim=self.dim
            ))
        for name in self.fluids:
            g3_1.append(VelocityGradient(dest=name, sources=self.fluids))
            g3_1.append(CalculatePressureGradient(
                dest=name, sources=self.fluids))
        equations.append(Group(equations=g3_1, name=get_grp_name(g3_1)))


        # Add equation to copy the velocity and pressure gradient to ghost 
        # particles
        g3_2 = []
        for name in all:
            g3_2.append(CopyGradVAandPToGhost(dest=name, sources=None))
        equations.append(Group(
            equations=g3_2, real=False, name=get_grp_name(g3_2)))


        # Add equation to compute the strain rate tensor and subsequently copy 
        # them to ghost particles
        g4_1 = []
        for name in self.fluids:
            g4_1.extend([
                StrainRateTensor(dest=name, sources=[name]),
            ])
        equations.append(Group(equations=g4_1, name=get_grp_name(g4_1)))
    
        g4_2 = []
        for name in self.fluids:
            g4_2.append(CopyTensorsToGhost(dest=name, sources=None))
        equations.append(Group(
            equations=g4_2, real=False, name=get_grp_name(g4_2)))

        
        # Add equation to compute the continuity equation
        g5 = []
        for name in self.fluids:
            g5.append(ContinuityEquationLES(
                dest=name, sources=all, prob_l=self.prob_l,
                C_delta=self.C_delta
            ))
        equations.append(Group(equations=g5, name=get_grp_name(g5)))

        
        # Add equation to compute the momentum equation
        g6 = []
        for name in self.fluids:
            g6.append(MomentumEquationLES(
                dest=name, sources=all, dim=self.dim, rho0=self.rho0,
                mu=self.mu, prob_l=self.prob_l, C_S=self.C_S,
                tensile_correction=self.tensile_correction
            ))
        equations.append(Group(equations=g6, name=get_grp_name(g6)))

        return equations

    def post_step(self, pa_arr, domain):
        if not self.pst:
            return
        
        if self.shifter is None:
            equations = []
            all = self.fluids + self.solids

            g0 = []
            for name in all:
                g0.append(SummationDensity(dest=name, sources=all))
            equations.append(Group(equations=g0))


            g0 = []
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
                g2.append(VelocityGradient(dest=name, sources=all))
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
            print('PST Equations:')
            print(equations, pa_arr)
            print(f"PST Frequency: {self.pst_freq}")

            self.shifter = SPHEvaluator(
                arrays=pa_arr, equations=equations, dim=self.dim,
                kernel=self.solver.kernel, backend='cython'
            )

        if self.pst_freq > 0 and self.solver.count % self.pst_freq == 0:
            self.shifter.update()
            self.shifter.evaluate(t=self.solver.t, dt=self.solver.dt)
    
    def setup_properties(self, particles, clean=True):
        dummy = get_particle_array_wcsph(name='junk')
        output_props = [
            'x', 'y', 'z',
            'u', 'v', 'w', 'du', 'dv', 'dw',
            'rho', 'm', 'h', 'p', 'rhoc',
            'gradv', 'gradp', 'S',
            'pid', 'gid', 'tag',
        ]

        props = list(dummy.properties.keys()) + [
            'V0',
            'du', 'dv', 'dw',
            'rhoc', 'rhoc0', 'ap', 'p0',
            dict(name='gradv', stride=9),
            dict(name='m_mat', stride=9),
            dict(name='gradp', stride=3),
            dict(name='gradrc', stride=3),
            dict(name='S', stride=9),
        ]
        props += [
            'vmax', 
            'ki', 'ki0', 
            dict(name='dpos', stride=3),
        ]

        for pa in particles:
            self._ensure_properties(pa, props, clean)
            pa.set_output_arrays(output_props)
            pa.add_constant('maxki0', 0.0)
            pa.add_constant('maxki', 0.0)