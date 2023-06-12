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
from pysph.sph.equation import Equation
from pysph.sph.scheme import Scheme
from compyle.api import declare
from math import pi, sin, cos, sqrt

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

class CalculateShiftVelocity(Equation):
    def __init__(
        self, dest, sources, hdx, prob_l, Ma, c0, Umax, xhi=0.2, n=4.
    ):
        self.hdx = hdx
        self.xhi = xhi
        self.n = n
        self.fac = prob_l*Ma*c0
        self.Umax = Umax
        super().__init__(dest, sources)
    
    def initialize(self, d_idx, d_du, d_dv, d_dw):
        d_du[d_idx] = 0.0
        d_dv[d_idx] = 0.0
        d_dw[d_idx] = 0.0
    
    def loop(
        self, d_idx, d_h, d_du, d_dv, d_dw, s_idx, s_m, s_rho,
        DWIJ, WIJ, SPH_KERNEL
    ):
        hi = d_h[d_idx]
        dx = hi / self.hdx
        wdx = SPH_KERNEL.kernel([0, 0, 0], dx, d_h[d_idx])
        Vj = s_m[s_idx] / s_rho[s_idx]
        
        tmp = (WIJ / wdx)**self.n
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

class VelocityGradient(Equation):
    def __init__(self, dest, sources, dim=1, rho0=1.0):
        self.dim = dim
        self.rho0 = rho0

        super(VelocityGradient, self).__init__(dest, sources)

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

class ContinuityEquationLESSPH(Equation):
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

class MomentumEquationLESSPH(Equation):
    def __init__(
        self, dest, sources, dim, rho0, mu, prob_l, C_S=0.12,
        tensile_correction=False
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
        term_1, term_2, term_3, term_4 = declare('matrix(3)', 4)
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