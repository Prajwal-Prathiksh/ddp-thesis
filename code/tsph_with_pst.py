from pysph.sph.scheme import Scheme
from pysph.sph.equation import Equation
from pysph.sph.integrator import Integrator, IntegratorStep
from pst import IterativePST, ModifiedFickian, DeltaPlusSPHPST, NumberDensityMoment
from pysph.sph.wc.linalg import gj_solve
from compyle.api import declare

from sph_integrators import (
    PECIntegrator, RK2Integrator, RK2Stepper, RK2StepperEDAC
)

class TaitEOS(Equation):
    def __init__(self, dest, sources, rho0, gamma, p0=0.0):
        self.rho0 = rho0
        self.rho01 = 1.0/rho0
        self.gamma = gamma
        self.gamma1 = 0.5*(gamma - 1.0)
        self.p0 = p0

        super(TaitEOS, self).__init__(dest, sources)

    def initialize(self, d_idx, d_rhoc, d_p, d_c0):
        ratio = d_rhoc[d_idx] * self.rho01
        tmp = pow(ratio, self.gamma)

        c0 = d_c0[0]
        B = self.rho0*c0*c0/self.gamma
        d_p[d_idx] = self.p0 + B * (tmp - 1.0)


class LinearEOS(Equation):
    def __init__(self, dest, sources, rho0, gamma=7, p0=0.0):
        self.rho0 = rho0
        self.rho01 = 1.0/rho0
        self.gamma = gamma
        self.gamma1 = 0.5*(gamma - 1.0)
        self.p0 = p0

        super(LinearEOS, self).__init__(dest, sources)

    def initialize(self, d_idx, d_rhoc, d_p, d_c0):
        ratio = d_rhoc[d_idx] * self.rho01
        tmp = ratio

        c0 = d_c0[0]
        B = self.rho0*c0*c0
        d_p[d_idx] = self.p0 + B * (tmp - 1.0)

class IterativePSTNew(IterativePST):
    def post_loop(self, d_idx, d_dpos, d_x, d_y, d_z, d_u, d_v, d_w, d_rho, d_tag, d_gid):

        idx = declare('int')
        if d_tag[d_idx] == 2:
            idx = d_gid[d_idx]
            d_x[d_idx] -= d_dpos[idx * 3]
            d_y[d_idx] -= d_dpos[idx * 3 + 1]
            d_z[d_idx] -= d_dpos[idx * 3 + 2]
        else:
            d_x[d_idx] -= d_dpos[d_idx * 3]
            d_y[d_idx] -= d_dpos[d_idx * 3 + 1]
            d_z[d_idx] -= d_dpos[d_idx * 3 + 2]


class UpdateforPST(Equation):
    def post_loop(self, d_idx, d_x, d_x0, d_y, d_y0, d_z, d_z0, d_rhoc, d_u, d_v, d_w, d_gradv, d_gradrc, d_dpos):

        dx = d_dpos[d_idx * 3]
        dy = d_dpos[d_idx * 3 + 1]
        dz = d_dpos[d_idx * 3 + 2]

        d_u[d_idx] += dx * d_gradv[9*d_idx] + \
            dy * d_gradv[9*d_idx + 1] + \
            dz * d_gradv[9*d_idx + 2]
        d_v[d_idx] += dx * d_gradv[9*d_idx + 3] + \
            dy * d_gradv[9*d_idx + 4] + \
            dz * d_gradv[9*d_idx + 5]
        d_w[d_idx] += dx * d_gradv[9*d_idx + 6] + \
            dy * d_gradv[9*d_idx + 7] + \
            dz * d_gradv[9*d_idx + 8]

        d_rhoc[d_idx] += dx * d_gradrc[3*d_idx] + dy * d_gradrc[3*d_idx + 1] +  dz * d_gradrc[3*d_idx + 2]


class UpdateVelocity(Equation):
    def post_loop(self, d_idx, d_x, d_x0, d_y, d_y0, d_z, d_z0, d_rho, d_u, d_v, d_w, d_gradv):

        dx = d_x0[d_idx] - d_x[d_idx]
        dy = d_y0[d_idx] - d_y[d_idx]
        dz = d_z0[d_idx] - d_z[d_idx]

        d_u[d_idx] -= dx * d_gradv[9*d_idx] +  dy * d_gradv[9*d_idx + 1] +  dz * d_gradv[9*d_idx + 2]
        d_v[d_idx] -= dx * d_gradv[9*d_idx + 3] +  dy * d_gradv[9*d_idx + 4] +  dz * d_gradv[9*d_idx + 5]
        d_w[d_idx] -= dx * d_gradv[9*d_idx + 6] +  dy * d_gradv[9*d_idx + 7] + dz * d_gradv[9*d_idx + 8]


class UpdateDensity(Equation):
    def post_loop(self, d_idx, d_x, d_x0, d_y, d_y0, d_z, d_z0, d_rhoc, d_u, d_v, d_w, d_gradv, d_gradrc, d_dpos):

        dx = d_x0[d_idx] - d_x[d_idx]
        dy = d_y0[d_idx] - d_y[d_idx]
        dz = d_z0[d_idx] - d_z[d_idx]

        d_rhoc[d_idx] -= dx * d_gradrc[3*d_idx] + dy * d_gradrc[3*d_idx + 1] + dz * d_gradrc[3*d_idx + 2]

class UpdatePressure(Equation):
    def post_loop(self, d_idx, d_x, d_x0, d_y, d_y0, d_z, d_z0, d_p, d_u, d_v, d_w, d_gradv, d_gradp, d_dpos):

        dx = d_x0[d_idx] - d_x[d_idx]
        dy = d_y0[d_idx] - d_y[d_idx]
        dz = d_z0[d_idx] - d_z[d_idx]

        d_p[d_idx] -= dx * d_gradp[3*d_idx] + dy * d_gradp[3*d_idx + 1] + \
            dz * d_gradp[3*d_idx + 2]



class SetRho(Equation):
    def initialize(self, d_idx, d_rhoc, d_rho):
        d_rho[d_idx] = d_rhoc[d_idx]

class EvaluateRhoc(Equation):
    def __init__(self, dest, sources, rho0):
        self.rho0 = rho0

        super(EvaluateRhoc, self).__init__(dest, sources)

    def Initialize(self, d_idx, d_rhoc, d_p, d_c0):
        d_rhoc[d_idx] = d_p[d_idx]/d_c0[0]**2 + self.rho0

class GradientCorrectionPreStepNew(Equation):

    def __init__(self, dest, sources, dim=2):
        self.dim = dim
        super(GradientCorrectionPreStepNew, self).__init__(dest, sources)

    def initialize(self, d_idx, d_m_mat):
        i = declare('int')
        for i in range(9):
            d_m_mat[9 * d_idx + i] = 0.0

    def loop_all(self, d_idx, d_m_mat, s_V0, d_x, d_y, d_z, d_h, s_x,
                 s_y, s_z, s_h, SPH_KERNEL, NBRS, N_NBRS, s_m, s_rho):
        x = d_x[d_idx]
        y = d_y[d_idx]
        z = d_z[d_idx]
        h = d_h[d_idx]
        i, j, s_idx, n = declare('int', 4)
        xij = declare('matrix(3)')
        dwij = declare('matrix(3)')
        n = self.dim
        for k in range(N_NBRS):
            s_idx = NBRS[k]
            xij[0] = x - s_x[s_idx]
            xij[1] = y - s_y[s_idx]
            xij[2] = z - s_z[s_idx]
            hij = (h + s_h[s_idx]) * 0.5
            r = sqrt(xij[0] * xij[0] + xij[1] * xij[1] + xij[2] * xij[2])
            SPH_KERNEL.gradient(xij, r, hij, dwij)
            # V = s_V0[s_idx]
            V = s_m[s_idx]/s_rho[s_idx]
            if r > 1.0e-12:
                for i in range(n):
                    for j in range(n):
                        xj = xij[j]
                        d_m_mat[9 * d_idx + 3 * i + j] -= V * dwij[i] * xj


class CopyPropsToGhost(Equation):
    def initialize(self, d_idx, d_tag, d_gid, d_p, d_rhoc, d_rho):
        idx = declare('int')
        if d_tag[d_idx] == 2:
            idx = d_gid[d_idx]
            d_p[d_idx] = d_p[idx]
            d_rhoc[d_idx] = d_rhoc[idx]
            d_rho[d_idx] = d_rho[idx]


class CopyPropsToGhostWithSolid(Equation):
    def initialize(self, d_idx, d_tag, d_gid, d_p, d_rhoc, d_rho, d_ug, d_vg, d_ug_star, d_vg_star):
        idx = declare('int')
        if d_tag[d_idx] == 2:
            idx = d_gid[d_idx]
            d_p[d_idx] = d_p[idx]
            d_rhoc[d_idx] = d_rhoc[idx]
            d_rho[d_idx] = d_rho[idx]
            d_ug[d_idx] = d_ug[idx]
            d_vg[d_idx] = d_vg[idx]
            d_ug_star[d_idx] = d_ug_star[idx]
            d_vg_star[d_idx] = d_vg_star[idx]


class CopyPropsToGhostWithSolidTVF(Equation):
    def initialize(self, d_idx, d_tag, d_gid, d_p, d_rhoc, d_rho, d_ug, d_vg):
        idx = declare('int')
        if d_tag[d_idx] == 2:
            idx = d_gid[d_idx]
            d_p[d_idx] = d_p[idx]
            d_rhoc[d_idx] = d_rhoc[idx]
            d_rho[d_idx] = d_rho[idx]
            d_ug[d_idx] = d_ug[idx]
            d_vg[d_idx] = d_vg[idx]


class CopyRhoToGhost(Equation):
    def initialize(self, d_idx, d_tag, d_gid, d_rho):
        idx = declare('int')
        if d_tag[d_idx] == 2:
            idx = d_gid[d_idx]
            d_rho[d_idx] = d_rho[idx]

class CopyMmatToGhost(Equation):
    def initialize(self, d_idx, d_tag, d_gid, d_m_mat):
        idx, idx9, i = declare('int', 3)
        if d_tag[d_idx] == 2:
            idx = d_gid[d_idx] * 9
            idx9 = d_idx * 9
            for i in range(9):
                d_m_mat[idx9 + i] = d_m_mat[idx + i]


class CopyGradRhoToGhost(Equation):
    def initialize(self, d_idx, d_tag, d_gid, d_gradrc):
        idx3, didx3, i = declare('int', 5)
        if d_tag[d_idx] == 2:
            idx3 = d_gid[d_idx] * 3
            didx3 = d_idx * 3
            for i in range(3):
                d_gradrc[didx3 + i] = d_gradrc[idx3 + i]


class CopyGradVToGhost(Equation):
    def initialize(self, d_idx, d_tag, d_gid, d_gradv):
        idx, idx9, i = declare('int', 3)
        if d_tag[d_idx] == 2:
            idx = d_gid[d_idx] * 9
            idx9 = d_idx * 9
            for i in range(9):
                d_gradv[idx9 + i] = d_gradv[idx + i]

class ContinuityEquation(Equation):
    def initialize(self, d_idx, d_arho):
        d_arho[d_idx] = 0.0

    def loop(self, d_idx, d_arho, s_idx, s_m, DWIJ, VIJ, d_rhoc, s_rho):
        vijdotdwij = DWIJ[0]*VIJ[0] + DWIJ[1]*VIJ[1] + DWIJ[2]*VIJ[2]
        d_arho[d_idx] += s_m[s_idx]*vijdotdwij/s_rho[s_idx] * d_rhoc[d_idx]


class ContinuityEquationSolid(Equation):
    def initialize(self, d_idx, d_arho):
        d_arho[d_idx] = 0.0

    def loop(self, d_idx, d_arho, s_idx, s_m, DWIJ, d_rhoc, s_rho, s_ug_star, s_vg_star, s_wg_star, d_u, d_v, d_w):
        uij = d_u[d_idx] - s_ug_star[s_idx]
        vij = d_v[d_idx] - s_vg_star[s_idx]
        wij = d_w[d_idx] - s_wg_star[s_idx]
        vijdotdwij = DWIJ[0]*uij + DWIJ[1]*vij + DWIJ[2]*wij
        d_arho[d_idx] += s_m[s_idx]*vijdotdwij/s_rho[s_idx] * d_rhoc[d_idx]


class ContinuityEquationOrg(ContinuityEquation):
    def initialize(self, d_idx, d_arho, d_rhoc, d_rho):
        d_arho[d_idx] = 0.0

        d_rhoc[d_idx] = d_rho[d_idx]


class EDACEquation(Equation):
    def __init__(self, dest, sources, rho0, gamma=7):
        self.rho0 = rho0
        self.gamma = gamma

        super(EDACEquation, self).__init__(dest, sources)

    def initialize(self, d_idx, d_ap):
        d_ap[d_idx] = 0.0

    def loop(self, d_idx, d_ap, s_idx, s_m, DWIJ, VIJ, d_rhoc, s_rho, d_c0):

        tmp = self.rho0 * d_c0[0]**2

        vijdotdwij = DWIJ[0]*VIJ[0] + DWIJ[1]*VIJ[1] + DWIJ[2]*VIJ[2]
        d_ap[d_idx] += tmp * s_m[s_idx]*vijdotdwij/s_rho[s_idx]


class EDACEquationOrg(EDACEquation):
    def initialize(self, d_idx, d_ap, d_rhoc, d_rho):
        d_ap[d_idx] = 0.0

        d_rhoc[d_idx] = d_rho[d_idx]


class MomentumEquationSecondOrder(Equation):
    # violeu p 5.131
    def __init__(self, dest, sources,
                 gx=0.0, gy=0.0, gz=0.0, rho0=1.0):

        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.rho0 = rho0
        super(MomentumEquationSecondOrder, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw, d_dt_cfl):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rhoc,
             d_p, d_au, d_av, d_aw, s_m,
             s_rho, s_cs, s_p, VIJ,
             XIJ, HIJ, R2IJ, RHOIJ1, EPS,
             DWIJ, WIJ, WDP, d_dt_cfl, d_m):

        tmp = (s_p[s_idx] - d_p[d_idx])/(d_rhoc[d_idx] * s_rho[s_idx])

        d_au[d_idx] += -s_m[s_idx] * (tmp) * DWIJ[0]
        d_av[d_idx] += -s_m[s_idx] * (tmp) * DWIJ[1]
        d_aw[d_idx] += -s_m[s_idx] * (tmp) * DWIJ[2]

    def post_loop(self, d_idx, d_au, d_av, d_aw, d_dt_force):
        d_au[d_idx] += self.gx
        d_av[d_idx] += self.gy
        d_aw[d_idx] += self.gz


class DensityGradient(Equation):
    def __init__(self, dest, sources, dim=1, rho0=1.0):
        self.dim = dim
        self.rho0 = rho0

        super(DensityGradient, self).__init__(dest, sources)

    def initialize(self, d_idx, d_gradrc):
        i = declare('int')
        for i in range(3):
            d_gradrc[3*d_idx+i] = 0.0

    def loop(self, d_idx, s_idx, s_m, s_rhoc,
             d_gradrc, DWIJ, d_rhoc, s_rho):

        i = declare('int')
        tmp = s_m[s_idx]/s_rho[s_idx]
        rij = s_rhoc[s_idx] - d_rhoc[d_idx]
        for i in range(3):
            d_gradrc[3*d_idx+i] += tmp * rij * DWIJ[i]


class VelocityGradient(Equation):
    def __init__(self, dest, sources, dim=1, rho0=1.0):
        self.dim = dim
        self.rho0 = rho0

        super(VelocityGradient, self).__init__(dest, sources)

    def initialize(self, d_idx, d_gradv):
        i = declare('int')
        for i in range(9):
            d_gradv[9*d_idx+i] = 0.0

    def loop(self, d_idx, s_idx, s_m, s_rho,
             d_gradv, DWIJ, VIJ):

        i, j = declare('int', 2)
        tmp = s_m[s_idx]/s_rho[s_idx]
        for i in range(3):
            for j in range(3):
                d_gradv[9*d_idx+3*i+j] += tmp * -VIJ[i] * DWIJ[j]


class VelocityGradientSoild(Equation):
    def __init__(self, dest, sources, dim=1, rho0=1.0):
        self.dim = dim
        self.rho0 = rho0

        super(VelocityGradientSoild, self).__init__(dest, sources)

    def initialize(self, d_idx, d_gradv):
        i = declare('int')
        for i in range(9):
            d_gradv[9*d_idx+i] = 0.0

    def loop(self, d_idx, s_idx, s_m, s_rho,
             d_gradv, DWIJ, s_ug, s_vg, s_wg, d_u, d_v, d_w):

        i, j = declare('int', 2)
        uji = declare('matrix(3)')
        tmp = s_m[s_idx]/s_rho[s_idx]
        uji[0] = s_ug[s_idx] - d_u[d_idx]
        uji[1] = s_vg[s_idx] - d_v[d_idx]
        uji[2] = s_wg[s_idx] - d_w[d_idx]
        for i in range(3):
            for j in range(3):
                d_gradv[9*d_idx+3*i+j] += tmp * uji[i] * DWIJ[j]


class VelocityGradientDestSoild(Equation):
    def __init__(self, dest, sources, dim=1, rho0=1.0):
        self.dim = dim
        self.rho0 = rho0

        super(VelocityGradientDestSoild, self).__init__(dest, sources)

    def initialize(self, d_idx, d_gradv):
        i = declare('int')
        for i in range(9):
            d_gradv[9*d_idx+i] = 0.0

    def loop(self, d_idx, s_idx, s_m, s_rho,
             d_gradv, DWIJ, s_u, s_v, s_w, d_ug, d_vg, d_wg):

        i, j = declare('int', 2)
        uij = declare('matrix(3)')
        tmp = s_m[s_idx]/s_rho[s_idx]
        uij[0] = d_ug[d_idx] - s_u[s_idx]
        uij[1] = d_vg[d_idx] - s_v[s_idx]
        uij[2] = d_wg[d_idx] - s_w[s_idx]
        for i in range(3):
            for j in range(3):
                d_gradv[9*d_idx+3*i+j] += -tmp * uij[i] * DWIJ[j]


class VelocityGradientSolidSoild(Equation):
    def __init__(self, dest, sources, dim=1, rho0=1.0):
        self.dim = dim
        self.rho0 = rho0

        super(VelocityGradientSolidSoild, self).__init__(dest, sources)

    def initialize(self, d_idx, d_gradv):
        i = declare('int')
        for i in range(9):
            d_gradv[9*d_idx+i] = 0.0

    def loop(self, d_idx, s_idx, s_m, s_rho,
             d_gradv, DWIJ, s_ug, s_vg, s_wg, d_ug, d_vg, d_wg):

        i, j = declare('int', 2)
        uij = declare('matrix(3)')
        tmp = s_m[s_idx]/s_rho[s_idx]
        uij[0] = d_ug[d_idx] - s_ug[s_idx]
        uij[1] = d_vg[d_idx] - s_vg[s_idx]
        uij[2] = d_wg[d_idx] - s_wg[s_idx]
        for i in range(3):
            for j in range(3):
                d_gradv[9*d_idx+3*i+j] += -tmp * uij[i] * DWIJ[j]



class PressureGradient(Equation):
    def __init__(self, dest, sources, dim=1, rho0=1.0):
        self.dim = dim
        self.rho0 = rho0

        super(PressureGradient, self).__init__(dest, sources)

    def initialize(self, d_idx, d_gradp):
        i = declare('int')
        for i in range(3):
            d_gradp[3*d_idx+i] = 0.0

    def loop(self, d_idx, s_idx, s_m, s_rho, s_p,
             d_gradp, DWIJ, d_p):

        i = declare('int')
        tmp = s_m[s_idx]/s_rho[s_idx]
        rij = s_p[s_idx] - d_p[d_idx]
        for i in range(3):
            d_gradp[3*d_idx+i] += tmp * rij * DWIJ[i]


class SaveInitialdistances(Equation):
    def initialize(self, d_idx, d_x, d_y, d_z, d_x0, d_y0, d_z0):
        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]
        d_z0[d_idx] = d_z[d_idx]

class DensityDamping(Equation):
    def __init__(self, dest, sources, gamma=0.1):
        self.gamma = gamma
        super(DensityDamping, self).__init__(dest, sources)

    def initialize(self, d_idx, d_arho):
        d_arho[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_arho, d_gradrc, s_gradrc, DWIJ, s_m, s_rho, d_h, d_c0):
        sidx3, didx3 = declare('int')
        sidx3 = 3*s_idx
        didx3 = 3*d_idx

        tmp = self.gamma * d_h[d_idx] * d_c0[0] * s_m[s_idx]/s_rho[s_idx]

        d_arho[d_idx] += -(
            tmp * (DWIJ[0] * (d_gradrc[didx3] - s_gradrc[sidx3]) + DWIJ[1] *
                   (d_gradrc[didx3 + 1] - s_gradrc[sidx3 + 1]) + DWIJ[2] *
                   (d_gradrc[didx3 + 2] - s_gradrc[sidx3 + 2])))


class PressureDamping(Equation):
    def __init__(self, dest, sources, gamma=0.1):
        self.gamma = gamma
        super(PressureDamping, self).__init__(dest, sources)

    def initialize(self, d_idx, d_ap):
        d_ap[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_ap, d_gradp, s_gradp, DWIJ, s_m, s_rho, d_h, d_c0):
        sidx3, didx3 = declare('int')
        sidx3 = 3*s_idx
        didx3 = 3*d_idx

        tmp = self.gamma * d_h[d_idx] * d_c0[0] * s_m[s_idx]/s_rho[s_idx]

        d_ap[d_idx] += -(
            tmp * (DWIJ[0] * (d_gradp[didx3] - s_gradp[sidx3]) + DWIJ[1] *
                   (d_gradp[didx3 + 1] - s_gradp[sidx3 + 1]) + DWIJ[2] *
                   (d_gradp[didx3 + 2] - s_gradp[sidx3 + 2])))



class DivGrad(Equation):
    def __init__(self, dest, sources, nu=0.0, rho0=1.0):
        self.nu = nu
        self.rho0 = rho0
        super(DivGrad, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_au, d_av, d_aw, s_gradv, d_gradv, DWIJ, d_idx, s_idx, s_m, s_rho, d_rhoc):

        sidx9, didx9 = declare('int')
        tmp = declare('double')
        sidx9 = 9*s_idx
        didx9 = 9*d_idx

        tmp = self.nu * s_m[s_idx]/s_rho[s_idx]

        d_au[d_idx] += -(
            tmp *
            (DWIJ[0] * (d_gradv[didx9] - s_gradv[sidx9]) +
            DWIJ[1] * (d_gradv[didx9 + 1] - s_gradv[sidx9 + 1]) +
            DWIJ[2] * (d_gradv[didx9 + 2] - s_gradv[sidx9 + 2])))

        d_av[d_idx] += -(
            tmp *
            (DWIJ[0] * (d_gradv[didx9 + 3] - s_gradv[sidx9 + 3]) +
            DWIJ[1] * (d_gradv[didx9 + 4] - s_gradv[sidx9 + 4]) +
            DWIJ[2] * (d_gradv[didx9 + 5] - s_gradv[sidx9 + 5])))

        d_aw[d_idx] += -(
            tmp *
            (DWIJ[0] * (d_gradv[didx9 + 6] - s_gradv[sidx9 + 6]) +
            DWIJ[1] * (d_gradv[didx9 + 7] - s_gradv[sidx9 + 7]) +
            DWIJ[2] * (d_gradv[didx9 + 8] - s_gradv[sidx9 + 8])))


class TSPHScheme(Scheme):
    def __init__(
        self, fluids, solids, dim, rho0, c0, h0, hdx, gx=0.0, gy=0.0, gz=0.0,
        nu=0.0, gamma=7.0, kernel_corr=False, pst_freq=0, method='no_sd',
        scm='wcsph', eos='tait', pst='ipst', damp_pre=False,
        periodic=True
    ):
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
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.kernel_corr = kernel_corr
        self.pst_freq = pst_freq
        self.shifter = None
        self.method = 'no_sd'
        self.scm = 'wcsph'
        self.eos = 'tait'
        self.pst ='ipst'
        self.damp_pre = damp_pre
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
            "--pst-freq", action="store", type=int, dest="pst_freq",
            default=None,
            help="Particle shifting frequency"
        )
        group.add_argument(
            '--method', action='store', dest='method', default=None,
            help="method for pressure eval 'sd' or 'no_sd'"
        )

        group.add_argument(
            '--scm', action='store', dest='scm', default='wcsph',
            help="scheme to be used scheme 'wcsph' or 'edac' or 'fatehi'"
        )

        group.add_argument(
            '--eos', action='store', dest='eos', default='tait',
            help="scheme to be used scheme 'tait' or 'linear'"
        )

        group.add_argument(
            '--pst', action='store', dest='pst', default='ipst',
            help="ipst or dppst"
        )

    def consume_user_options(self, options):
        vars = ["kernel_corr", "pst_freq", "method", "scm", "eos", "pst", "damp_pre"]

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
        if self.scm == 'edac':
            step_cls = RK2StepperEDAC
        # step_cls = RK3Stepper
        for name in self.fluids + self.solids:
            if name not in steppers:
                steppers[name] = step_cls()

        integrator = cls(**steppers)
        print(integrator)
        
        from pysph.solver.solver import Solver
        self.solver = Solver(
            dim=self.dim, integrator=integrator, kernel=kernel, **kw
        )

    def get_equations(self):
        from pysph.sph.equation import Group
        from pysph.sph.basic_equations import \
            (SummationDensity)
        # from pysph.sph.wc.basic import TaitEOS
        from pysph.sph.wc.transport_velocity import MomentumEquationArtificialViscosity

        equations = []
        g1 = []
        all = self.fluids + self.solids

        g0 = []
        for name in self.fluids:
            if self.scm == 'edac':
                g0.append(EvaluateRhoc(dest=name, sources=None, rho0=self.rho0))
            if self.scm == 'wcsph' or self.scm == 'fatehi':
                if self.eos == 'tait':
                    g0.append(TaitEOS(
                        dest=name, sources=None, rho0=self.rho0,
                        gamma=self.gamma
                    ))
                elif self.eos == 'linear':
                    g0.append(LinearEOS(
                        dest=name, sources=None, rho0=self.rho0,
                        gamma=self.gamma
                    ))
        equations.append(Group(equations=g0))

        g1 = []
        for name in self.solids:
            from solid_bc import (
                SourceNumberDensity, AdamiPressureBC, AdamiWallVelocity, 
                AdamiSlipWallVelocity
            )
            g1.append(SourceNumberDensity(dest=name, sources=self.fluids))
            g1.append(
                AdamiPressureBC(dest=name,
                                rho0=self.rho0,
                                p0=0.0,
                                sources=self.fluids,
                                gx=self.gx,
                                gy=self.gy,
                                gz=self.gz))
            g1.append(AdamiWallVelocity(dest=name, sources=self.fluids))
            g1.append(AdamiSlipWallVelocity(dest=name, sources=self.fluids))

        for name in all:
            g1.append(SummationDensity(dest=name, sources=all))
        equations.append(Group(equations=g1))

        g1 = []
        if self.periodic:
            for name in self.fluids:
                g1.append(CopyPropsToGhost(dest=name, sources=None))
            for name in self.solids:
                g1.append(CopyPropsToGhostWithSolid(dest=name, sources=None))
            equations.append(Group(equations=g1, real=False))

        if self.kernel_corr:
            from pysph.sph.wc.kernel_correction import GradientCorrectionPreStep, GradientCorrection
            g1 = []
            for name in all:
                g1.append(GradientCorrectionPreStepNew(
                    dest=name, sources=all, dim=self.dim
                ))
            equations.append(Group(equations=g1))

        g2 = []
        if self.nu > 1e-14:
            if self.scm == 'fatehi':
                from scheme_equation import (GradientCorrectionPreStepFatehi,
                                             CreateFatehiOrder2Tensor,
                                             CreateFatehiOrder3Tensor,
                                             CreateFatehiVector,
                                             EvaluateFatehiCorrection)
                for name in self.fluids:
                    g2.extend([
                        GradientCorrectionPreStepFatehi(dest=name,
                                                        sources=all,
                                                        dim=self.dim),
                        CreateFatehiOrder3Tensor(dest=name, sources=all),
                        CreateFatehiOrder2Tensor(dest=name, sources=all),
                        CreateFatehiVector(dest=name, sources=all),
                        EvaluateFatehiCorrection(dest=name,
                                                sources=all,
                                                dim=self.dim),
                    ])

            if self.kernel_corr:
                for name in all:
                    g2.append(GradientCorrection(
                        dest=name, sources=all, dim=self.dim
                    ))
            for name in self.fluids:
                g2.append(VelocityGradient(
                    dest=name, sources=self.fluids, dim=self.dim
                ))
                if len(self.solids) > 0:
                    g2.append(VelocityGradientSoild(
                        dest=name, sources=self.solids, dim=self.dim
                    ))
                if self.damp_pre:
                    if self.scm == 'edac':
                        g2.append(PressureGradient(
                            dest=name, sources=all, dim=self.dim
                        ))
                    elif self.scm == 'wcsph':
                        g2.append(DensityGradient(
                            dest=name, sources=all, dim=self.dim
                        ))
            for name in self.solids:
                g2.append(VelocityGradientDestSoild(
                    dest=name, sources=self.fluids, dim=self.dim
                ))
                g2.append(VelocityGradientSolidSoild(
                    dest=name, sources=self.solids, dim=self.dim
                ))
            equations.append(Group(equations=g2))

            g1 = []
            if self.periodic:
                for name in all:
                    g1.append(CopyGradVToGhost(dest=name, sources=None))
                equations.append(Group(equations=g1, real=False))

        elif self.damp_pre:
            for name in all:
                g2.append(GradientCorrection(
                    dest=name, sources=all, dim=self.dim
                ))
            for name in self.fluids:
                g2.append(DensityGradient(
                    dest=name, sources=all, dim=self.dim
                ))
            equations.append(Group(equations=g2))

            g1 = []
            if self.periodic:
                for name in all:
                    g1.append(CopyGradRhoToGhost(dest=name, sources=None))
                equations.append(Group(equations=g1, real=False))

        g3 = []
        for name in self.fluids:
            if self.nu > 1e-14:
                if self.scm == 'fatehi':
                    from scheme_equation import FatehiViscosityCorrected
                    g3.append(FatehiViscosityCorrected(
                        dest=name, sources=all, nu=self.nu, rho0=self.rho0,
                        dim=self.dim
                    ))
        if self.kernel_corr:
            for name in self.fluids:
                g3.append(GradientCorrection(
                    dest=name, sources=all, dim=self.dim
                ))
        for name in self.fluids:
            if self.scm == 'wcsph' or self.scm == 'fatehi':
                g3.append(ContinuityEquation(dest=name, sources=self.fluids))
                if len(self.solids) > 0:
                    g3.append(ContinuityEquationSolid(dest=name, sources=self.solids))
                if self.damp_pre:
                    g3.append(DensityDamping(dest=name, sources=all, gamma=0.1))
            else:
                g3.append(EDACEquation(dest=name, sources=all, rho0=self.rho0))
                if self.damp_pre:
                    g3.append(PressureDamping(
                        dest=name, sources=all, gamma=0.1
                    ))
        for name in self.fluids:
            if self.nu > 1e-14:
                if not self.scm == 'fatehi':
                    g3.append(DivGrad(
                        dest=name, sources=all, nu=self.nu, rho0=self.rho0
                    ))
            g3.append(MomentumEquationSecondOrder(
                dest=name, sources=all, gx=self.gx, gy=self.gy, gz=self.gz
            ))

        equations.append(Group(equations=g3))

        return equations

    def setup_properties(self, particles, clean=True):
        from pysph.base.utils import get_particle_array_wcsph
        dummy = get_particle_array_wcsph(name='junk')
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
        if len(self.solids) > 0:
            props += ['wf', 'wg', 'ug', 'vf', 'uf', 'vg', 'wij', 'vg_star', 'wg_star', 'ug_star']
        output_props = ['x', 'y', 'z', 'u', 'v', 'w', 'rho', 'm', 'h',
                        'pid', 'gid', 'tag', 'p', 'rhoc', 'gradv']
        if self.kernel_corr:
            delta_sph_props = [
                {'name': 'm_mat', 'stride': 9},
                {'name': 'gradv', 'stride': 9},
            ]
            props += delta_sph_props
        if self.scm == 'fatehi':
            fatehi_props = [
                {'name': 'L1', 'stride': 3},
                {'name': 'L2', 'stride': 9},
                {'name': 'bt', 'stride': 9},
                {'name': 'L', 'stride': 9},
                {'name': 'L3', 'stride': 27},
            ]
            props += fatehi_props

        integrator_name = self.solver.integrator.__class__.__name__
        if 'RK' in integrator_name:
            stepper_props = [
                'xi', 'yi', 'zi', 'ui', 'vi', 'wi', 'rhoi', 'rhoci'
            ]
            props += stepper_props

        for pa in particles:
            self._ensure_properties(pa, props, clean)
            pa.set_output_arrays(output_props)
            pa.add_constant('maxki0', 0.0)
            pa.add_constant('maxki', 0.0)


    def post_step_dppst(self, pa_arr, domain):
        if self.shifter is None:
            from pysph.tools.sph_evaluator import SPHEvaluator
            from pysph.sph.wc.kernel_correction import GradientCorrection, GradientCorrectionPreStep
            from pst import ModifiedFickian, IterativePST, NumberDensityMoment
            from pysph.sph.equation import Group
            from pysph.sph.basic_equations import \
                (SummationDensity)

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
            equations.append(Group(equations=g1))

            g2 = []
            for name in self.fluids:
                g2.append(GradientCorrection(dest=name, sources=all, dim=self.dim))
                g2.append(VelocityGradient(dest=name, sources=all, dim=self.dim))
                if self.scm == 'edac':
                    g2.append(PressureGradient(dest=name, sources=all, dim=self.dim))
                else:
                    g2.append(DensityGradient(dest=name, sources=all, dim=self.dim))
            equations.append(Group(equations=g2))

            g3 = []
            for name in self.fluids:
                g3.append(DeltaPlusSPHPST(dest=name, sources=all, hdx=self.hdx))
            for name in self.fluids:
                g3.append(UpdateforPST(dest=name, sources=None))
                # if self.scm == 'edac':
                #     g3.append(UpdatePressure(dest=name, sources=None))
                # else:
                #     g3.append(UpdateDensity(dest=name, sources=None))
            equations.append(Group(equations=g3))


            # print(equations, pa_arr)

            self.shifter = SPHEvaluator(
                arrays=pa_arr, equations=equations, dim=self.dim,
                kernel=self.solver.kernel, backend='cython'
            )

        else:
            self.shifter.update()
            self.shifter.evaluate(t=self.solver.t, dt=self.solver.dt)

    def post_step_ipst(self, pa_arr, domain):
        if self.shifter is None:
            from pysph.tools.sph_evaluator import SPHEvaluator
            from pysph.sph.wc.kernel_correction import GradientCorrection, GradientCorrectionPreStep
            from pst import ModifiedFickian, IterativePST, NumberDensityMoment
            from pysph.sph.equation import Group
            from pysph.sph.basic_equations import \
                (SummationDensity)

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

            # g0 = []
            # for name in all:
            #     g0.append(CopyMmatToGhost(dest=name, sources=all))
            # equations.append(Group(equations=g0, real=False))


            g2 = []
            for name in self.fluids:
                g2.append(GradientCorrection(dest=name, sources=all, dim=self.dim))
                g2.append(VelocityGradient(dest=name, sources=all, dim=self.dim))
                if self.scm == 'edac':
                    g2.append(PressureGradient(dest=name, sources=all, dim=self.dim))
                else:
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
                if self.scm == 'edac':
                    g4.append(UpdatePressure(dest=name, sources=None))
                else:
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

    def post_step(self, pa_arr, domain):
        if self.pst == 'ipst':
            self.post_step_ipst(pa_arr, domain)
        elif self.pst == 'dppst':
            self.post_step_dppst(pa_arr, domain)