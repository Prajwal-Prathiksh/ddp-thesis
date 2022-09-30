from pysph.sph.scheme import Scheme
from pysph.sph.equation import Equation
from pysph.sph.integrator import Integrator, IntegratorStep
from pst import IterativePST, ModifiedFickian, DeltaPlusSPHPST
from pysph.sph.wc.linalg import gj_solve
from compyle.api import declare
from pysph.sph.integrator import EulerIntegrator

class RK2Integrator(Integrator):
    def one_timestep(self, t, dt):
        # Initialise `U^{n}`
        self.initialize()

        # Stage 1 - Compute and store `U^{1}`
        self.compute_accelerations()
        self.stage1()
        self.update_domain()
        # Call any post-stage functions
        self.do_post_stage(0.5*dt, 1)

        # Stage 2 - Compute and store `U^{n+1}`
        self.compute_accelerations()
        self.stage2()
        self.update_domain()
        # Call any post-stage functions
        self.do_post_stage(dt, 2)

### EulerStep
class EulerStepNew(IntegratorStep):
    def stage1(
        self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
        d_u0, d_v0, d_w0, d_u, d_v, d_w, d_au, d_av, d_aw,
        dt, d_rhoc, d_arho, d_rhoc0
    ):
        # Compute `U^{1}`
        d_x[d_idx] += dt*d_u[d_idx]
        d_y[d_idx] += dt*d_v[d_idx]
        d_z[d_idx] += dt*d_w[d_idx]

        d_u[d_idx] += dt*d_au[d_idx]
        d_v[d_idx] += dt*d_av[d_idx]
        d_w[d_idx] += dt*d_aw[d_idx]

        d_rhoc[d_idx] += dt*d_arho[d_idx]

### Runge-Kutta Second-Order Integrator Step-------------------------------
class RK2Stepper(IntegratorStep):
    def initialize(
        self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
        d_u0, d_v0, d_w0, d_u, d_v, d_w, d_rhoc, d_rhoc0
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
        d_u0, d_v0, d_w0, d_u, d_v, d_w, d_au, d_av, d_aw,
        dt, d_rhoc, d_arho, d_rhoc0
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

    def stage2(
        self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
        d_u0, d_v0, d_w0, d_u, d_v, d_w, d_au, d_av, d_aw,
        dt, d_rhoc, d_arho, d_rhoc0
    ):
        # Compute `U^{n+1}`
        d_x[d_idx] = d_x0[d_idx] + dt*d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dt*d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dt*d_w[d_idx]

        d_u[d_idx] = d_u0[d_idx] + dt*d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dt*d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dt*d_aw[d_idx]

        d_rhoc[d_idx] = d_rhoc0[d_idx] + dt*d_arho[d_idx]



class RK2StepperEDAC(IntegratorStep):
    def initialize(
        self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
        d_u0, d_v0, d_w0, d_u, d_v, d_w, d_p, d_p0
    ):
        # Initialise `U^{n}`
        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]
        d_z0[d_idx] = d_z[d_idx]

        d_u0[d_idx] = d_u[d_idx]
        d_v0[d_idx] = d_v[d_idx]
        d_w0[d_idx] = d_w[d_idx]

        d_p0[d_idx] = d_p[d_idx]

    def stage1(
        self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
        d_u0, d_v0, d_w0, d_u, d_v, d_w, d_au, d_av, d_aw,
        dt, d_p, d_ap, d_p0
    ):
        dtb2 = 0.5*dt

        # Compute `U^{1}`
        d_x[d_idx] = d_x0[d_idx] + dtb2*d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dtb2*d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dtb2*d_w[d_idx]

        d_u[d_idx] = d_u0[d_idx] + dtb2*d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dtb2*d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dtb2*d_aw[d_idx]

        d_p[d_idx] = d_p0[d_idx] + dtb2*d_ap[d_idx]

    def stage2(
        self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
        d_u0, d_v0, d_w0, d_u, d_v, d_w, d_au, d_av, d_aw,
        dt, d_p, d_ap, d_p0
    ):
        # Compute `U^{n+1}`
        d_x[d_idx] = d_x0[d_idx] + dt*d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dt*d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dt*d_w[d_idx]

        d_u[d_idx] = d_u0[d_idx] + dt*d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dt*d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dt*d_aw[d_idx]

        d_p[d_idx] = d_p0[d_idx] + dt*d_ap[d_idx]


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

        d_rhoc[d_idx] += dx * d_gradrc[3*d_idx] + \
            dy * d_gradrc[3*d_idx + 1] + \
            dz * d_gradrc[3*d_idx + 2]


class UpdateVelocity(Equation):
    def post_loop(self, d_idx, d_x, d_x0, d_y, d_y0, d_z, d_z0, d_rho, d_u, d_v, d_w, d_gradv, d_gradrc, d_dpos):

        dx = d_x0[d_idx] - d_x[d_idx]
        dy = d_y0[d_idx] - d_y[d_idx]
        dz = d_z0[d_idx] - d_z[d_idx]

        d_u[d_idx] -= dx * d_gradv[9*d_idx] + \
            dy * d_gradv[9*d_idx + 1] + \
            dz * d_gradv[9*d_idx + 2]
        d_v[d_idx] -= dx * d_gradv[9*d_idx + 3] + \
            dy * d_gradv[9*d_idx + 4] + \
            dz * d_gradv[9*d_idx + 5]
        d_w[d_idx] -= dx * d_gradv[9*d_idx + 6] + \
            dy * d_gradv[9*d_idx + 7] + \
            dz * d_gradv[9*d_idx + 8]


class UpdateDensity(Equation):
    def post_loop(self, d_idx, d_x, d_x0, d_y, d_y0, d_z, d_z0, d_rhoc, d_u, d_v, d_w, d_gradv, d_gradrc, d_dpos):

        dx = d_x0[d_idx] - d_x[d_idx]
        dy = d_y0[d_idx] - d_y[d_idx]
        dz = d_z0[d_idx] - d_z[d_idx]

        d_rhoc[d_idx] -= dx * d_gradrc[3*d_idx] + \
            dy * d_gradrc[3*d_idx + 1] + \
            dz * d_gradrc[3*d_idx + 2]

class UpdatePressure(Equation):
    def post_loop(self, d_idx, d_x, d_x0, d_y, d_y0, d_z, d_z0, d_p, d_u, d_v, d_w, d_gradv, d_gradp, d_dpos):

        dx = d_x0[d_idx] - d_x[d_idx]
        dy = d_y0[d_idx] - d_y[d_idx]
        dz = d_z0[d_idx] - d_z[d_idx]

        d_p[d_idx] -= dx * d_gradp[3*d_idx] + \
            dy * d_gradp[3*d_idx + 1] + \
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


class ContinuityEquationWrong(Equation):
    def initialize(self, d_idx, d_arho):
        d_arho[d_idx] = 0.0

    def loop(self, d_idx, d_arho, s_idx, s_m, DWIJ, d_rhoc, s_rho, s_u, s_v, s_w, d_u, d_v, d_w):
        uij = d_u[d_idx] + s_u[s_idx]
        vij = d_v[d_idx] + s_v[s_idx]
        wij = d_w[d_idx] + s_w[s_idx]
        vijdotdwij = DWIJ[0]*uij + DWIJ[1]*vij + DWIJ[2]*wij
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

class SetDircheletVel(Equation):
    def initialize(
        self, d_u, d_v, d_w, d_ug_star, d_vg_star, d_wg_star, d_ug, d_vg, d_wg,
        d_idx):
        d_u[d_idx] = 0.0
        d_v[d_idx] = 0.0
        d_w[d_idx] = 0.0
        d_ug[d_idx] = 0.0
        d_vg[d_idx] = 0.0
        d_wg[d_idx] = 0.0
        d_ug_star[d_idx] = 0.0
        d_vg_star[d_idx] = 0.0
        d_wg_star[d_idx] = 0.0


class Dummy(Equation):
    '''
    A dummy bc doing nothing
    '''
    def initialize(self, d_p, d_idx):
        d_p[d_idx] = d_p[d_idx]


class EDACEquation(Equation):
    def __init__(self, dest, sources, rho0, gamma=7):
        self.rho0 = rho0
        self.gamma = gamma

        super(EDACEquation, self).__init__(dest, sources)

    def initialize(self, d_idx, d_ap):
        d_ap[d_idx] = 0.0

    def loop(self, d_idx, d_ap, s_idx, s_m, DWIJ, VIJ, d_rhoc, s_rho, d_c0):

        tmp = d_rhoc[d_idx] * d_c0[0]**2

        vijdotdwij = DWIJ[0]*VIJ[0] + DWIJ[1]*VIJ[1] + DWIJ[2]*VIJ[2]
        d_ap[d_idx] += tmp * s_m[s_idx]*vijdotdwij/s_rho[s_idx]


class EDACEquationSolid(Equation):
    def __init__(self, dest, sources, rho0, gamma=7):
        self.rho0 = rho0
        self.gamma = gamma

        super(EDACEquationSolid, self).__init__(dest, sources)

    def initialize(self, d_idx, d_ap):
        d_ap[d_idx] = 0.0

    def loop(self, d_idx, d_ap, s_idx, s_m, DWIJ, VIJ, d_rhoc, s_rho, d_c0,  s_ug_star, s_vg_star, s_wg_star, d_u, d_v, d_w):

        tmp = d_rhoc[d_idx] * d_c0[0]**2

        uij = d_u[d_idx] - s_ug_star[s_idx]
        vij = d_v[d_idx] - s_vg_star[s_idx]
        wij = d_w[d_idx] - s_wg_star[s_idx]
        vijdotdwij = DWIJ[0]*uij + DWIJ[1]*vij + DWIJ[2]*wij
        d_ap[d_idx] += tmp * s_m[s_idx]*vijdotdwij/s_rho[s_idx]

class MomentumEquationSecondOrder(Equation):
    # violeu p 5.131
    def __init__(self, dest, sources,
                 gx=0.0, gy=0.0, gz=0.0, rho0=1.0):

        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.rho0 = rho0
        super(MomentumEquationSecondOrder, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rhoc, d_p, d_au, d_av, d_aw, s_m,
             s_rho, s_p, DWIJ):

        tmp = (s_p[s_idx] - d_p[d_idx])/(d_rhoc[d_idx] * s_rho[s_idx])

        d_au[d_idx] += -s_m[s_idx] * (tmp) * DWIJ[0]
        d_av[d_idx] += -s_m[s_idx] * (tmp) * DWIJ[1]
        d_aw[d_idx] += -s_m[s_idx] * (tmp) * DWIJ[2]

    def post_loop(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] += self.gx
        d_av[d_idx] += self.gy
        d_aw[d_idx] += self.gz


class MomentumEquationSymmertric(Equation):
    def loop(self, d_idx, s_idx, d_rhoc,
             d_p, d_au, d_av, d_aw, s_m,
             s_rho, s_p, DWIJ):

        tmp = (s_p[s_idx] + d_p[d_idx])/(d_rhoc[d_idx] * s_rho[s_idx])

        d_au[d_idx] += -s_m[s_idx] * (tmp) * DWIJ[0]
        d_av[d_idx] += -s_m[s_idx] * (tmp) * DWIJ[1]
        d_aw[d_idx] += -s_m[s_idx] * (tmp) * DWIJ[2]

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

        sidx9, didx9 = declare('int', 2)
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
    def __init__(self, fluids, solids, dim, rho0,
                 hdx, ios=[], p0=0.0, nu=0.0, gamma=7.0, kernel_corr=False, pst_freq=10,
                 method='no_sd', scm='wcsph', eos='linear', pst='ipst',
                 intg='rk2'):
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
        self.ios = ios
        self.solver = None
        self.rho0 = rho0
        self.gamma = gamma
        self.dim = dim
        self.hdx = hdx
        self.nu = nu
        self.p0 = p0
        self.kernel_corr = kernel_corr
        self.pst_freq = pst_freq
        self.shifter = None
        self.method = 'no_sd'
        self.scm = 'wcsph'
        self.eos = 'linear'
        self.pst ='ipst'
        self.pe = 'asym'
        self.ce = 'div'
        self.ve = 'coupled'
        self.intg = 'rk2'


    def add_user_options(self, group):
        from pysph.sph.scheme import add_bool_argument
        add_bool_argument(
            group, "kernel-corr", dest="kernel_corr",
            help="Use this if kernel correction is required",
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
            '--eos', action='store', dest='eos', default='linear',
            help="scheme to be used scheme 'tait' or 'linear'"
        )

        group.add_argument(
            '--pe', action='store', dest='pe', default='asym',
            help="pressure equation to be used 'sym', 'asym' "
        )

        group.add_argument(
            '--ce', action='store', dest='ce', default='div',
            help="continuity equation to be used 'div', 'wrong' "
        )

        group.add_argument(
            '--ve', action='store', dest='ve', default='coupled',
            help="continuity equation to be used 'coupled', 'cleary'"
        )

        group.add_argument(
            '--pst', action='store', dest='pst', default='ipst',
            help="ipst or dppst"
        )

        group.add_argument(
            '--intg', action='store', dest='intg', default='rk2',
            help="euler or rk2"
        )

    def consume_user_options(self, options):
        vars = ["kernel_corr", "pst_freq", "method", "scm", "eos", "pst", "pe", "ce", "ve", "intg"]

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

        cls = RK2Integrator
        step_cls = RK2Stepper
        if self.intg == 'euler':
            cls = EulerIntegrator
            step_cls = EulerStepNew
            for stepper in steppers.keys():

                steppers[stepper] = EulerStepNew()
        print(steppers)

        if self.scm == 'edac':
            step_cls = RK2StepperEDAC
        print(self.scm, step_cls)
        for name in self.fluids:
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
        # from pysph.sph.wc.basic import TaitEOS
        from pysph.sph.wc.transport_velocity import MomentumEquationArtificialViscosity

        equations = []
        g1 = []
        all = self.fluids + self.solids + self.ios

        g0 = []
        self.pressure_density_eq(g0)
        for name in self.solids:
            from config_mms import SetValuesonSolid
            g0.append(SetValuesonSolid(dest=name, sources=None))
        for name in self.ios:
            from config_mms import SetValuesonSolid
            g0.append(SetValuesonSolid(dest=name, sources=None))
        equations.append(Group(equations=g0))

        g0 = []
        for name in all:
            g0.append(SummationDensity(dest=name, sources=all))
        equations.append(Group(equations=g0))

        g0 = []
        from pysph.sph.wc.kernel_correction import GradientCorrectionPreStep, GradientCorrection
        for name in self.fluids+self.ios:
            g0.append(
                GradientCorrectionPreStep(dest=name,
                                            sources=all,
                                            dim=self.dim))
        for name in self.solids:
            g0.append(
            GradientCorrectionPreStep(dest=name,
                                    sources=all,
                                    dim=self.dim))
        equations.append(Group(equations=g0))

        g2 = []
        self.compute_vel_grad(g2)
        equations.append(Group(equations=g2))


        g3 = []
        if self.kernel_corr:
            for name in self.fluids:
                g3.append(GradientCorrection(dest=name, sources=all))
        self.get_continuity_eq(g3)
        self.get_viscous_eq(g3)
        self.get_pressure_grad(g3)
        self.add_source_term(g3)


        equations.append(Group(equations=g3))

        return equations

    def compute_vel_grad(self, g2):
        all = self.fluids + self.solids + self.ios
        from pysph.sph.wc.kernel_correction import GradientCorrectionPreStep, GradientCorrection
        for name in self.fluids+self.ios:
            g2.append(GradientCorrection(dest=name, sources=all))
        for name in self.solids:
            g2.append(GradientCorrection(dest=name, sources=all))
        for name in self.fluids+self.ios:
            g2.append(VelocityGradient(dest=name, sources=self.fluids+self.ios))
            g2.append(VelocityGradientSoild(dest=name, sources=self.solids))
        for name in self.solids:
            g2.append(VelocityGradientDestSoild(dest=name, sources=self.fluids+self.ios))
            g2.append(VelocityGradientSolidSoild(dest=name, sources=self.solids))

    def solid_bc(self, g0):
        for name in self.solids:
            g0.append(
                Dummy(dest=name, sources=None))

    def pressure_density_eq(self, g0):
        for name in self.fluids:
            if self.scm == 'edac':
                g0.append(EvaluateRhoc(dest=name, sources=None, rho0=self.rho0))
            if self.scm == 'wcsph' or self.scm == 'fatehi':
                if self.eos == 'tait':
                    g0.append(TaitEOS(dest=name, sources=None, rho0=self.rho0, gamma=self.gamma))
                elif self.eos == 'linear':
                    g0.append(LinearEOS(dest=name, sources=None, rho0=self.rho0, gamma=self.gamma))

    def get_viscous_eq(self, g3):
        all = self.fluids + self.solids + self.ios
        for name in self.fluids:
            if self.ve == 'coupled':
                g3.append(DivGrad(dest=name, sources=all, nu=self.nu, rho0=self.rho0))
            elif self.ve == 'cleary':
                from scheme_equation import ViscosityCleary
                g3.append(ViscosityCleary(dest=name, sources=all, nu=self.nu, rho0=self.rho0))


    def get_pressure_grad(self, g3):
        all = self.fluids + self.solids + self.ios
        for name in self.fluids:
            if self.pe == 'sym':
                g3.append(MomentumEquationSymmertric(dest=name, sources=all))
            elif self.pe == 'asym':
                g3.append(MomentumEquationSecondOrder(dest=name, sources=all))

    def get_continuity_eq(self, g3):
        all = self.fluids + self.ios
        for name in self.fluids:
            if self.scm == 'wcsph':
                if self.ce == 'div':
                    g3.append(ContinuityEquation(dest=name, sources=all))
                elif self.ce == 'wrong':
                    g3.append(ContinuityEquationWrong(dest=name, sources=all))
                g3.append(ContinuityEquationSolid(dest=name, sources=self.solids))
            else:
                g3.append(EDACEquation(dest=name, sources=all, rho0=self.rho0))
                g3.append(EDACEquationSolid(dest=name, sources=self.solids, rho0=self.rho0))

    def add_source_term(self, g3):
        for name in self.fluids:
            from config_mms import AddContinuitySourceTerm, AddMomentumSourceTerm, AddPressureEvolutionSourceTerm
            g3.append(AddMomentumSourceTerm(dest=name, sources=None))
            if self.scm == 'edac':
                g3.append(AddPressureEvolutionSourceTerm(dest=name, sources=None))
            else:
                g3.append(AddContinuitySourceTerm(dest=name, sources=None))

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
                'name': 'normal',
                'stride': 3
            }, {
                'name': 'gradp',
                'stride': 3
            }, 'ki', 'ki0', 'rhoc', 'rhoc0', 'ap', 'p0', 'V0', 'ug_star',
            'vg_star', 'wg_star', 'ug', 'vg', 'wg', 'wij', 'wf', 'uf', 'vf'
        ]
        props += ['xi', 'yi', 'zi', 'ui', 'vi', 'wi', 'rhoi']
        output_props = ['x', 'y', 'z', 'u', 'v', 'w', 'rho', 'm', 'h',
                        'pid', 'gid', 'tag', 'p', 'rhoc', 'arho', 'normal', 'ug', 'vg']
        delta_sph_props = [
            {'name': 'm_mat', 'stride': 9},
            {'name': 'gradv', 'stride': 9},
        ]
        props += delta_sph_props
        print(props)
        for pa in particles:
            self._ensure_properties(pa, props, clean)
            pa.set_output_arrays(output_props)
            pa.add_constant('maxki0', 0.0)
            pa.add_constant('maxki', 0.0)


    def post_step_dppst(self, pa_arr, domain, get_mms_eq, mms, bc, solids=None):
        if self.shifter is None:
            from pysph.tools.sph_evaluator import SPHEvaluator
            from pysph.sph.wc.kernel_correction import GradientCorrection, GradientCorrectionPreStep
            from pst import ModifiedFickian, IterativePST, NumberDensityMoment
            from pysph.sph.equation import Group
            from pysph.sph.basic_equations import \
                (SummationDensity)
            from config_mms import SetValuesonSolid

            equations = []
            if solids is None:
                _solids = self.solids
            else:
                _solids = solids

            all = self.fluids + _solids + self.ios

            g0 = []
            for name in _solids:
                g0.append(SetValuesonSolid(dest=name, sources=None))
            for name in all:
                g0.append(SummationDensity(dest=name, sources=all))
            equations.append(Group(equations=g0))

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
            equations.append(Group(equations=g3))

            equations = get_mms_eq(equations, mms, bc)
            print(equations)

            self.shifter = SPHEvaluator(
                arrays=pa_arr, equations=equations, dim=self.dim,
                kernel=self.solver.kernel, backend='cython'
            )

        else:
            self.shifter.update()
            self.shifter.evaluate(t=self.solver.t, dt=self.solver.dt)

    def post_step_ipst(self, pa_arr, domain, get_mms_eq, mms, bc, solids=None):
        if self.shifter is None:
            from pysph.tools.sph_evaluator import SPHEvaluator
            from pysph.sph.wc.kernel_correction import GradientCorrection, GradientCorrectionPreStep
            from pst import ModifiedFickian, IterativePST, NumberDensityMoment
            from pysph.sph.equation import Group
            from pysph.sph.basic_equations import \
                (SummationDensity)
            from config_mms import SetValuesonSolid

            equations = []
            if solids is None:
                _solids = self.solids
            else:
                _solids = solids
            all = self.fluids + _solids + self.ios

            g0 = []
            for name in _solids + self.ios:
                g0.append(SetValuesonSolid(dest=name, sources=None))
            for name in all:
                g0.append(SummationDensity(dest=name, sources=all))
            equations.append(Group(equations=g0))


            g1 = []
            for name in self.fluids:
                g1.append(GradientCorrectionPreStep(dest=name, sources=all, dim=self.dim))
                g1.append(SaveInitialdistances(dest=name, sources=None))
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


            equations = get_mms_eq(equations, mms, bc)
            print(equations)

            self.shifter = SPHEvaluator(
                arrays=pa_arr, equations=equations, dim=self.dim,
                kernel=self.solver.kernel, backend='cython'
            )

        if (self.pst_freq > 0 and self.solver.count % self.pst_freq == 0) & (self.solver.count > 1):
            self.shifter.update()
            self.shifter.evaluate(t=self.solver.t, dt=self.solver.dt)

    def post_step(self, pa_arr, domain, get_mms_eq, mms='mms1', bc='mms', solids=None):
        if self.pst == 'ipst':
            self.post_step_ipst(pa_arr, domain, get_mms_eq, mms, bc, solids)
        elif self.pst == 'dppst':
            self.post_step_dppst(pa_arr, domain, get_mms_eq, mms, bc, solids)