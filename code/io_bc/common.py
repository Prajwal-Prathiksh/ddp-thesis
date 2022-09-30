from pysph.sph.integrator_step import IntegratorStep
from pysph.sph.equation import Equation
from compyle.api import declare
from solid_bc.marrone import LiuCorrection, LiuCorrectionPreStep


class InletStep(IntegratorStep):
    def initialize(self, d_x0, d_idx, d_x):
        d_x0[d_idx] = d_x[d_idx]

    def stage1(self, d_idx, d_x, d_x0, d_u, dt):
        dtb2 = 0.5 * dt
        d_x[d_idx] = d_x0[d_idx] + dtb2*d_u[d_idx]

    def stage2(self, d_idx, d_x, d_x0, d_u, dt):
        d_x[d_idx] = d_x0[d_idx] + dt*d_u[d_idx]


class OutletStep(IntegratorStep):
    def initialize(self, d_x0, d_idx, d_x):
        d_x0[d_idx] = d_x[d_idx]

    def stage1(self, d_idx, d_x, d_x0, d_u, dt):
        dtb2 = 0.5 * dt
        d_x[d_idx] = d_x0[d_idx] + dtb2*d_u[d_idx]

    def stage2(self, d_idx, d_x, d_x0, d_u, dt):
        d_x[d_idx] = d_x0[d_idx] + dt*d_u[d_idx]


class MirrorStep(IntegratorStep):
    def initialize(self, d_x0, d_idx, d_x):
        d_x0[d_idx] = d_x[d_idx]

    def stage1(self, d_idx, d_x, d_x0, d_u, dt):
        dtb2 = 0.5 * dt
        d_x[d_idx] = d_x0[d_idx] - dtb2*d_u[d_idx]

    def stage2(self, d_idx, d_x, d_x0, d_u, dt):
        d_x[d_idx] = d_x0[d_idx] - dt*d_u[d_idx]


class UpdateNormalsAndDisplacements(Equation):
    def __init__(self, dest, sources, xn, yn, zn, xo, yo, zo):
        """Update normal and perpendicular distance from the interface
        for the inlet/outlet particles

        Parameters
        ----------

        dest : str
            destination particle array name
        sources : list
            List of source particle arrays
        xn : float
            x component of interface outward normal
        yn : float
            y component of interface outward normal
        zn : float
            z component of interface outward normal
        xo : float
            x coordinate of interface point
        yo : float
            y coordinate of interface point
        zo : float
            z coordinate of interface point
        """
        self.xn = xn
        self.yn = yn
        self.zn = zn
        self.xo = xo
        self.yo = yo
        self.zo = zo

        super(UpdateNormalsAndDisplacements, self).__init__(dest, sources)

    def initialize(self, d_idx, d_xn, d_yn, d_zn, d_x, d_y, d_z, d_disp):
        d_xn[d_idx] = self.xn
        d_yn[d_idx] = self.yn
        d_zn[d_idx] = self.zn

        xij = declare('matrix(3)')
        xij[0] = d_x[d_idx] - self.xo
        xij[1] = d_y[d_idx] - self.yo
        xij[2] = d_z[d_idx] - self.zo

        d_disp[d_idx] = abs(xij[0]*d_xn[d_idx] + xij[1]*d_yn[d_idx] +
                            xij[2]*d_yn[d_idx])


class CopyNormalsandDistances(Equation):
    """Copy normals and distances from outlet/inlet particles to ghosts"""

    def initialize_pair(self, d_idx, d_xn, d_yn, d_zn, s_xn, s_yn, s_zn,
                        d_disp, s_disp):
        d_xn[d_idx] = s_xn[d_idx]
        d_yn[d_idx] = s_yn[d_idx]
        d_zn[d_idx] = s_zn[d_idx]

        d_disp[d_idx] = s_disp[d_idx]

class PressureBC(Equation):
    def initialize(self, d_idx, d_p):
        d_p[d_idx] = 0.0

    def loop(self, d_idx, s_p, s_m, s_rho, d_p, s_idx, d_wij):
        d_p[d_idx] += s_p[s_idx] * d_wij[d_idx] * s_m[s_idx] / s_rho[s_idx]


class CopyPressureMirrorToGhost(Equation):
    def initialize(self, d_idx, d_p):
        d_p[d_idx] = 0.0

    def initialize_pair(self, d_idx, d_p, s_p):
        d_p[d_idx] = s_p[d_idx]


class VelocityBC(Equation):
    def initialize(self, d_idx, d_u, d_v, d_w):
        d_u[d_idx] = 0.0
        d_v[d_idx] = 0.0
        d_w[d_idx] = 0.0

    def loop(self, d_idx, s_u, s_v, s_w, s_m, s_rho, d_u, d_v,
             d_w, s_idx, d_wij):
        d_u[d_idx] += s_u[s_idx] * d_wij[d_idx] * s_m[s_idx] / s_rho[s_idx]
        d_v[d_idx] += s_v[s_idx] * d_wij[d_idx] * s_m[s_idx] / s_rho[s_idx]
        d_w[d_idx] += s_w[s_idx] * d_wij[d_idx] * s_m[s_idx] / s_rho[s_idx]


class VelocityGradientJ(Equation):
    def __init__(self, dest, sources, dim=1, rho0=1.0):
        self.dim = dim
        self.rho0 = rho0

        super(VelocityGradientJ, self).__init__(dest, sources)

    def initialize(self, d_idx, d_gradv):
        i = declare('int')
        for i in range(9):
            d_gradv[9*d_idx+i] = 0.0

    def loop(self, d_idx, s_idx, s_m, s_rho,
             d_gradv, DWIJ, s_u, s_v, s_w):

        i, j = declare('int', 2)
        vel = declare('matrix(3)')
        vel[0] = s_u[s_idx]
        vel[1] = s_v[s_idx]
        vel[2] = s_w[s_idx]
        tmp = s_m[s_idx]/s_rho[s_idx]
        for i in range(3):
            for j in range(3):
                d_gradv[9*d_idx+3*i+j] += tmp * vel[i] * DWIJ[j]


class PressureGradientJ(Equation):
    def __init__(self, dest, sources, dim=1, rho0=1.0):
        self.dim = dim
        self.rho0 = rho0

        super(PressureGradientJ, self).__init__(dest, sources)

    def initialize(self, d_idx, d_gradp):
        i = declare('int')
        for i in range(3):
            d_gradp[3*d_idx+i] = 0.0

    def loop(self, d_idx, s_idx, s_m, s_rho, s_p,
             d_gradp, DWIJ):

        i = declare('int')
        tmp = s_m[s_idx]/s_rho[s_idx]
        for i in range(3):
            d_gradp[3*d_idx+i] += tmp * s_p[s_idx] * DWIJ[i]


class CopySlipMirrorToGhost(Equation):
    def initialize(self, d_idx, d_ug_star, d_vg_star, d_wg_star):
        d_ug_star[d_idx] = 0.0
        d_vg_star[d_idx] = 0.0
        d_wg_star[d_idx] = 0.0

    def initialize_pair(self, d_idx, d_ug_star, d_vg_star, d_wg_star, s_u, s_v,
                        s_w):
        d_ug_star[d_idx] = s_u[d_idx]
        d_ug_star[d_idx] = s_v[d_idx]
        d_ug_star[d_idx] = s_w[d_idx]


class CopyNoSlipMirrorToGhost(Equation):
    def initialize(self, d_idx, d_ug, d_vg, d_wg):
        d_ug[d_idx] = 0.0
        d_vg[d_idx] = 0.0
        d_wg[d_idx] = 0.0

    def initialize_pair(self, d_idx, d_ug, d_vg, d_wg, s_u, s_v,
                        s_w):
        d_ug[d_idx] = -s_u[d_idx]
        d_ug[d_idx] = -s_v[d_idx]
        d_ug[d_idx] = -s_w[d_idx]