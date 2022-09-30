'''
S. Adami, X. Y. Hu, and N. A. Adams, “A generalized wall boundary condition
for smoothed particle hydrodynamics,” Journal of Computational Physics, vol.
231, no. 21, pp. 7057–7075, Aug. 2012, doi: 10.1016/j.jcp.2012.05.005.
'''

from pysph.sph.equation import Equation
from compyle.api import declare
from pysph.sph.wc.edac import SourceNumberDensity
from solid_bc.takeda import SelfNumberDensity, EvaluateVelocity


def boundary_props():
    '''
    bid: Boundary particle closest to fluid
    cid: greater than zero if corner particle
    cbid: Boundary particle closed to fluid particle
    '''
    return ['wij', 'swij']


def get_bc_names():
    return ['solid0', 'solid1']


def requires():
    is_mirror = False
    is_boundary = False
    is_ghost = True
    is_ghost_mirror = False
    is_boundary_shift = False

    return is_mirror, is_boundary, is_ghost, is_ghost_mirror, is_boundary_shift



class ComputeNumberDensity(SourceNumberDensity):
    pass



class AdamiPressureBC(Equation):
    '''
    [1]S. Adami, X. Y. Hu, and N. A. Adams, “A generalized wall boundary condition for smoothed particle hydrodynamics,” Journal of Computational Physics, vol. 231, no. 21, pp. 7057–7075, Aug. 2012, doi: 10.1016/j.jcp.2012.05.005.
    '''
    def __init__(self, dest, sources, rho0, p0, b=1.0, gx=0.0, gy=0.0, gz=0.0):
        self.rho0 = rho0
        self.p0 = p0
        self.gx = gx
        self.gy = gy
        self.gz = gz

        super(AdamiPressureBC, self).__init__(dest, sources)

    def initialize(self, d_idx, d_p):
        d_p[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_p, s_p, s_rho,
             d_au, d_av, d_aw, WIJ, XIJ):

        gdotxij = (self.gx - d_au[d_idx])*XIJ[0] + \
            (self.gy - d_av[d_idx])*XIJ[1] + \
            (self.gz - d_aw[d_idx])*XIJ[2]

        d_p[d_idx] += s_p[s_idx]*WIJ + s_rho[s_idx]*gdotxij*WIJ

    def post_loop(self, d_idx, d_wij, d_p, d_rho):
        if d_wij[d_idx] > 1e-14:
            d_p[d_idx] /= d_wij[d_idx]


class AdamiWallVelocity(Equation):
    def __init__(self, dest, sources, U=0.0, V=0.0, W=0.0):
        self.U = U
        self.V = V
        self.W = W
        super(AdamiWallVelocity, self).__init__(dest, sources)

    def initialize(self, d_idx, d_uf, d_vf, d_wf, d_u, d_v, d_w):
        d_uf[d_idx] = 0.0
        d_vf[d_idx] = 0.0
        d_wf[d_idx] = 0.0
        d_u[d_idx] = self.U
        d_v[d_idx] = self.V
        d_w[d_idx] = self.W

    def loop(self, d_idx, s_idx, d_uf, d_vf, d_wf,
             s_u, s_v, s_w, WIJ):

        d_uf[d_idx] += s_u[s_idx] * WIJ
        d_vf[d_idx] += s_v[s_idx] * WIJ
        d_wf[d_idx] += s_w[s_idx] * WIJ

    def post_loop(self, d_uf, d_vf, d_wf, d_wij, d_idx,
                  d_ug, d_vg, d_wg, d_u, d_v, d_w):

        if d_wij[d_idx] > 1e-12:
            d_uf[d_idx] /= d_wij[d_idx]
            d_vf[d_idx] /= d_wij[d_idx]
            d_wf[d_idx] /= d_wij[d_idx]

        d_ug[d_idx] = 2*d_u[d_idx] - d_uf[d_idx]
        d_vg[d_idx] = 2*d_v[d_idx] - d_vf[d_idx]
        d_wg[d_idx] = 2*d_w[d_idx] - d_wf[d_idx]


class AdamiSlipWallVelocity(Equation):
    # free slip required for divergence operator
    def initialize(self, d_idx, d_ug_star, d_vg_star, d_wg_star):
        d_ug_star[d_idx] = 0.0
        d_vg_star[d_idx] = 0.0
        d_wg_star[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_ug_star, d_vg_star, d_wg_star,
             s_u, s_v, s_w, WIJ):

        d_ug_star[d_idx] += s_u[s_idx] * WIJ
        d_vg_star[d_idx] += s_v[s_idx] * WIJ
        d_wg_star[d_idx] += s_w[s_idx] * WIJ

    def post_loop(self, d_idx, d_wij, d_ug_star, d_vg_star,
                  d_wg_star, d_normal, d_u, d_v, d_w):
        idx = declare('int')
        idx = 3*d_idx
        if d_wij[d_idx] > 1e-14:
            d_ug_star[d_idx] /= d_wij[d_idx]
            d_vg_star[d_idx] /= d_wij[d_idx]
            d_wg_star[d_idx] /= d_wij[d_idx]

        # u_g \cdot n = 2*(u_wall \cdot n ) - (u_f \cdot n)
        # u_g \cdot t = (u_f \cdot t) = u_f - (u_f \cdot n)
        tmp1 = d_u[d_idx] - d_ug_star[d_idx]
        tmp2 = d_v[d_idx] - d_vg_star[d_idx]
        tmp3 = d_w[d_idx] - d_wg_star[d_idx]

        projection = (tmp1*d_normal[idx] +
                      tmp2*d_normal[idx+1] +
                      tmp3*d_normal[idx+2])

        d_ug_star[d_idx] += 2*projection * d_normal[idx]
        d_vg_star[d_idx] += 2*projection * d_normal[idx+1]
        d_wg_star[d_idx] += 2*projection * d_normal[idx+2]


class AdamiCopySlipWallVelocity(Equation):
    def post_loop(self, d_ug_star, d_vg_star,
                  d_wg_star, d_idx,
                  d_ug, d_vg, d_wg,):

        d_ug[d_idx] = d_ug_star[d_idx]
        d_vg[d_idx] = d_vg_star[d_idx]
        d_wg[d_idx] = d_wg_star[d_idx]


def solid_bc(bcs, fluids, rho0, p0):
    print(bcs)
    g0 = []
    g1 = []
    for bc in bcs:
        if bc == 'u_no_slip':
            for name in bcs[bc]:
                g0.extend([
                    ComputeNumberDensity(dest='solid0', sources=['fluid', 'solid1']),
                    SelfNumberDensity(dest='solid0', sources=['solid0']),
                    AdamiWallVelocity(dest='solid0', sources=['fluid', 'solid1'])
                ])
                g1.append(
                    EvaluateVelocity(dest='solid0', sources=['solid0'])
                )
                return [g0, g1]
        if bc == 'u_slip':
            for name in bcs[bc]:
                g0.append(
                    ComputeNumberDensity(dest=name, sources=['fluid', 'solid1']))
                g0.append(
                    AdamiSlipWallVelocity(dest=name, sources=['fluid', 'solid1']))
                g0.append(
                    AdamiCopySlipWallVelocity(dest=name, sources=None))
                return [g0]
        if bc == 'p_solid':
            for name in bcs[bc]:
                g0.append(
                    ComputeNumberDensity(dest=name, sources=fluids))
                g0.append(
                    AdamiPressureBC(dest=name,
                                    sources=fluids,
                                    rho0=rho0,
                                    p0=p0))
                return [g0]