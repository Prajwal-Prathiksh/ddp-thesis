''' [1]G. Fourtakas, J. M. Dominguez, R. Vacondio, and B. D. Rogers, “Local
uniform stencil (LUST) boundary condition for arbitrary 3-D boundaries in
parallel smoothed particle hydrodynamics (SPH) models,” Computers & Fluids,
vol. 190, pp. 346–361, Aug. 2019, doi: 10.1016/j.compfluid.2019.06.009.

We require to evaluate gradient at each neighbor particles but the neighbor
particle do not have corrected kernel gradients.

We modified this algorithm in the current paper
'''

from numpy import RankWarning
from pysph.sph.equation import Equation
from compyle.api import declare


def boundary_props():
    '''
    bid: Boundary particle closest to fluid
    cid: greater than zero if corner particle
    cbid: Boundary particle closed to fluid particle
    '''
    return ['proj']


def get_bc_names():
    return ['solid0', 'solid1']


def requires():
    is_mirror = False
    is_boundary = True
    is_ghost = True
    is_ghost_mirror = False
    is_boundary_shift = False

    return is_mirror, is_boundary, is_ghost, is_ghost_mirror, is_boundary_shift


class FindFluidNeighbors(Equation):
    ''' For every fluid particle find the nearest boundary particle in the normal
    direction'''
    def initialize(self, d_idx, d_bid, d_proj):
        d_bid[d_idx] = -1
        d_proj[d_idx] = 10000

    def loop_all(self, d_idx, d_bid, d_x, d_y, d_z, s_x, s_y, s_z, s_normal,
                 NBRS, N_NBRS, d_proj, s_cid, d_normal):
        s_idx = declare('int')
        dmin = 10000

        for i in range(N_NBRS):
            s_idx = NBRS[i]
            xij = d_x[d_idx] - s_x[s_idx]
            yij = d_y[d_idx] - s_y[s_idx]
            zij = d_z[d_idx] - s_z[s_idx]
            rij = sqrt(xij**2 + yij**2 + zij**2)
            if s_cid[s_idx] < 0:
                if rij < dmin:
                    proj_d = xij * s_normal[3 * s_idx] + yij * s_normal[
                        3 * s_idx + 1] + zij * s_normal[3 * s_idx + 2]
                    if proj_d < d_proj[d_idx]:
                        dmin = rij
                        d_proj[d_idx] = proj_d
                        d_bid[d_idx] = s_idx
                        d_normal[3 * d_idx] = s_normal[3*s_idx]
                        d_normal[3 * d_idx + 1] = s_normal[3*s_idx + 1]
                        d_normal[3 * d_idx + 2] = s_normal[3*s_idx + 2]


class FindDistanceFromBoundary(Equation):
    #dest is fluid and source is solid
    def initialize(self, d_proj, d_idx):
        d_proj[d_idx] = 0.0

    def loop_all(self, d_idx, d_x, d_y, d_z, s_x, s_y, s_z, d_normal, d_proj, NBRS, N_NBRS):
        i, s_idx = declare("int", 2)
        dmin = 10000

        for i in range(N_NBRS):
            s_idx = NBRS[i]
            xij = d_x[d_idx] - s_x[s_idx]
            yij = d_y[d_idx] - s_y[s_idx]
            zij = d_z[d_idx] - s_z[s_idx]
            rij = sqrt(xij**2 + yij**2 + zij**2)
            if (rij < dmin):
                proj = xij * d_normal[3 *d_idx] + yij * d_normal[3*d_idx + 1]
                d_proj[d_idx] = proj


class MomentumEquationFourtakas(Equation):
    # violeu p 5.131
    def __init__(self, dest, sources,
                 gx=0.0, gy=0.0, gz=0.0, rho0=1.0):

        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.rho0 = rho0
        super(MomentumEquationFourtakas, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop_all(self, d_idx, d_p, s_p, d_x, d_y, d_z, s_x, s_y, s_z, SPH_KERNEL, N_NBRS, NBRS,
                 d_h, d_rhoc, s_rho, d_au, d_av, d_aw, s_m, d_normal, s_proj, d_proj):
        s_idx, i = declare('int', 2)
        dwij = declare('matrix(3)')

        proj = d_proj[d_idx]

        for i in range(N_NBRS):
            s_idx = NBRS[i]
            xij = d_x[d_idx] - s_x[s_idx]
            yij = d_y[d_idx] - s_y[s_idx]
            zij = d_z[d_idx] - s_z[s_idx]
            rij = sqrt(xij**2 + yij**2 + zij**2)
            hij = d_h[d_idx]
            SPH_KERNEL.gradient([xij, yij, zij], rij, hij, dwij)
            tmp = (s_p[s_idx] - d_p[d_idx])/(d_rhoc[d_idx] * s_rho[s_idx])

            d_au[d_idx] += -s_m[s_idx] * (tmp) * dwij[0]
            d_av[d_idx] += -s_m[s_idx] * (tmp) * dwij[1]
            d_aw[d_idx] += -s_m[s_idx] * (tmp) * dwij[2]

            # the particle close to the boundary has a virtual mirror about
            # the boundary interface
            if (proj > 1e-14) & (proj < 3 * d_h[d_idx]):
                sproj = s_proj[s_idx]

                if (sproj > 1e-14) & (sproj < 2*d_h[d_idx]):
                    xsnew = s_x[s_idx] + 2 * d_normal[3 * d_idx] * s_proj[s_idx]
                    ysnew = s_y[s_idx] + 2 * d_normal[3 * d_idx + 1] * s_proj[s_idx]

                    xij = d_x[d_idx] - xsnew
                    yij = d_y[d_idx] - ysnew
                    zij = 0.0
                    rij = sqrt(xij**2 + yij**2 + zij**2)
                    hij = d_h[d_idx]
                    SPH_KERNEL.gradient([xij, yij, zij], rij, hij, dwij)

                    tmp = (s_p[s_idx] - d_p[d_idx])/(d_rhoc[d_idx] * s_rho[s_idx])
                    d_au[d_idx] += -s_m[s_idx] * (tmp) * dwij[0]
                    d_av[d_idx] += -s_m[s_idx] * (tmp) * dwij[1]
                    d_aw[d_idx] += -s_m[s_idx] * (tmp) * dwij[2]


class VelocityGradientNoSlip(Equation):
    def initialize(self, d_idx, d_gradv):
        i = declare('int')
        for i in range(9):
            d_gradv[9*d_idx+i] = 0.0

    def loop_all(self, d_idx, d_x, d_y, d_z, s_x, s_y, s_z, SPH_KERNEL, N_NBRS, NBRS,
                 d_h, s_rho, d_gradv, s_m, d_normal, s_proj, d_proj, s_u, s_v, s_w,
                 d_u, d_v, d_w):
        s_idx, i, j = declare('int', 3)
        dwij = declare('matrix(3)')
        uij = declare('matrix(3)')

        proj = d_proj[d_idx]
        tmp = s_m[s_idx]/s_rho[s_idx]

        for i in range(N_NBRS):
            s_idx = NBRS[i]
            xij = d_x[d_idx] - s_x[s_idx]
            yij = d_y[d_idx] - s_y[s_idx]
            zij = d_z[d_idx] - s_z[s_idx]

            uij[0] = d_u[d_idx] - s_u[s_idx]
            uij[1] = d_v[d_idx] - s_v[s_idx]
            uij[2] = d_w[d_idx] - s_w[s_idx]

            rij = sqrt(xij**2 + yij**2 + zij**2)
            hij = d_h[d_idx]
            SPH_KERNEL.gradient([xij, yij, zij], rij, hij, dwij)

            for i in range(3):
                for j in range(3):
                    d_gradv[9*d_idx+3*i+j] += tmp * -uij[i] * dwij[j]

            # the particle close to the boundary has a virtual mirror about
            # the boundary interface
            if (proj > 1e-14) & (proj < 3 * d_h[d_idx]):
                sproj = s_proj[s_idx]

                if (sproj > 1e-14) & (sproj < 2*d_h[d_idx]):
                    xsnew = s_x[s_idx] + 2 * d_normal[3 * d_idx] * s_proj[s_idx]
                    ysnew = s_y[s_idx] + 2 * d_normal[3 * d_idx + 1] * s_proj[s_idx]

                    xij = d_x[d_idx] - xsnew
                    yij = d_y[d_idx] - ysnew
                    zij = 0.0

                    uij[0] = d_u[d_idx] + s_u[s_idx]
                    uij[1] = d_v[d_idx] + s_v[s_idx]
                    uij[2] = d_w[d_idx] + s_w[s_idx]

                    rij = sqrt(xij**2 + yij**2 + zij**2)
                    hij = d_h[d_idx]
                    SPH_KERNEL.gradient([xij, yij, zij], rij, hij, dwij)

                    for i in range(3):
                        for j in range(3):
                            d_gradv[9*d_idx+3*i+j] += tmp * -uij[i] * dwij[j]


class DivGradFourtakas(Equation):
    def __init__(self, dest, sources, nu=0.0, rho0=1.0):
        self.nu = nu
        self.rho0 = rho0
        super(DivGradFourtakas, self).__init__(dest, sources)


    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop_all(self, d_idx, d_x, d_y, d_z, s_x, s_y, s_z, SPH_KERNEL, N_NBRS, NBRS,
                 d_h, s_rho, d_au, d_av, d_aw, s_m, s_normal, s_proj, d_proj, d_gradv, s_gradv):
        s_idx, i = declare('int', 2)
        dwij = declare('matrix(3)')
        sidx9, didx9 = declare('int', 2)
        tmp = declare('double')
        didx9 = 9*d_idx

        proj = d_proj[d_idx]
        tmp = self.nu * s_m[s_idx]/s_rho[s_idx]

        for i in range(N_NBRS):
            s_idx = NBRS[i]
            sidx9 = 9*s_idx
            xij = d_x[d_idx] - s_x[s_idx]
            yij = d_y[d_idx] - s_y[s_idx]
            zij = d_z[d_idx] - s_z[s_idx]
            rij = sqrt(xij**2 + yij**2 + zij**2)
            hij = d_h[d_idx]
            SPH_KERNEL.gradient([xij, yij, zij], rij, hij, dwij)

            d_au[d_idx] += -(
                tmp *
                (dwij[0] * (d_gradv[didx9] - s_gradv[sidx9]) +
                dwij[1] * (d_gradv[didx9 + 1] - s_gradv[sidx9 + 1]) +
                dwij[2] * (d_gradv[didx9 + 2] - s_gradv[sidx9 + 2])))

            d_av[d_idx] += -(
                tmp *
                (dwij[0] * (d_gradv[didx9 + 3] - s_gradv[sidx9 + 3]) +
                dwij[1] * (d_gradv[didx9 + 4] - s_gradv[sidx9 + 4]) +
                dwij[2] * (d_gradv[didx9 + 5] - s_gradv[sidx9 + 5])))

            d_aw[d_idx] += -(
                tmp *
                (dwij[0] * (d_gradv[didx9 + 6] - s_gradv[sidx9 + 6]) +
                dwij[1] * (d_gradv[didx9 + 7] - s_gradv[sidx9 + 7]) +
                dwij[2] * (d_gradv[didx9 + 8] - s_gradv[sidx9 + 8])))

            # the particle close to the boundary has a virtual mirror about
            # the boundary interface
            if (proj > 1e-14) & (proj < 3 * d_h[d_idx]):
                sproj = s_proj[s_idx]

                if (sproj > 1e-14) & (sproj < 2*d_h[d_idx]):
                    xsnew = s_x[s_idx] + 2 * s_normal[3 * s_idx] * s_proj[s_idx]
                    ysnew = s_y[s_idx] + 2 * s_normal[3 * s_idx + 1] * s_proj[s_idx]

                    xij = d_x[d_idx] - xsnew
                    yij = d_y[d_idx] - ysnew
                    zij = 0.0
                    rij = sqrt(xij**2 + yij**2 + zij**2)
                    hij = d_h[d_idx]
                    SPH_KERNEL.gradient([xij, yij, zij], rij, hij, dwij)

                    d_au[d_idx] += -(
                        tmp *
                        (dwij[0] * (d_gradv[didx9] - s_gradv[sidx9]) +
                        dwij[1] * (d_gradv[didx9 + 1] - s_gradv[sidx9 + 1]) +
                        dwij[2] * (d_gradv[didx9 + 2] - s_gradv[sidx9 + 2])))

                    d_av[d_idx] += -(
                        tmp *
                        (dwij[0] * (d_gradv[didx9 + 3] - s_gradv[sidx9 + 3]) +
                        dwij[1] * (d_gradv[didx9 + 4] - s_gradv[sidx9 + 4]) +
                        dwij[2] * (d_gradv[didx9 + 5] - s_gradv[sidx9 + 5])))

                    d_aw[d_idx] += -(
                        tmp *
                        (dwij[0] * (d_gradv[didx9 + 6] - s_gradv[sidx9 + 6]) +
                        dwij[1] * (d_gradv[didx9 + 7] - s_gradv[sidx9 + 7]) +
                        dwij[2] * (d_gradv[didx9 + 8] - s_gradv[sidx9 + 8])))



def solid_bc(bcs, fluids, rho0, p0):
    import sys
    print(bcs)
    for bc in bcs:
        if bc == 'u_no_slip':
            g0 = [
                FindDistanceFromBoundary(dest='fluid', sources=['boundary'])
            ]
            return [g0]
        if bc == 'u_slip':
            print("slip bc doesn't exist")
            sys.exit(0)
        if bc == 'p_solid':
            g0 = [
                FindDistanceFromBoundary(dest='fluid', sources=['boundary'])
            ]
            return [g0]


def replace_momentum_eqns(eqns):
    for group in eqns:
        for i, equation in enumerate(group.equations):
            kclass = equation.__class__.__name__
            if kclass == 'MomentumEquationSecondOrder':
                dest = equation.dest
                sources = equation.sources
                sources.remove('solid0')
                group.equations[i] = MomentumEquationFourtakas(dest=dest,
                sources=sources)

    return eqns


def replace_viscous_operator(eqns):
    for group in eqns:
        toremove = []
        for i, equation in enumerate(group.equations):
            kclass = equation.__class__.__name__
            if kclass == 'DivGrad':
                dest = equation.dest
                sources = equation.sources
                sources.remove('solid0')
                group.equations[i] = DivGradFourtakas(dest=dest,
                sources=sources, nu=equation.nu, rho0=1)
            if kclass == 'VelocityGradient':
                dest = equation.dest
                sources = equation.sources
                group.equations[i] = VelocityGradientNoSlip(dest=dest,
                sources=sources)
            if kclass == 'VelocityGradientSoild':
                toremove.append(equation)
            if kclass == 'VelocityGradientDestSoild':
                toremove.append(equation)
            if kclass == 'VelocityGradientSolidSoild':
                toremove.append(equation)
        for eq in toremove:
            group.equations.remove(eq)
    return eqns
