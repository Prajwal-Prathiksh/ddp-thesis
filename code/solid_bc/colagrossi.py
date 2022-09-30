'''
[1]A. Colagrossi and M. Landrini, “Numerical simulation of interfacial flows by
smoothed particle hydrodynamics,” Journal of Computational Physics, vol. 191,
no. 2, pp. 448–475, Nov. 2003, doi: 10.1016/S0021-9991(03)00324-3.
'''

from pysph.base.utils import get_particle_array
from pysph.sph.equation import Equation
from compyle.api import declare
import numpy as np


def boundary_props():
    '''
    bid: Boundary particle closest to fluid
    cid: greater than zero if corner particle
    cbid: Boundary particle closed to fluid particle
    '''
    return [{
        'name': 'bid',
        'type': 'int'
    }, {
        'name': 'cbid',
        'type': 'int'
    }, 'proj']


def requires():
    is_mirror = True
    is_boundary = True
    is_ghost = False
    is_ghost_mirror = False
    is_boundary_shift = False

    return is_mirror, is_boundary, is_ghost, is_ghost_mirror, is_boundary_shift


def get_bc_names():
    return ['mirror', 'solid1']


def create_mirror_particles(eval, solver, particles, domain):
    print('In post step colagrossi')
    from pysph.tools.sph_evaluator import SPHEvaluator
    fluid = None
    boundary = None
    mirror = None
    for pa_arr in particles:
        if pa_arr.name == "fluid":
            fluid = pa_arr
        if pa_arr.name == "boundary":
            boundary = pa_arr
        if pa_arr.name == "mirror":
            mirror = pa_arr

    if solver.count < 1:
        eqns=[
            FindFluidNeighbors(dest='fluid', sources=['boundary']),
            FindCornerBoundaryFluidParticles(dest='fluid', sources=['boundary']),
        ]

        eval = SPHEvaluator([fluid, boundary], equations=eqns, dim=solver.dim,
                                kernel=solver.kernel)

    eval.update()
    eval.evaluate()


    bids = fluid.bid
    cond = bids > -1e-14
    bids = np.array(bids[cond], dtype=int)
    ids = np.arange(len(fluid.x))
    x = fluid.x[cond]
    y = fluid.y[cond]
    z = fluid.z[cond]
    proj = fluid.proj[cond]
    xn = boundary.normal[0::3]
    yn = boundary.normal[1::3]
    zn = boundary.normal[2::3]

    xnew = x - 2 * xn[bids] * abs(proj)
    ynew = y - 2 * yn[bids] * abs(proj)
    znew = z - 2 * zn[bids] * abs(proj)
    idnew = ids[np.where(cond)[0]]

    cbids = fluid.cbid
    cond0 = cbids > -1e-14
    idnew0 = ids[np.where(cond0)]
    x = fluid.x[cond0]
    y = fluid.y[cond0]
    z = fluid.z[cond0]
    xb0 = boundary.x[0]
    yb0 = boundary.y[0]
    zb0 = boundary.z[0]
    xb1 = boundary.x[-1]
    yb1 = boundary.y[-1]
    zb1 = boundary.z[-1]

    cond = (x < 0.5)
    xnew0 = x[cond] - 2 * (x[cond]-xb0)
    ynew0 = y[cond] - 2 * (y[cond]-yb0)
    znew0 = z[cond] - 2 * (z[cond]-zb0)

    cond = (x > 0.5)
    xnew1 = x[cond] - 2 * (x[cond]-xb1)
    ynew1 = y[cond] - 2 * (y[cond]-yb1)
    znew1 = z[cond] - 2 * (z[cond]-zb1)

    xnew = np.concatenate([xnew, xnew0, xnew1])
    ynew = np.concatenate([ynew, ynew0, ynew1])
    znew = np.concatenate([znew, znew0, znew1])
    idnew = np.concatenate([idnew, idnew0])
    print(idnew, len(idnew))

    npar = mirror.get_number_of_particles()
    m = fluid.m[0]
    h = fluid.h[0]
    print(npar)
    if npar > len(xnew):
        ids = np.arange(npar-len(xnew))
        mirror.remove_particles(ids)
    elif npar < len(xnew) :
        ids = np.arange(len(xnew) - npar)
        junk = get_particle_array(name='junk', x=np.zeros(len(ids)), m=m, h=h)
        mirror.add_particles(**junk.get_property_arrays())
        print(len(mirror.x), len(ids), len(junk.x), len(xnew))

    print(mirror.h)
    mirror.x = xnew
    mirror.y = ynew
    mirror.z = znew
    mirror.bid = idnew

    return eval


class FindFluidNeighbors(Equation):
    ''' For every fluid particle find the nearest fluid particle in the normal
    direction'''
    def initialize(self, d_idx, d_bid, d_proj):
        d_bid[d_idx] = -1
        d_proj[d_idx] = 10000

    def loop_all(self, d_idx, d_bid, d_x, d_y, d_z, s_x, s_y, s_z, s_normal,
                 NBRS, N_NBRS, d_proj, s_cid):
        s_idx = declare('int')
        dmin = 10000

        for i in range(N_NBRS):
            s_idx = NBRS[i]
            xij = d_x[d_idx] - s_x[s_idx]
            yij = d_y[d_idx] - s_y[s_idx]
            zij = d_z[d_idx] - s_z[s_idx]
            rij = sqrt(xij**2 + yij**2 + zij**2)
            # if s_cid[s_idx] < 0:
            if rij < dmin:
                proj_d = xij * s_normal[3 * s_idx] + yij * s_normal[
                    3 * s_idx + 1] + zij * s_normal[3 * s_idx + 2]
                # if proj_d < d_proj[d_idx]:
                dmin = rij
                d_proj[d_idx] = proj_d
                d_bid[d_idx] = s_idx


class FindCornerBoundaryFluidParticles(Equation):
    ''' For every fluid particle find the nearest corner particle'''
    def initialize(self, d_idx, d_cbid):
        d_cbid[d_idx] = -1

    def loop_all(self, d_idx, d_cbid, d_x, d_y, d_z, s_x, s_y, s_z, s_normal,
                 NBRS, N_NBRS, s_cid):
        s_idx = declare('int')
        dmin = 10000

        for i in range(N_NBRS):
            s_idx = NBRS[i]
            xij = d_x[d_idx] - s_x[s_idx]
            yij = d_y[d_idx] - s_y[s_idx]
            zij = d_z[d_idx] - s_z[s_idx]
            rij = sqrt(xij**2 + yij**2 + zij**2)
            # if s_cid[s_idx] > 0:
            #     if rij < dmin:
            #         dmin = rij
            #         d_cbid[d_idx] = s_idx


class ColagrossiSlipBoundary(Equation):
    def initialize(self, d_idx, d_ug_star, d_vg_star, d_wg_star):
        d_ug_star[d_idx] = 0.0
        d_vg_star[d_idx] = 0.0
        d_wg_star[d_idx] = 0.0

    def initialize_pair(self, d_idx, d_ug_star, d_vg_star, d_wg_star, s_u, s_v, s_w, d_bid, s_normal):
        idx = declare('int')
        idx = d_bid[d_idx]
        un = (s_u[idx] * s_normal[3*idx] + s_v[idx] * s_normal[3*idx+1] + s_w[idx] * s_normal[3*idx+2])
        # plus sign since the normals are inverted
        d_ug_star[d_idx] = s_u[idx] + 2 * un * s_normal[3*idx]
        d_vg_star[d_idx] = s_v[idx] + 2 * un * s_normal[3*idx + 1]
        d_wg_star[d_idx] = s_w[idx] + 2 * un * s_normal[3*idx + 2]


class ColagrossNoSlipBoundary(Equation):
    def initialize(self, d_idx, d_ug, d_vg, d_wg):
        d_ug[d_idx] = 0.0
        d_vg[d_idx] = 0.0
        d_wg[d_idx] = 0.0

    def initialize_pair(self, d_idx, d_ug, d_vg, d_wg, s_u, s_v, s_w, d_bid):
        idx = declare('int')
        idx = d_bid[d_idx]
        d_ug[d_idx] = -s_u[idx]
        d_vg[d_idx] = -s_v[idx]
        d_wg[d_idx] = -s_w[idx]


class ColagrossPressureBoundary(Equation):
    def initialize(self, d_idx, d_p):
        d_p[d_idx] = 0.0

    def initialize_pair(self, d_idx, d_p, s_p, d_bid):
        idx = declare('int')
        idx = d_bid[d_idx]
        d_p[d_idx] = s_p[idx]


def solid_bc(bcs, fluids, rho0, p0):
    print(bcs)
    g0 = []
    for bc in bcs:
        if bc == 'u_no_slip':
            for name in bcs[bc]:
                g0.append(ColagrossNoSlipBoundary(dest='mirror', sources=['fluid']))
        if bc == 'u_slip':
            for name in bcs[bc]:
                g0.append(ColagrossiSlipBoundary(dest='mirror', sources=['fluid']))
        if bc == 'p_solid':
            for name in bcs[bc]:
                g0.append(ColagrossPressureBoundary(dest='mirror', sources=['fluid']))
    print(g0)
    return [g0]