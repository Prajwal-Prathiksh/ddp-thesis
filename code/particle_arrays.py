import os
from math import cos, exp, pi, sin

import numpy as np
from compyle.api import Elementwise, annotate, declare, get_config, wrap
from pysph.base.utils import get_particle_array
from pysph.sph.equation import Equation, Group
from pysph.tools.geometry import remove_overlap_particles
from pysph.tools.sph_evaluator import SPHEvaluator

from config_mms import get_props

R0 = 0.25
R1 = 0.5

class ComputeNormalsFromFluid(Equation):
    def initialize(self, d_idx, d_normal, d_tangent):
        d_normal[3*d_idx + 0] = 0.0
        d_normal[3*d_idx + 1] = 0.0
        d_normal[3*d_idx + 2] = 0.0
        d_tangent[3*d_idx + 0] = 0.0
        d_tangent[3*d_idx + 1] = 0.0
        d_tangent[3*d_idx + 2] = 0.0

    def loop(self, d_idx, d_normal, s_normal, WIJ, s_idx, d_tangent, s_tangent):
        d_normal[3*d_idx] += s_normal[3*s_idx] * WIJ
        d_normal[3*d_idx+1] += s_normal[3*s_idx+1] * WIJ
        d_normal[3*d_idx+2] += s_normal[3*s_idx+2] * WIJ
        d_tangent[3*d_idx] += s_tangent[3*s_idx] * WIJ
        d_tangent[3*d_idx+1] += s_tangent[3*s_idx+1] * WIJ
        d_tangent[3*d_idx+2] += s_tangent[3*s_idx+2] * WIJ

    def post_loop(self, d_idx, d_normal, d_tangent):
        mag = sqrt(d_normal[3*d_idx]**2 + d_normal[3*d_idx+1]**2 +d_normal[3*d_idx+2]**2)

        d_normal[3*d_idx] /= -mag
        d_normal[3*d_idx+1] /= -mag
        d_normal[3*d_idx+2] /= -mag

        mag = sqrt(d_tangent[3*d_idx]**2 + d_tangent[3*d_idx+1]**2 +d_tangent[3*d_idx+2]**2)

        d_tangent[3*d_idx] /= mag
        d_tangent[3*d_idx+1] /= mag
        d_tangent[3*d_idx+2] /= mag


def boundary_curve_gauss(x, L):
    return L + (x - 0.5) * np.exp(-30*(x-0.5)**2)


def boundary_curve_incline(x, L):
    return L + 0.5*x


def boundary_curve_line(x, L):
    return L + np.zeros_like(x)


def boundary_curve_cone(x, L):
    y = np.zeros_like(x)
    cond = (x < 0.25)
    y[cond] = L
    cond = (x >= 0.25) & (x < 0.5)
    y[cond] = L + (x[cond] - 0.25)
    cond = (x >= 0.5) & (x < 0.75)
    y[cond] = L + 0.25 - (x[cond] - 0.5)
    cond = (x >= 0.75)
    y[cond] = L
    return y


def boundary_curve(x, L, domain=1):
    if domain == 1:
        return boundary_curve_line(x, L)
    elif domain == 2:
        return boundary_curve_line(x, L)
    elif domain == 3:
        return boundary_curve_gauss(x, L)
    elif domain == 4:
        return boundary_curve_incline(x, L)
    elif domain == 5:
        return boundary_curve_cone(x, L)


def boundary_normal_line(x):
    return np.zeros_like(x), -1 * np.ones_like(x)


def boundary_normal_incline(x):
    xn = 0.5*np.ones_like(x)
    yn = -1 * np.ones_like(x)

    mag = np.sqrt(xn**2 + yn**2)

    return xn/mag, yn/mag


def boundary_normal_line_with_corner(x):
    xn = np.zeros_like(x)
    yn = -1 * np.ones_like(x)

    return xn, yn


def boundary_normal_cone(x):
    xt = np.zeros_like(x)
    yt = np.ones_like(x)
    cond = (x < 0.25)
    xt[cond] = 1.0
    yt[cond] = 0.0
    cond = (x >= 0.25) & (x < 0.5)
    xt[cond] = 1.0
    yt[cond] = 1.0
    cond = (x >= 0.5) & (x < 0.75)
    xt[cond] = 1.0
    yt[cond] = -1.0
    cond = (x >= 0.75)
    xt[cond] = 1.0
    yt[cond] = 0.0

    mag = np.sqrt(xt**2 + yt**2)
    xt = xt/mag
    yt = yt/mag

    projx =  yt*xt
    projy =  yt*yt
    xn = (projx)
    yn = -(1 - projy)

    mag = np.sqrt(xn**2 + yn**2)
    return xn/mag, yn/mag


def boundary_normal_guass(x):
    xt = 1 * np.ones_like(x)
    yt = (1 - 60 * (x - 0.5)**2) * np.exp(-30 * (x - 0.5)**2)
    mag = np.sqrt(xt**2 + yt**2)
    xt = xt/mag
    yt = yt/mag

    projx =  yt*xt
    projy =  yt*yt
    xn = (projx)
    yn = -(1 - projy)

    mag = np.sqrt(xn**2 + yn**2)

    return xn/mag, yn/mag


def boundary_normal(x, domain=1):
    if domain == 1:
        return boundary_normal_line(x)
    elif domain == 2:
        return boundary_normal_line_with_corner(x)
    elif domain == 3:
        return boundary_normal_guass(x)
    elif domain == 4:
        return boundary_normal_incline(x)
    elif domain == 5:
        return boundary_normal_cone(x)


@annotate(double='x0, y0, x1, y1, p0, q0, p1, q1', intersect='doublep')
def intersection_of_two_lines(
    x0, y0, x1, y1, p0, q0, p1, q1, intersect=[0.0, 0.0]):

    slope2 = (q1 - q0) / (p1 - p0)
    intercept2 = q1 - slope2 * p1
    slope1 = 0.0
    intercept1 = 0.0
    if abs(x1 - x0) > 1e-6:
        slope1 = (y1 - y0) / (x1 - x0)
        intercept1 = y1 - slope1 * x1
        dist1 = sqrt((y0 - y1)**2 + (x0 - x1)**2)
        dist2 = sqrt((p0 - p1)**2 + (q0 - q1)**2)

        dot = ((x1 - x0) * (p1 - p0) + (y1 - y0) * (q1 - q0))/(dist1 * dist2)
        if (abs(dot) - 0.99)< 1e-14:
            intersect[0] = -(intercept1 - intercept2) / (slope1 - slope2)
            intersect[1] = slope1 * intersect[0] + intercept1
    else:
        intersect[0] = x0
        intersect[1] = slope2 * intersect[0] + intercept2


@annotate(double='p0, q0, p1, q1, x_int, y_int', return_='int')
def is_inside(p0, q0, p1, q1, x_int, y_int):

    dist1 = sqrt((p0 - x_int)**2 + (q0 - y_int)**2)
    dist2 = sqrt((p0 - p1)**2 + (q0 - q1)**2)

    if (dist1 - dist2 < -1e-14):
        return 1

    return 0


@annotate(i='int', doublep='xs, ys, xn, yn, xb, yb, proj', lenb='int')
def find_distance(i, xs, ys, xn, yn, xb, yb, proj, lenb):
    j = declare('int')
    intersect = declare('matrix(2)')
    for j in range(lenb - 1):
        x0 = xs[i]
        y0 = ys[i]
        x1 = xs[i] + xn[i] * 2.0
        y1 = ys[i] + yn[i] * 2.0
        p0 = xb[j]
        q0 = yb[j]
        p1 = xb[j + 1]
        q1 = yb[j + 1]

        intersection_of_two_lines(x0, y0, x1, y1, p0, q0, p1, q1, intersect)
        if (is_inside(p0, q0, p1, q1, intersect[0], intersect[1])):
            proj[i] = sqrt((x0 - intersect[0])**2 + (y0 - intersect[1])**2)


def find_projections(solid0, boundary):
    xs = solid0.x
    ys = solid0.y

    xn = solid0.normal[0::3].copy()
    yn = solid0.normal[1::3].copy()

    xb = boundary.x
    yb = boundary.y

    proj = np.zeros_like(xs)

    Nb = len(xb)

    backend = 'cython'
    # get_config().use_openmp = True
    xs, ys, xb, yb, xn, yn, proj = wrap(xs, ys, xb, yb, xn, yn, proj, backend=backend)
    e = Elementwise(find_distance, backend=backend)
    e(xs, ys, xn, yn, xb, yb, proj, Nb)

    proj.pull()
    solid0.proj[:] = proj


class SetNormalFromBoundary(Equation):
    def loop_all(self, d_idx, d_normal, s_normal, d_x, d_y, d_z, s_x, s_y, s_z,
                 NBRS, N_NBRS, SPH_KERNEL, d_h):
        s_idx, i = declare('int', 2)
        dmin = 10000
        proj = 10000
        WIJ = 0.0

        for i in range(N_NBRS):
            s_idx = NBRS[i]
            xij = d_x[d_idx] - s_x[s_idx]
            yij = d_y[d_idx] - s_y[s_idx]
            zij = d_z[d_idx] - s_z[s_idx]
            rij = sqrt(xij**2 + yij**2 + zij**2)
            wij = SPH_KERNEL.kernel([xij, yij, zij], rij, d_h[d_idx])
            WIJ += wij
            d_normal[3*d_idx] += s_normal[3*s_idx] * wij
            d_normal[3*d_idx+1] += s_normal[3*s_idx+1] * wij
            d_normal[3*d_idx+2] += s_normal[3*s_idx+2] * wij
        if WIJ > 1e-14:
            d_normal[3*d_idx] /= WIJ
            d_normal[3*d_idx+1] /= WIJ
            d_normal[3*d_idx+2] /= WIJ


class SetNormalUsingNeighbour(Equation):
    def loop_all(self, d_idx, d_normal, s_normal, d_x, d_y, d_z, s_x, s_y, s_z,
                 NBRS, N_NBRS, SPH_KERNEL, d_h):
        s_idx, i = declare('int', 2)
        dmin = 10000
        proj = 10000
        WIJ = 0.0

        for i in range(N_NBRS):
            s_idx = NBRS[i]
            xij = d_x[d_idx] - s_x[s_idx]
            yij = d_y[d_idx] - s_y[s_idx]
            zij = d_z[d_idx] - s_z[s_idx]
            rij = sqrt(xij**2 + yij**2 + zij**2)
            wij = SPH_KERNEL.kernel([xij, yij, zij], rij, d_h[d_idx])
            mag = d_normal[3*d_idx]**2 + d_normal[3*d_idx+1]**2 + d_normal[3*d_idx+2]**2
            if mag < 1e-14:
                WIJ += wij
                d_normal[3*d_idx] += s_normal[3*s_idx] * wij
                d_normal[3*d_idx+1] += s_normal[3*s_idx+1] * wij
                d_normal[3*d_idx+2] += s_normal[3*s_idx+2] * wij


def find_normals(solids, L, dx, mirror=False, domain=1):
    from pysph.base.kernels import QuinticSpline
    from pysph.base.utils import get_particle_array
    from pysph.tools.sph_evaluator import SPHEvaluator

    from wall_normal import ComputeNormals, InvertNormal, SmoothNormals

    name = solids[0].name
    sources = [name]
    if len(solids) == 2:
        sources = ['solid0', 'solid1']

    g0 = []
    eqns  = []
    for dest in sources:
        g0.append(ComputeNormals(dest=dest, sources=sources))
    eqns.append(Group(equations=g0))
    g0 = []
    for dest in sources:
        g0.append(InvertNormal(dest=dest, sources=None))
    if not 'fluid' in sources:
        eqns.append(Group(equations=g0))
    g1 = []
    for dest in sources:
        g1.append(SmoothNormals(dest=dest, sources=sources))
    eqns.append(Group(equations=g1))

    # print(eqns)
    sph_eval = SPHEvaluator(
        solids, equations=eqns, dim=2, kernel=QuinticSpline(dim=2))
    sph_eval.evaluate()

    for i in range(len(solids)):

        cond = []
        if domain < 5:
            x = solids[i].x
            cond = (x < 0.1) | (x > 0.9)
        xn = solids[i].normal[0::3]
        yn = solids[i].normal[1::3]
        if not (domain == 4):
            if solids[0].name == 'solid0':
                xn[cond] = 0.0
                yn[cond] = -1.0
            if solids[0].name == 'fluid':
                xn[cond] = 0.0
                yn[cond] = 1.0
        else:
            if solids[0].name == 'solid0':
                xn[:] = 0.5
                yn[:] = -1.0
            if solids[0].name == 'fluid':
                xn[:] = -0.5
                yn[:] = 1.0


        dxn = np.sqrt(xn**2 + yn**2)
        cond = dxn>1e-14
        xn[cond] /= dxn[cond]
        yn[cond] /= dxn[cond]

        xt = -1 * yn
        yt = 0 + xn

        solids[i].normal[0::3] = xn
        solids[i].normal[1::3] = yn
        solids[i].normal[2::3] = 0.0

        if ('tangent' in solids[0].get_property_arrays()):
            solids[i].tangent[0::3] = xt
            solids[i].tangent[1::3] = yt
            solids[i].tangent[2::3] = 0.0


def find_projection(solids, domain, L, dx):
    if domain == 5:
        x = solids.x
        y = solids.y
        dist = np.sqrt((x-0.5)**2 + (y-0.5)**2)
        solids.proj = dist - R0
    elif domain == 6:
        x = solids.x
        y = solids.y
        dist = np.sqrt((x-0.5)**2 + (y-0.5)**2)
        solids.proj = R1 - dist
    else:
        x = np.mgrid[0-10*dx:L+10*dx:dx]
        y = boundary_curve(x, L, domain)

        h = solids.h[0]
        m= solids.m[0]
        boundary = get_particle_array(name='boundary', x=x, y=y, m=m, h=2*h)
        boundary.add_property('normal', stride=3)
        normal = boundary_normal(x, domain)
        boundary.normal[0::3] = normal[0]
        boundary.normal[1::3] = normal[1]
        boundary.normal[2::3] = 0.0

        find_projections(solids, boundary)


def create_boundary_mirror(name, particles, app, m, h, rho0):
    '''Mirror to the fluid particles created during the simulation'''
    xm = np.zeros(1)
    ym = np.zeros(1)
    usm, vsm, wsm, rhocsm, psm = get_props(
        xm, ym, 0.0, 0.0, app.c0, app.mms)
    b_mirror = get_particle_array(
        name=name, x=xm, y=ym, m=m, h=h, rho=rho0,
        rhoc=rhocsm, u=usm, v=vsm, w=wsm, p=psm)
    particles.append(b_mirror)


def create_ghost_mirror(name, particles, solid0, app, m, h, rho0):
    x = solid0.x
    y = solid0.y
    z = solid0.z
    xn = solid0.normal[0::3]
    yn = solid0.normal[1::3]
    zn = solid0.normal[2::3]
    proj = solid0.proj
    xgm, ygm, zgm = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)

    xgm = x + 2* abs(proj) * xn
    ygm = y + 2* abs(proj) * yn
    zgm = z + 2* abs(proj) * zn

    usgm, vsgm, wsgm, rhocsgm, psgm = get_props(
        xgm, ygm, 0.0, 0.0, app.c0, app.mms)
    ghost_mirror = get_particle_array(
        name=name, x=xgm, y=ygm, m=m, h=h, rho=rho0,
        rhoc=rhocsgm, u=usgm, v=vsgm, w=wsgm, p=psgm)
    particles.append(ghost_mirror)


def create_boundary_shifts(name, particles, boundary, dx, app, m, h, rho0):


    xb = boundary.x
    yb = boundary.y
    zb = boundary.z
    xn = boundary.normal[0::3]
    yn = boundary.normal[1::3]
    zn = boundary.normal[2::3]
    ids = np.arange(len(xb))
    x, y, z, bid = [], [], [], []
    for i in range(4): # requires four layer of particles
        x.extend(list(xb + (i+1) * xn * dx))
        y.extend(list(yb + (i+1) * yn * dx))
        z.extend(list(zb + (i+1) * zn * dx))
        bid.extend(ids)

    x, y, z, bid = [np.array(t) for t in (x, y, z, bid)]
    u, v, w, rhoc, p = get_props(
        x, y, z, 0.0, app.c0, app.mms)
    boundary_shift = get_particle_array(
        name=name, x=x, y=y, z=z, m=m, h=h, rho=rho0,
        rhoc=rhoc, u=u, v=v, w=w, p=p)

    boundary_shift.add_property('bid', type='int')
    boundary_shift.bid[:] = bid

    particles.append(boundary_shift)


def create_square_fluid(name, dx, L, m, h, rho, app, domain, pack=False):
    xf, yf = np.mgrid[dx / 2:L:dx, dx / 2:2 * L:dx]
    xf, yf = [np.ravel(t) for t in (xf, yf)]
    # if pack:
    #     from packing import get_marrone_boundary
    #     xs0, ys0, xf, yf = create_packed_fluid(dx, L, nl)
    #     # xs0, ys0, xn0, yn0 = get_marrone_boundary(L, dx)
    cond = yf <= boundary_curve(xf, L, domain)

    uf, vf, wf, rhocf, pf = get_props(xf[cond], yf[cond], 0.0, 0.0, app.c0,
                                      app.mms)
    # if app.options.method == 'org':
    #     rho = rhocf.copy()
    fluid = get_particle_array(
        name=name, x=xf[cond], y=yf[cond], m=m, h=h, rho=rho, rhoc=rhocf,
        u=uf, v=vf, w=wf, p=pf)

    fluid.add_property('normal', stride=3)
    fluid.add_property('tangent', stride=3)
    fluid.add_property('normal_tmp', stride=3)
    find_normals([fluid], L, dx, False, domain)

    particles = [fluid]

    return particles


def read_packed_data(nx=50):
    folder = os.path.join("code", "mesh")
    filename = os.path.join(folder, "nx_%d_circ.npz"%nx)
    data = np.load(filename)
    x = data['x']
    y = data['y']
    return x, y


def create_circular_fluid(name, dx, L, m, h, rho, app, domain, pack=False):
    xf, yf = np.mgrid[dx / 2:L:dx, dx / 2:2 * L:dx]

    if (pack):
        xf, yf =  read_packed_data(nx=int(1/dx))
        xf += 0.5
        yf += 0.5

    xf, yf = [np.ravel(t) for t in (xf, yf)]

    radius = np.sqrt((xf-0.5)**2 + (yf-0.5)**2)
    cond = (radius - R0 > 1e-14) & (radius - R1 < 1e-14)

    uf, vf, wf, rhocf, pf = get_props(xf[cond], yf[cond], 0.0, 0.0, app.c0,
                                      app.mms)
    # if app.options.method == 'org':
    #     rho = rhocf.copy()
    fluid = get_particle_array(
        name=name, x=xf[cond], y=yf[cond], m=m, h=h, rho=rho, rhoc=rhocf,
        u=uf, v=vf, w=wf, p=pf)

    fluid.add_property('normal', stride=3)
    fluid.add_property('tangent', stride=3)
    fluid.add_property('normal_tmp', stride=3)
    find_normals([fluid], L, dx, False, domain)

    # if domain == 6:
    #     fluid.normal[:] = -1 * fluid.normal[:]

    particles = [fluid]
    return particles


def create_square_MMS_boundary(name, particles, nl, dx, L, domain, app, m, h, rho0):
    xs, ys = np.mgrid[dx / 2 - nl:L + nl:dx, dx / 2 - nl:2*L + nl:dx]
    cond0 = (xs>0) & (xs<L) & (ys>0) & (ys<= boundary_curve(xs, L, domain))
    cond1 = ~cond0 & (ys<= boundary_curve(xs, L, domain))
    cond2 = ~cond0 &  ((xs<0)|(xs>L)) & (ys < (boundary_curve(xs, L, domain)+nl))
    cond3 = cond2 | cond1
    us1, vs1, ws1, rhocs1, ps1 = get_props(xs[cond3], ys[cond3], 0.0, 0.0,
                                           app.c0, app.mms)

    solid1 = get_particle_array(
        name=name, x=xs[cond3], y=ys[cond3], m=m, h=h, rho=rho0,
        rhoc=rhocs1, u=us1, v=vs1, w=ws1, p=ps1, proj=0)
    solid1.add_property('normal', stride=3)
    solid1.add_property('tangent', stride=3)
    solid1.add_property('normal_tmp', stride=3)
    particles.append(solid1)


def create_circle_inner_boundary(name, particles, nl, dx, L, domain, app, m, h, rho0, pack, ghost_mirror=False):
    xs, ys = np.mgrid[dx / 2 - nl:L + nl:dx, dx / 2 - nl:2*L + nl:dx]

    if (pack):
        xs, ys =  read_packed_data(nx=int(1/dx))
        xs += 0.5
        ys += 0.5

    radius = np.sqrt((xs-0.5)**2 + (ys-0.5)**2)
    cond3 = (radius - R0 + nl > 1e-14)  & (radius - R0 < 1e-14)
    us1, vs1, ws1, rhocs1, ps1 = get_props(xs[cond3], ys[cond3], 0.0, 0.0,
                                           app.c0, app.mms)

    solid1 = get_particle_array(
        name=name, x=xs[cond3], y=ys[cond3], m=m, h=h, rho=rho0,
        rhoc=rhocs1, u=us1, v=vs1, w=ws1, p=ps1, proj=0)
    solid1.add_property('normal', stride=3)
    solid1.add_property('tangent', stride=3)

    xn = xs[cond3] - 0.5
    yn = ys[cond3] - 0.5
    dxn = np.sqrt(xn**2 + yn**2)
    solid1.normal[0::3] = xn/dxn
    solid1.normal[1::3] = yn/dxn
    solid1.tangent[0::3] = -yn/dxn
    solid1.tangent[1::3] = xn/dxn

    particles.append(solid1)


def create_square_test_boundary(name, particles, solid1, fluid, dx, L, nl, domain, app, m, h, rho0, pack, ghost_mirror=False):
    xs, ys = np.mgrid[dx / 2 - nl:L + nl:dx, dx / 2 - nl:2 * L + nl:dx]
    # if pack:
    #     xs, ys = xs0, ys0
    cond0 = ((ys > boundary_curve(xs, L, domain)) &
            (ys < boundary_curve(xs, L, domain) + nl)) & ((xs > 0) & (xs < L))
    if not pack:
        cond0 = ((ys < boundary_curve(xs, L, domain) + nl))


    us0, vs0, ws0, rhocs0, ps0 = get_props(xs[cond0], ys[cond0], 0.0, 0.0,
                                        app.c0, app.mms)
    solid0 = get_particle_array(
        name=name, x=xs[cond0], y=ys[cond0], m=m, h=h, rho=rho0,
        rhoc=rhocs0, u=us0, v=vs0, w=ws0, p=ps0, proj=0)

    if not pack:
        remove_overlap_particles(solid0, fluid, dx, 2)
        remove_overlap_particles(solid0, solid1, dx, 2)

    solid0.add_property('normal', stride=3)
    solid0.add_property('tangent', stride=3)
    solid0.add_property('normal_tmp', stride=3)
    find_normals([solid0, solid1], L, dx, mirror=ghost_mirror, domain=domain)
    particles.append(solid0)


def create_circle_outer_boundary(name, particles, dx, L, nl, domain, app, m, h, rho0, pack, ghost_mirror=False):
    xs, ys = np.mgrid[dx / 2 - nl:L + nl:dx, dx / 2 - nl:2*L + nl:dx]

    if (pack):
        xs, ys =  read_packed_data(nx=int(1/dx))
        xs += 0.5
        ys += 0.5

    radius = np.sqrt((xs-0.5)**2 + (ys-0.5)**2)
    cond0 = (radius - R1 > 1e-14)  & (radius - R1 - nl < 1e-14)

    us0, vs0, ws0, rhocs0, ps0 = get_props(xs[cond0], ys[cond0], 0.0, 0.0,
                                        app.c0, app.mms)
    solid0 = get_particle_array(
        name=name, x=xs[cond0], y=ys[cond0], m=m, h=h, rho=rho0,
        rhoc=rhocs0, u=us0, v=vs0, w=ws0, p=ps0, proj=0)

    solid0.add_property('normal', stride=3)
    solid0.add_property('tangent', stride=3)
    xn = xs[cond0] - 0.5
    yn = ys[cond0] - 0.5
    dxn = np.sqrt(xn**2 + yn**2)
    solid0.normal[0::3] = -xn/dxn
    solid0.normal[1::3] = -yn/dxn
    solid0.tangent[0::3] = yn/dxn
    solid0.tangent[1::3] = -xn/dxn
    particles.append(solid0)


def create_square_1_layer_boundary(name, particles, fluid, dx, L, app, m, h, rho0, domain):
    xb = np.mgrid[dx / 2:L:dx]
    yb = boundary_curve(xb, L, domain)
    edge = np.arange(len(xb)) + 1
    edge[-1] = -1
    usb, vsb, wsb, rhocsb, psb = get_props(
        xb, yb, 0.0, 0.0, app.c0, app.mms)
    boundary = get_particle_array(
        name=name, x=xb, y=yb, m=m, h=h, rho=rho0,
        rhoc=rhocsb, u=usb, v=vsb, w=wsb, p=psb)
    boundary.add_property('normal', stride=3)
    boundary.add_property('tangent', stride=3)

    eqn = Group(equations=[ComputeNormalsFromFluid(dest='boundary', sources=['fluid'])])
    eval = SPHEvaluator([fluid, boundary], [eqn], dim=2)
    eval.evaluate()

    boundary.add_property('cid')
    boundary.add_property('edge')

    boundary.edge = edge

    boundary.cid[:] = -1
    boundary.cid[0] = 1
    boundary.cid[-1] = 1
    particles.append(boundary)


def create_inner_1_layer_boundary(name, particles, fluid, dx, L, app, m, h, rho0, domain):
    length = 2 * np.pi * R0
    n = int(round(length/dx, 1))
    thetas = np.mgrid[0:2*np.pi:n*1j]
    thetas = thetas[:-1]

    xb = []
    yb = []

    for theta in thetas:
        xb.append(R0 * cos(theta) + 0.5)
        yb.append(R0 * sin(theta) + 0.5)

    xb = np.array(xb)
    yb = np.array(yb)

    usb, vsb, wsb, rhocsb, psb = get_props(
        xb, yb, 0.0, 0.0, app.c0, app.mms)
    boundary = get_particle_array(
        name=name, x=xb, y=yb, m=m, h=h, rho=rho0,
        rhoc=rhocsb, u=usb, v=vsb, w=wsb, p=psb)
    boundary.add_property('normal', stride=3)
    boundary.add_property('tangent', stride=3)

    eqn = Group(equations=[ComputeNormalsFromFluid(dest='boundary', sources=['fluid'])])
    eval = SPHEvaluator([fluid, boundary], [eqn], dim=2)
    eval.evaluate()

    boundary.add_property('cid')
    boundary.add_property('edge')

    boundary.edge[:] = 1

    boundary.cid[:] = -1
    boundary.cid[0] = 1
    boundary.cid[-1] = 1
    particles.append(boundary)


def create_outer_1_layer_boundary(name, particles, fluid, dx, L, app, m, h, rho0, domain):
    length = 2 * np.pi * R1
    n = int(round(length/dx, 1))
    thetas = np.mgrid[0:2*np.pi:n*1j]
    thetas = thetas[:-1]

    xb = []
    yb = []

    for theta in thetas:
        xb.append(R1 * cos(theta) + 0.5)
        yb.append(R1 * sin(theta) + 0.5)

    xb = np.array(xb)
    yb = np.array(yb)

    usb, vsb, wsb, rhocsb, psb = get_props(
        xb, yb, 0.0, 0.0, app.c0, app.mms)
    boundary = get_particle_array(
        name=name, x=xb, y=yb, m=m, h=h, rho=rho0,
        rhoc=rhocsb, u=usb, v=vsb, w=wsb, p=psb)
    boundary.add_property('normal', stride=3)
    boundary.add_property('tangent', stride=3)

    eqn = Group(equations=[ComputeNormalsFromFluid(dest='boundary', sources=['fluid'])])
    eval = SPHEvaluator([fluid, boundary], [eqn], dim=2)
    eval.evaluate()

    boundary.add_property('cid')
    boundary.add_property('edge')

    boundary.edge[:] = 1

    boundary.cid[:] = -1
    boundary.cid[0] = 1
    boundary.cid[-1] = 1
    particles.append(boundary)


def create_particles(app, rho0, L, b_mirror=False, boundary=False,
                     ghost=False, ghost_mirror=False,
                     boundary_shift=False, domain=1, pack=False):

    dx = app.dx
    nl = app.nl * dx
    xf, yf = np.mgrid[dx / 2:L:dx, dx / 2:2 * L:dx]
    xf, yf = [np.ravel(t) for t in (xf, yf)]

    # Initialize
    m = app.volume * rho0
    V0 = app.volume
    h = app.hdx * dx
    rho = rho0

    xs0, ys0 = None, None
    particles = None
    name = 'fluid'
    if (domain < 5): particles = create_square_fluid(
        name, dx, L, m, h, rho, app, domain, pack)
    elif (domain >= 5): particles = create_circular_fluid(
        name, dx, L, m, h, rho, app, domain, pack)

    name = 'solid1'
    if (domain < 5): create_square_MMS_boundary(
        name, particles, nl, dx, L, domain, app, m, h, rho0)
    elif (domain == 5): create_circle_outer_boundary(
        name, particles, dx, L, nl, domain, app, m, h, rho0, pack, ghost_mirror
    )
    elif (domain == 6): create_circle_inner_boundary(
        name, particles, nl, dx, L, domain, app, m, h, rho0, pack, ghost_mirror
    )

    if ghost:
        name = 'solid0'
        if (domain < 5): create_square_test_boundary(
            name, particles, particles[1], particles[0], dx, L, nl,
            domain, app, m, h, rho0, pack, ghost_mirror)
        elif (domain == 5): create_circle_inner_boundary(
            name, particles, nl, dx, L, domain, app, m, h, rho0, pack, ghost_mirror
        )
        elif (domain == 6): create_circle_outer_boundary(
            name, particles, dx, L, nl, domain, app, m, h, rho0, pack, ghost_mirror
        )

    if boundary:
        name = 'boundary'
        if (domain < 5): create_square_1_layer_boundary(
            name, particles, particles[0], dx, L, app, m, h, rho0, domain)
        elif (domain == 5): create_inner_1_layer_boundary(
            name, particles, particles[0], dx, L, app, m, h, rho0, domain
        )
        elif (domain == 6): create_outer_1_layer_boundary(
            name, particles, particles[0], dx, L, app, m, h, rho0, domain
        )

    if b_mirror:
        name = 'mirror'
        create_boundary_mirror(name, particles, app, m, h, rho0)

    if boundary_shift:
        if boundary == None:
            import sys
            print('Cannot be created without boundary')
            sys.exit(0)
        name = 'boundary_shift'
        create_boundary_shifts(name, particles, particles[-1], dx, app, m, h, rho0)


    if ghost_mirror == True:
        ''' mirror to the ghost particles created inside the fluid remains fixed
        during the simulation'''
        find_projection(particles[2], domain, L, dx)

        if ghost == None:
            import sys
            print('Cannot be created without ghost')
            sys.exit(0)
        name = 'ghost_mirror'
        create_ghost_mirror(name, particles, particles[2], app, m, h, rho0)

    return particles


def create_particles_io(app, rho0, L, mirror_inlet=False, mirror_outlet=False):
    import os

    from pysph.base.utils import get_particle_array
    from pysph.tools.geometry import remove_overlap_particles

    dx = app.dx
    nl = app.nl * dx
    xf, yf = np.mgrid[dx / 2:L:dx, dx / 2:L:dx]
    xf, yf = [np.ravel(t) for t in (xf, yf)]

    xs, ys = np.mgrid[dx / 2 - nl:L + nl:dx, dx / 2 - nl:L + nl:dx]
    cond = ((ys < 0) | (ys > L))

    # Initialize
    m = app.volume * rho0
    V0 = app.volume
    h = app.hdx * dx
    rho = rho0

    uf, vf, wf, rhocf, pf = get_props(xf, yf, 0.0, 0.0, app.c0, app.mms)
    fluid = get_particle_array(name='fluid', x=xf, y=yf, m=m, h=h, rho=rho,
                               rhoc=rhocf, u=uf, v=vf, w=wf, p=pf, xn=0, yn=0, zn=0)

    uw, vw, ww, rhocw, pw = get_props(xs[cond], ys[cond], 0.0, 0.0, app.c0,
                                      app.mms)
    wall = get_particle_array(name='wall', x=xs[cond], y=ys[cond], m=m,
                              h=h, rho=rho, rhoc=rhocw, u=uw, v=vw,
                              w=ww, p=pw)

    condi = (xs < 0.5) & (~cond)
    ui, vi, wi, rhoci, pi = get_props(xs[condi], ys[condi], 0.0, 0.0, app.c0,
                                      app.mms)
    inlet = get_particle_array(name='inlet', x=xs[condi], y=ys[condi], m=m,
                               h=h, rho=rho, rhoc=rhoci, u=ui, v=vi,
                               w=wi, p=pi, xn=-1.0, yn=0.0, zn=0.0)

    condo = (xs > 0.5) & (~cond)
    uo, vo, wo, rhoco, po = get_props(xs[condo], ys[condo], 0.0, 0.0, app.c0,
                                      app.mms)
    outlet = get_particle_array(name='outlet', x=xs[condo], y=ys[condo], m=m,
                                h=h, rho=rho, rhoc=rhoco, u=uo, v=vo,
                                w=wo, p=po, xn=1.0, yn=0.0, zn=0.0)

    remove_overlap_particles(inlet, fluid, dx, dim=2)
    remove_overlap_particles(outlet, fluid, dx, dim=2)
    particles = [fluid, wall, inlet, outlet]

    for pa in particles:
        pa.add_constant('uref', 1.0)

    if mirror_inlet == True:
        ''' mirror to the inlet particles created inside the fluid remains fixed
        during the simulation'''
        x = inlet.x
        y = inlet.y
        z = inlet.z
        xgm, ygm, zgm = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)

        xgm = -x
        ygm = y
        zgm = z

        usgm, vsgm, wsgm, rhocsgm, psgm = get_props(
            xgm, ygm, 0.0, 0.0, app.c0, app.mms)
        mirror_inlet = get_particle_array(
            name='mirror_inlet', x=xgm, y=ygm, m=m, h=h, rho=rho0,
            rhoc=rhocsgm, u=usgm, v=vsgm, w=wsgm, p=psgm)
        particles.append(mirror_inlet)

    if mirror_outlet == True:
        ''' mirror to the inlet particles created inside the fluid remains fixed
        during the simulation'''
        x = outlet.x
        y = outlet.y
        z = outlet.z
        xgm, ygm, zgm = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)

        xgm = 2*L - x
        ygm = y
        zgm = z

        usgm, vsgm, wsgm, rhocsgm, psgm = get_props(
            xgm, ygm, 0.0, 0.0, app.c0, app.mms)
        mirror_outlet = get_particle_array(
            name='mirror_outlet', x=xgm, y=ygm, m=m, h=h, rho=rho0,
            rhoc=rhocsgm, u=usgm, v=vsgm, w=wsgm, p=psgm)
        particles.append(mirror_outlet)

    return particles

