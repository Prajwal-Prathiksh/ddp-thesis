''' [1]J. C. Marongiu, F. Leboeuf, and E. Parkinson, “Numerical simulation
of the flow in a Pelton turbine using the meshless method smoothed particle
hydrodynamics: A new simple solid boundary treatment,” Proceedings of the
Institution of Mechanical Engineers, Part A: Journal of Power and Energy,
vol. 221, no. 6, pp. 849–856, Jan. 2007, doi: 10.1243/09576509JPE465.  '''

from numpy import RankWarning
from pysph.sph.equation import Equation
from pysph.sph.integrator import IntegratorStep
from compyle.api import declare
from pysph.sph.wc.edac import SourceNumberDensity
from solid_bc.takeda import SelfNumberDensity, EvaluateVelocity


def boundary_props():
    '''
    bid: Boundary particle closest to fluid
    cid: greater than zero if corner particle
    cbid: Boundary particle closed to fluid particle
    '''
    return ['wij', { 'name': 'gradbu', 'stride': 3 }, { 'name': 'gradrc', 'stride': 3 }]


def get_bc_names():
    return ['boundary', 'solid1']


def requires():
    is_mirror = False
    is_boundary = True
    is_ghost = False
    is_ghost_mirror = False
    is_boundary_shift = True

    return is_mirror, is_boundary, is_ghost, is_ghost_mirror,\
        is_boundary_shift


class BoundaryRK2Stepper(IntegratorStep):
    def initialize( self, d_idx, d_rhoc, d_rhoc0):
        d_rhoc0[d_idx] = d_rhoc[d_idx]

    def stage1( self, d_idx, dt, d_rhoc, d_arho, d_rhoc0):
        dtb2 = 0.5*dt
        d_rhoc[d_idx] = d_rhoc0[d_idx] + dtb2*d_arho[d_idx]

    def stage2( self, d_idx, dt, d_rhoc, d_arho, d_rhoc0):
        d_rhoc[d_idx] = d_rhoc0[d_idx] + dt*d_arho[d_idx]


class EvaluateProperty(Equation):
    ''' Calculate property on boundary and boundary shift particles to
    calculate gradient
    '''
    def initialize(self, d_idx, d_wij, d_u, d_v, d_w, d_rhoc):
        d_u[d_idx] = 0.0
        d_v[d_idx] = 0.0
        d_w[d_idx] = 0.0
        d_rhoc[d_idx] = 0.0
        d_wij[d_idx] = 0.0

    def loop(self, d_idx, d_u, d_v, d_w, d_wij, WIJ, s_u, s_v, s_w,
             s_idx, d_rhoc, s_rhoc):
        d_u[d_idx] += s_u[s_idx] * WIJ
        d_v[d_idx] += s_v[s_idx] * WIJ
        d_w[d_idx] += s_w[s_idx] * WIJ
        d_rhoc[d_idx] += s_rhoc[s_idx] * WIJ
        d_wij[d_idx] += WIJ

    def post_loop(self, d_idx, d_u, d_v, d_w, d_rhoc, d_wij):
        if d_wij[d_idx] > 1e-14:
            d_u[d_idx] /= d_wij[d_idx]
            d_v[d_idx] /= d_wij[d_idx]
            d_w[d_idx] /= d_wij[d_idx]
            d_rhoc[d_idx] /= d_wij[d_idx]


class EvaluateGradients(Equation):
    '''Coefficient for 5 point stencil
    a = [[1, 1, 1, 1, 1],
         [0, -1, -2, -3, -4],
         [0, 1/2, 2, 9/2, 32/3],
         [0, -1/6, -4/3, -9/2, -128/3],
         [0, 1/24, 2/3, 27/8, 128/3]]
    sol = [ 1.28244838, -1.27433628, -0.31858407,  0.33038348, -0.0199115 ]
    '''
    def initialize(self, d_idx, d_gradrc, d_gradbu):
        i, j = declare('int', 2)
        for i in range(3):
            d_gradrc[3 * d_idx + i] = 0.0
            d_gradbu[3 * d_idx + i] = 0.0

    def initialize_pair(self, d_idx, s_u, s_v, s_w, d_u, d_v, d_w, d_gradbu,
                        d_gradrc, s_rhoc, d_rhoc, d_x, d_y, d_z, s_x, s_y, s_z,
                        d_normal):

        xn = d_normal[3 * d_idx]
        yn = d_normal[3 * d_idx + 1]
        zn = d_normal[3 * d_idx + 2]

        ui = d_u[d_idx]*xn + d_v[d_idx]*yn + d_w[d_idx]*zn
        uim1 = s_u[d_idx]*xn + s_v[d_idx]*yn + s_w[d_idx]*zn
        uim2 = s_u[2*d_idx]*xn + s_v[2*d_idx]*yn + s_w[2*d_idx]*zn
        uim3 = s_u[3*d_idx]*xn + s_v[3*d_idx]*yn + s_w[3*d_idx]*zn
        uim4 = s_u[4*d_idx]*xn + s_v[4*d_idx]*yn + s_w[4*d_idx]*zn

        ri = d_rhoc[d_idx]
        rim1 = s_rhoc[d_idx]
        rim2 = s_rhoc[2 * d_idx]
        rim3 = s_rhoc[3 * d_idx]
        rim4 = s_rhoc[4 * d_idx]

        ci = 1.28244838
        cim1 = -1.27433628
        cim2 = -0.31858407
        cim3 = 0.33038348
        cim4 = -0.0199115

        dx = d_x[d_idx] - s_x[d_idx]
        dy = d_y[d_idx] - s_y[d_idx]
        dz = d_z[d_idx] - s_z[d_idx]

        if dx > 1e-14:
            d_gradbu[3 * d_idx] = (ci * ui + cim1 * uim1 + cim2 * uim2 +
                                   cim3 * uim3 + cim4 * uim4) / dx
            d_gradrc[3 * d_idx] = (ci * ri + cim1 * rim1 + cim2 * rim2 +
                                   cim3 * rim3 + cim4 * rim4) / dx
        if dy > 1e-14:
            d_gradbu[3 * d_idx + 1] = (ci * ui + cim1 * uim1 + cim2 * uim2 +
                                       cim3 * uim3 + cim4 * uim4) / dy
            d_gradrc[3 * d_idx + 1] = (ci * ri + cim1 * rim1 + cim2 * rim2 +
                                       cim3 * rim3 + cim4 * rim4) / dy
        if dz > 1e-14:
            d_gradbu[3 * d_idx + 2] = (ci * ui + cim1 * uim1 + cim2 * uim2 +
                                       cim3 * uim3 + cim4 * uim4) / dz
            d_gradrc[3 * d_idx + 2] = (ci * ri + cim1 * rim1 + cim2 * rim2 +
                                       cim3 * rim3 + cim4 * rim4) / dz


class ComputeBoundaryAcceleration(Equation):
    def initialize(self, d_arho, d_idx, d_normal, d_gradbu, d_gradrc, d_c0,
                   d_rhoc, d_p):

        c0 = d_c0[0]
        xn = d_normal[3 * d_idx]
        yn = d_normal[3 * d_idx + 1]
        zn = d_normal[3 * d_idx + 2]

        drdn = d_gradrc[3 * d_idx] * xn + d_gradrc[ 3 * d_idx + 1] * yn\
             + d_gradrc[3 * d_idx + 2] * zn
        dundn = d_gradbu[3 * d_idx] * xn + d_gradbu[ 3 * d_idx + 1] * yn\
             + d_gradbu[3 * d_idx + 2] * zn

        d_arho[d_idx] = c0 * drdn - d_rhoc[d_idx] * dundn

        d_p[d_idx] = c0**2 * (d_rhoc[d_idx] - 1.0)


def solid_bc(bcs, fluids, rho0, p0):
    import sys
    print(bcs)
    g0 = []
    g1 = []
    g2 = []
    for bc in bcs:
        if bc == 'u_no_slip':
            print("no slip bc doesn't exist")
            sys.exit(0)
        if bc == 'u_slip':
            print("slip bc doesn't exist")
            sys.exit(0)
        if bc == 'p_solid':
            for name in bcs[bc]:
                g0.extend([
                    EvaluateProperty(dest='boundary', sources=['fluid']),
                    EvaluateProperty(dest='boundary_shift', sources=['fluid']),
                ])
                g1.extend([
                    EvaluateGradients(dest='boundary', sources=['boundary_shift']),
                ])
                g2.append(
                    ComputeBoundaryAcceleration(dest='boundary', sources=None)
                )
    return [g0, g1, g2]