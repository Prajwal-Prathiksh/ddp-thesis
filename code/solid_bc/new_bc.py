from re import I
from pysph.sph.equation import Equation
from compyle.api import declare
from pysph.sph.basic_equations import SummationDensity
from pysph.sph.wc.edac import SourceNumberDensity
from solid_bc.takeda import SelfNumberDensity, EvaluateVelocity
from solid_bc.marrone import LiuCorrection, LiuCorrectionPreStep
from pysph.sph.wc.linalg import gj_solve, augmented_matrix, identity

RELAX = 0.001

def boundary_props():
    '''
    bid: Boundary particle closest to fluid
    cid: greater than zero if corner particle
    cbid: Boundary particle closed to fluid particle
    '''
    return [{'name':'bid', 'type':'int'}, 'graddotn', 'wij', 'wijn',
            'swij', 'V', {'name':'L', 'stride':16},
            'vdott', 'vdotn',  {'name':'gradvdotn', 'stride':3},]


def get_bc_names():
    return ['solid0', 'solid1']


def has_iterative(bcs):
    if 'u_slip' in bcs:
        return [2, 3]
    elif 'u_no_slip' in bcs:
        # return [1, 2, 3]
        return [2, 3]
    elif 'p_solid' in bcs:
        return [2, 3]

def requires():
    is_mirror = False
    is_boundary = True
    is_ghost = True
    is_ghost_mirror = True
    is_boundary_shift = False

    return is_mirror, is_boundary, is_ghost, is_ghost_mirror, is_boundary_shift


class SetValuesonMirror(Equation):
    def initialize(self, d_idx, d_p, d_u, d_v, d_w):
        d_p[d_idx] = 0.0
        d_u[d_idx] = 0.0
        d_v[d_idx] = 0.0
        d_w[d_idx] = 0.0

    def loop_all(self, d_idx, d_bid, d_x, d_y, d_z, s_x, s_y, s_z, N_NBRS, NBRS, d_h,
                 s_u, s_v, s_w, s_p, d_p, d_u, d_v, d_w):
        s_idx = declare('int')
        dmin = 10000

        for i in range(N_NBRS):
            s_idx = NBRS[i]
            xij = d_x[d_idx] - s_x[s_idx]
            yij = d_y[d_idx] - s_y[s_idx]
            zij = d_z[d_idx] - s_z[s_idx]
            rij = sqrt(xij**2 + yij**2 + zij**2)
            if (rij < dmin):
                dmin = rij
                d_p[d_idx] = s_p[s_idx]
                d_u[d_idx] = s_u[s_idx]
                d_v[d_idx] = s_v[s_idx]
                d_w[d_idx] = s_w[s_idx]


class FindNearestSolidParticle(Equation):
    def initialize(self, d_idx, d_bid, d_p):
        # bid is the parameter which allows the changes in the property by
        # the iterative solver -1 means no change
        d_bid[d_idx] = -1

    def loop_all(self, d_idx, d_bid, d_x, d_y, d_z, s_x, s_y, s_z, N_NBRS, NBRS, d_h):
        s_idx = declare('int')
        dmin = 10000

        for i in range(N_NBRS):
            s_idx = NBRS[i]
            xij = d_x[d_idx] - s_x[s_idx]
            yij = d_y[d_idx] - s_y[s_idx]
            zij = d_z[d_idx] - s_z[s_idx]
            rij = sqrt(xij**2 + yij**2 + zij**2)
            if ((rij < 1*d_h[d_idx]) & (rij < dmin)):
                dmin = rij
                d_bid[d_idx] = s_idx


class FindGradientVeldotNormal(Equation):
    def initialize(self, d_idx, d_gradvdotn):
        d_gradvdotn[3*d_idx] = 0.0
        d_gradvdotn[3*d_idx + 1] = 0.0
        d_gradvdotn[3*d_idx + 2] = 0.0

    def loop(self, d_idx, d_gradvdotn, DWIJ, d_normal, s_idx, s_m, s_rho, s_u, s_v, s_w):
        omega = s_m[s_idx] / s_rho[s_idx]
        d_gradvdotn[3 * d_idx] += s_u[s_idx] * (DWIJ[0] * d_normal[3 * d_idx] +
                DWIJ[1] * d_normal[3 * d_idx + 1]) * omega
        d_gradvdotn[3 * d_idx + 1] += s_v[s_idx] * (DWIJ[0] * d_normal[3 * d_idx] +
                DWIJ[1] * d_normal[3 * d_idx + 1]) * omega
        d_gradvdotn[3 * d_idx + 2] += s_w[s_idx] * (DWIJ[0] * d_normal[3 * d_idx] +
                DWIJ[1] * d_normal[3 * d_idx + 1]) * omega


class FindGradientVeldotNormalSolid(Equation):
    def initialize(self, d_idx, d_gradvdotn):
        d_gradvdotn[3*d_idx] = 0.0
        d_gradvdotn[3*d_idx + 1] = 0.0
        d_gradvdotn[3*d_idx + 2] = 0.0

    def loop(self, d_idx, d_gradvdotn, DWIJ, d_normal, s_idx, s_m, s_rho, s_ug, s_vg, s_wg, s_bid, d_wijn):
        omega = s_m[s_idx] / s_rho[s_idx]
        d_gradvdotn[3 * d_idx] += s_ug[s_idx] * (DWIJ[0] * d_normal[3 * d_idx] +
                DWIJ[1] * d_normal[3 * d_idx + 1]) * omega
        d_gradvdotn[3 * d_idx + 1] += s_vg[s_idx] * (DWIJ[0] * d_normal[3 * s_idx] +
                DWIJ[1] * d_normal[3 * d_idx + 1]) * omega
        d_gradvdotn[3 * d_idx + 2] += s_wg[s_idx] * (DWIJ[0] * d_normal[3 * d_idx] +
                DWIJ[1] * d_normal[3 * d_idx + 1]) * omega
        if (d_idx == s_bid[s_idx]):
            d_wijn[d_idx] = (DWIJ[0] * d_normal[3 * d_idx] +
                DWIJ[1] * d_normal[3 * d_idx + 1]) * omega


class FindVelocity(Equation):
    def initialize(self, d_idx, d_u, d_v, d_w):
        d_u[d_idx] = 0.0
        d_v[d_idx] = 0.0
        d_w[d_idx] = 0.0

    def loop(self, d_idx, d_u, d_v, d_w, d_wij, s_idx, s_m, s_rho,
             s_u, s_v, s_w):
        omega = s_m[s_idx] / s_rho[s_idx]
        d_u[d_idx] += s_u[s_idx] * d_wij[d_idx] * omega
        d_v[d_idx] += s_v[s_idx] * d_wij[d_idx] * omega
        d_w[d_idx] += s_w[s_idx] * d_wij[d_idx] * omega


class FindVelocitySolid(Equation):
    def __init__(self, dest, sources):
        self.converge = -1.0
        super(FindVelocitySolid, self).__init__(dest, sources)

    def initialize(self, d_idx, d_u, d_v, d_w):
        d_u[d_idx] = 0.0
        d_v[d_idx] = 0.0
        d_w[d_idx] = 0.0

    def loop(self, d_idx, d_u, d_v, d_w, d_wij, s_idx, s_m, s_rho,
             s_ug, s_vg, s_wg):
        omega = s_m[s_idx] / s_rho[s_idx]
        d_u[d_idx] += s_ug[s_idx] * d_wij[d_idx] * omega
        d_v[d_idx] += s_vg[s_idx] * d_wij[d_idx] * omega
        d_w[d_idx] += s_wg[s_idx] * d_wij[d_idx] * omega

    def reduce(self, dst, t, dt):
        # sumgrad = sum(numpy.abs(dst.graddotn))/len(dst.graddotn)
        sumgrad = max(numpy.abs(dst.u))
        h = dst.h[0]
        print(sumgrad, h, 'comp', self.converge, dst.name, t)
        if (sumgrad < 1e-6):
            print("converge")
            self.converge = 1.0
        else:
            self.converge = -1.0

    def converged(self):
        print(self.converge, 'in grad')
        return self.converge


class ComputeNoSlipVelocity(Equation):
    def __init__(self, dest, sources):
        self.relax = RELAX
        self.converge = -1.0
        super(ComputeNoSlipVelocity, self).__init__(dest, sources)

    def loop(self, d_idx, s_idx, d_m, d_rho, d_bid, s_normal, WIJ, s_gradvdotn, d_ug, d_vg, d_wg,
             s_wijn, d_h):
        bid = declare('int')
        omega = d_m[d_idx] / d_rho[d_idx]
        bid = d_bid[d_idx]

        # if (s_idx == bid):
        #     den = s_wijn[s_idx] #(DWIJ[0] * s_normal[3 * s_idx] +
        #         # DWIJ[1] * s_normal[3 * s_idx + 1]) * omega
        #     # print(d_p[d_idx], d_idx, s_graddotn[s_idx], den)
        #     # the plus in numerator/ minus in deno is there as the
        #     # derivative is inverted
        #     delta = d_ug[d_idx] + (s_gradvdotn[3*s_idx] + d_ug[d_idx] * den) / (-den)
        #     d_ug[d_idx] -= delta * d_h[d_idx]
        #     delta = d_vg[d_idx] + (s_gradvdotn[3*s_idx + 1] + d_vg[d_idx] * den) / (-den)
        #     d_vg[d_idx] -= delta *  d_h[d_idx]
        #     delta = d_wg[d_idx] + (s_gradvdotn[3*s_idx + 2] + d_wg[d_idx] * den) / (-den)
        #     d_wg[d_idx] -= delta * d_h[d_idx]

    def reduce(self, dst, t, dt):
        # sumgrad = sum(numpy.abs(dst.graddotn))/len(dst.graddotn)
        sumgrad = max(numpy.abs(dst.gradvdotn))
        h = dst.h[0]
        print(sumgrad, h, 'comp', self.converge, dst.name, t)
        if (sumgrad < h):
            print("converge")
            self.converge = 1.0
        else:
            self.converge = -1.0

    def converged(self):
        print(self.converge, 'in grad')
        return self.converge


class ComputeNoSlipVelocity2(Equation):
    def __init__(self, dest, sources):
        self.relax = RELAX
        self.converge = -1.0
        super(ComputeNoSlipVelocity2, self).__init__(dest, sources)

    def loop_all(self, d_idx, d_m, d_rho, d_bid, NBRS, N_NBRS, s_u, s_v, s_w, d_ug, d_vg, d_wg,
                 d_x, d_y, d_z, s_x, s_y, s_z, SPH_KERNEL, d_h):
        bid, s_idx, i = declare('int', 3)
        omega = d_m[d_idx] / d_rho[d_idx]
        bid = d_bid[d_idx]
        numu, den = 0.0, 0.0
        numv = 0.0

        if (bid > -1):
            for i in range(N_NBRS):
                s_idx = NBRS[i]
                xij = d_x[d_idx] - s_x[s_idx]
                yij = d_y[d_idx] - s_y[s_idx]
                zij = d_z[d_idx] - s_z[s_idx]
                rij = sqrt(xij**2 + yij**2 + zij**2)
                wij = SPH_KERNEL.kernel([0, 0, 0], rij, d_h[d_idx])
                if (bid == s_idx):
                    numu += omega * wij * s_u[s_idx]
                    den += omega * omega * wij * wij
                    numv += omega * wij * s_v[s_idx]

            # printf("%f, %f, %f\n", numu, den, numv)
            if (abs(den) > 1e-14):
                d_ug[d_idx] -= numu/den * 0.1 #d_h[d_idx]
                d_vg[d_idx] -= numv/den * 0.1 #d_h[d_idx]


class FindVelocitydotNormal(Equation):
    def initialize(self, d_idx, d_u, d_v, d_w):
        d_u[d_idx] = 0.0
        d_v[d_idx] = 0.0
        d_w[d_idx] = 0.0

    def loop(self, d_idx, d_u, d_v, d_w, d_wij, s_idx, s_m, s_rho,
             s_u, s_v, s_w, s_ug_star, s_vg_star, s_wg_star):
        omega = s_m[s_idx] / s_rho[s_idx]
        d_u[d_idx] += s_u[s_idx] * d_wij[d_idx] * omega
        d_v[d_idx] += s_v[s_idx] * d_wij[d_idx] * omega
        d_w[d_idx] += s_w[s_idx] * d_wij[d_idx] * omega
        if abs(s_u[s_idx]) < 1e-14 and (abs(s_v[s_idx]) < 1e-14):
            d_u[d_idx] += s_ug_star[s_idx] * d_wij[d_idx] * omega
            d_v[d_idx] += s_vg_star[s_idx] * d_wij[d_idx] * omega
            d_w[d_idx] += s_wg_star[s_idx] * d_wij[d_idx] * omega

    def post_loop(self, d_idx, d_vdotn, d_u, d_v, d_w, d_normal, d_tangent, d_vdott):
        d_vdotn[d_idx] = d_u[d_idx] * d_normal[3*d_idx] + d_v[d_idx] * d_normal[3*d_idx + 1] +\
             d_w[d_idx] * d_normal[3*d_idx + 2]
        d_vdott[d_idx] = d_u[d_idx] * d_tangent[3*d_idx] + d_v[d_idx] * d_tangent[3*d_idx + 1] +\
             d_w[d_idx] * d_tangent[3*d_idx + 2]


class ComputeSlipVelocity(Equation):
    def __init__(self, dest, sources):
        self.relax = RELAX
        super(ComputeSlipVelocity, self).__init__(dest, sources)

    def loop(self, d_idx, s_idx, s_m, s_rho, d_bid, d_vdotn, s_vdotn,
             d_vdott, s_vdott, WIJ, d_normal, d_tangent, d_ug_star, d_vg_star, d_h):
        bid = declare('int')
        omega = s_m[s_idx] / s_rho[s_idx]
        bid = d_bid[d_idx]
        if (s_idx == bid):
            den = WIJ * omega
            # solve the system
            if (abs(s_vdotn[s_idx] - d_vdotn[d_idx] * den) - abs(den) < 1e-14):
                delta = d_vdotn[d_idx] - (s_vdotn[s_idx] - d_vdotn[d_idx] * den) / (den)
                d_vdotn[d_idx] -= delta * d_h[d_idx] #self.relax
            if (abs(s_vdott[s_idx] - d_vdott[d_idx] * den) - abs(den) < 1e-14):
                delta = d_vdott[d_idx] - (s_vdott[s_idx] - d_vdott[d_idx] * den) / (den)
                d_vdott[d_idx] -= delta * self.relax

            d_ug_star[d_idx] = d_vdotn[d_idx] * d_normal[3*d_idx] + d_vdott[d_idx] * d_tangent[3*d_idx]
            d_vg_star[d_idx] = d_vdotn[d_idx] * d_normal[3*d_idx+1] + d_vdott[d_idx] * d_tangent[3*d_idx+1]
            # print(d_u[d_idx])


class FindGradientdotNormal(Equation):
    def __init__(self, dest, sources):
        self.converge = -1.0
        super(FindGradientdotNormal, self).__init__(dest, sources)

    def initialize(self, d_idx, d_graddotn):
        d_graddotn[d_idx] = 0.0

    def loop(self, d_idx, d_graddotn, s_p, DWIJ, d_normal, s_idx, s_m, s_rho):
        omega = s_m[s_idx] / s_rho[s_idx]
        d_graddotn[d_idx] += s_p[s_idx] * (DWIJ[0] * d_normal[3 * d_idx] +
                DWIJ[1] * d_normal[3 * d_idx + 1]) * omega

    def reduce(self, dst, t, dt):
        # sumgrad = sum(numpy.abs(dst.graddotn))/len(dst.graddotn)
        sumgrad = max(numpy.abs(dst.graddotn))
        h = dst.h[0]
        print(sumgrad, h, 'comp', self.converge, dst.name, t)
        if (sumgrad < h):
            print("converge")
            self.converge = 1.0
        else:
            self.converge = -1.0

    def converged(self):
        print(self.converge, 'in grad')
        return self.converge

class FindGradientdotNormalSolid(Equation):
    def __init__(self, dest, sources):
        self.converge = -1.0
        super(FindGradientdotNormalSolid, self).__init__(dest, sources)

    def initialize(self, d_idx, d_graddotn):
        d_graddotn[d_idx] = 0.0

    def loop(self, d_idx, d_graddotn, s_p, DWIJ, d_normal, s_idx, s_m, s_rho, d_wijn, s_bid):
        omega = s_m[s_idx] / s_rho[s_idx]
        d_graddotn[d_idx] += s_p[s_idx] * (DWIJ[0] * d_normal[3 * d_idx] +
                DWIJ[1] * d_normal[3 * d_idx + 1]) * omega

    def reduce(self, dst, t, dt):
        # sumgrad = sum(numpy.abs(dst.graddotn))/len(dst.graddotn)
        sumgrad = max(numpy.abs(dst.graddotn))
        h = dst.h[0]
        print(sumgrad, h, 'comp', self.converge, dst.name, t)
        if (sumgrad < h):
            print("converge")
            self.converge = 1.0
        else:
            self.converge = -1.0

    def converged(self):
        print(self.converge, 'in grad')
        return self.converge

class Computecoeff(Equation):
    def __init__(self, dest, sources):
        self.converge = -1.0
        self.relax = RELAX
        super(Computecoeff, self).__init__(dest, sources)

    def loop_all(self, d_idx, d_m, d_rho, d_bid, NBRS, N_NBRS, s_graddotn, d_p,
                 d_x, d_y, d_z, s_x, s_y, s_z, SPH_KERNEL, d_h, s_normal):
        bid, s_idx, i = declare('int', 3)
        dwij = declare('matrix(3)')
        omega = d_m[d_idx] / d_rho[d_idx]
        bid = d_bid[d_idx]
        nump, den = 0.0, 0.0

        if (bid > -1):
            for i in range(N_NBRS):
                s_idx = NBRS[i]
                xij = d_x[d_idx] - s_x[s_idx]
                yij = d_y[d_idx] - s_y[s_idx]
                zij = d_z[d_idx] - s_z[s_idx]
                rij = sqrt(xij**2 + yij**2 + zij**2)
                SPH_KERNEL.gradient([xij, yij, zij], rij, d_h[d_idx], dwij)
                # if (bid == s_idx):
                fac = -(dwij[0] * s_normal[3 * s_idx] + dwij[1] * s_normal[3 * s_idx + 1]) * omega
                nump += fac * s_graddotn[s_idx]
                den += fac**2

            # printf("%f, %f, %f\n", numu, den, numv)
            if (abs(den) > 1e-14):
                d_p[d_idx] -= nump/den * 0.1 #d_h[d_idx]


def solid_bc(bcs, fluids, rho0, p0):
    print(bcs)
    g0 = []
    g1 = []
    g2 = []
    for bc in bcs:
        if bc == 'u_no_slip':
            from solid_bc.marrone import LiuCorrection, LiuCorrectionPreStep, CopyNoSlipMirrorToGhost

            g3, g0_ = [], []
            g0.append(
                FindNearestSolidParticle(dest='solid0', sources=['boundary']))
            g0.append(
                SetValuesonMirror(dest='ghost_mirror', sources=['fluid', 'solid1']))
            g0.append(
                SummationDensity(dest='solid0', sources=['solid0', 'fluid', 'solid1']))
            g0.append(
                SummationDensity(dest='fluid', sources=['solid0', 'fluid', 'solid1']))
            g0.append(
                SummationDensity(dest='solid1', sources=['solid0', 'fluid', 'solid1']))


            g0_.append(
                LiuCorrectionPreStep(dest='boundary', sources=['fluid', 'solid1', 'solid0']))
            g0_.append(
                CopyNoSlipMirrorToGhost(dest='solid0', sources=['ghost_mirror'])
            )

            g1.append(
                LiuCorrection(dest='boundary', sources=['fluid', 'solid1', 'solid0']))
            # g1.append(
            #     FindGradientVeldotNormal(dest='boundary', sources=['fluid', 'solid1']))
            # g1.append(
            #     FindGradientVeldotNormalSolid(dest='boundary', sources=['solid0']))
            g1.append(
                FindVelocity(dest='boundary', sources=['fluid', 'solid1']))
            g1.append(
                FindVelocitySolid(dest='boundary', sources=['solid0']))
            # g2.append(
            #     ComputeNoSlipVelocity(dest='solid0', sources=['boundary']))
            g2.append(
                ComputeNoSlipVelocity2(dest='solid0', sources=['boundary']))
            return [g0, g0_, g1, g2]
        if bc == 'u_slip':
            g3 = []
            g0.append(
                FindNearestBoundaryParticle(dest='solid0', sources=['boundary']))
            g0.append(
                SummationDensity(dest='solid0', sources=['solid0', 'fluid', 'solid1']))
            g0.append(
                SummationDensity(dest='fluid', sources=['solid0', 'fluid', 'solid1']))
            g0.append(
                SummationDensity(dest='solid1', sources=['solid0', 'fluid', 'solid1']))
            g0.append(
                SummationDensity(dest='boundary', sources=['solid0', 'fluid', 'solid1']))
            g1.append(
                LiuCorrectionPreStep(dest='boundary', sources=['solid0', 'fluid', 'solid1']))
            g2.append(
                LiuCorrection(dest='boundary', sources=['solid0', 'fluid', 'solid1']))
            g2.append(
                FindVelocitydotNormal(dest='boundary', sources=['solid0', 'fluid', 'solid1']))
            g3.append(
                ComputeSlipVelocity(dest='solid0', sources=['boundary']))
            return [g0, g1, g2, g3]

        if bc == 'p_solid':
            from solid_bc.marrone import LiuCorrection, LiuCorrectionPreStep, CopyPressureMirrorToGhost

            g0_=[]
            g2 = []
            g0.append(
                FindNearestSolidParticle(dest='solid0', sources=['boundary']))
            g0.append(
                SummationDensity(dest='solid0', sources=['solid0', 'fluid', 'solid1']))
            g0.append(
                SetValuesonMirror(dest='ghost_mirror', sources=['fluid', 'solid1']))
            g0.append(
                SummationDensity(dest='fluid', sources=['solid0', 'fluid', 'solid1']))
            g0.append(
                SummationDensity(dest='solid1', sources=['solid0', 'fluid', 'solid1']))

            g0_.append(
                LiuCorrectionPreStep(dest='boundary', sources=['fluid', 'solid1', 'solid0']))
            g0_.append(
                CopyPressureMirrorToGhost(dest='solid0', sources=['ghost_mirror'])
            )

            g1.append(
                LiuCorrection(dest='boundary', sources=['fluid', 'solid1', 'solid0']))
            g1.append(
                FindGradientdotNormal(dest='boundary', sources=['fluid', 'solid1']))
            g1.append(
                FindGradientdotNormalSolid(dest='boundary', sources=['solid0']))
            # g1.append(
            #     LiuCorrection(dest='solid0', sources=['boundary']))
            g2.append(
                Computecoeff(dest='solid0', sources=['boundary']))
            return [g0, g0_, g1, g2]
    print(g0)
    return [g0, g1]