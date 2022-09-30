from pysph.sph.scheme import Scheme
from pysph.sph.equation import Equation, Group
from pysph.sph.wc.linalg import gj_solve, augmented_matrix
from compyle.api import declare

from math import sin, cos, sqrt, pi, atan2

M_PI = pi

def get_basis_poynomials(order=1, xij=0.0, yij=0.0, zij=0.0, dim=1, basis=[0.0, 0.0]):
    index, i, j, k = declare('int', 4)
    index = 0
    if dim == 1:
        for i in range(order+1):
            if (i <= order):
                basis[index] = (xij**i)
                index += 1
    elif dim == 2:
        for i in range(order+1):
            for j in range(order+1):
                if (i + j <= order):
                    basis[index] = (xij**i * yij**j)
                    index += 1
    elif dim == 3:
        for i in range(order+1):
            for j in range(order+1):
                for k in range(order+1):
                    if (i + j + k <= order):
                        basis[index] = (xij**i * yij**j * zij**k)
                        index += 1


class EvaluateReproducingCoefficients(Equation):
    def _get_helpers_(self):
        return [gj_solve, augmented_matrix, get_basis_poynomials]

    def __init__(self, dest, sources, dim=2, fac=1):
        self.dim = dim
        self.fac = fac
        self.order = 2
        self.np = 6

        super(EvaluateReproducingCoefficients, self).__init__(dest, sources)

    def initialize(self, d_coeff, d_idx):
        idx10, i = declare('int', 2)
        idx10 = 10 * d_idx
        for i in range(10):
            d_coeff[idx10+i] = 0.0

    def loop_all(
        self, d_idx, SPH_KERNEL, NBRS, N_NBRS, s_x, s_y, s_z,
        d_h, s_h, d_x, d_y, d_z, d_coeff, s_V):
        i, j, k, s_idx, np, order, idx10 = declare('int', 7)
        aug_mat = declare('matrix(110)')
        b = declare('matrix(10)')
        mom_mat = declare('matrix(100)')
        basis = declare('matrix(10)')
        res = declare('matrix(10)')

        idx10 = 10 * d_idx
        np = self.np
        order = self.order

        for i in range(100):
            mom_mat[i] = 0.0
        for i in range(110):
            aug_mat[i] = 0.0
        for i in range(10):
            b[i] = 0.0
            res[i] = 0.0
            basis[i] = 0.0
        b[0] = 1.0

        for i in range(N_NBRS):
            s_idx = NBRS[i]
            XIJ = d_x[d_idx] - s_x[s_idx]
            YIJ = d_y[d_idx] - s_y[s_idx]
            ZIJ = d_z[d_idx] - s_z[s_idx]
            RIJ = sqrt(XIJ**2 + YIJ**2 + ZIJ**2)
            HIJ = self.fac * 0.5 * (d_h[d_idx] + s_h[s_idx])
            get_basis_poynomials(order, XIJ, YIJ, ZIJ, self.dim, basis)
            WIJ = SPH_KERNEL.kernel(xij=[0.0, 0.0, 0.0], rij=RIJ, h=HIJ)

            for j in range(np):
                for k in range(np):
                    mom_mat[np*j+k] += basis[j] * basis[k] * WIJ * s_V[s_idx]

        augmented_matrix(mom_mat, b, n=np, na=1, nmax=np, result=aug_mat)
        gj_solve(aug_mat, n=np, nb=1, result=res)
        for i in range(10):
            d_coeff[idx10+i] = res[i]


class Volume(Equation):
    def initialize(self, d_idx, d_V):
        d_V[d_idx] = 0.0

    def loop_all(self, d_idx, d_V, d_x, d_y, d_z, s_x, s_y, s_z, d_h, s_h, NBRS, N_NBRS, SPH_KERNEL):
        i, s_idx = declare('int', 2)
        v1 =0.0
        for i in range(N_NBRS):
            s_idx = NBRS[i]
            XIJ = d_x[d_idx] - s_x[s_idx]
            YIJ = d_y[d_idx] - s_y[s_idx]
            ZIJ = d_z[d_idx] - s_z[s_idx]
            RIJ = sqrt(XIJ**2 + YIJ**2 + ZIJ**2)
            HIJ = 0.5 * (d_h[d_idx] + s_h[s_idx])
            WIJ = SPH_KERNEL.kernel(xij=[0.0, 0.0, 0.0], rij=RIJ, h=HIJ)
            v1 += WIJ

        if (v1 > 1e-14):
            d_V[d_idx] = 1/v1

class VelocityApproxRKPM(Equation):
    def _get_helpers_(self):
        return [get_basis_poynomials]

    def __init__(self, dest, sources, dim=2):
        self.dim = dim
        self.order = 2
        self.np = 6

        super(VelocityApproxRKPM, self).__init__(dest, sources)

    def initialize(self, d_idx, d_u, d_v, d_w):
        d_u[d_idx] = 0.0
        d_v[d_idx] = 0.0
        d_w[d_idx] = 0.0

    def loop_all(self, d_idx, d_u, d_v, d_w, s_u, s_v, s_w, SPH_KERNEL, N_NBRS, NBRS,
                 d_coeff, d_x, d_y, d_z, s_x, s_y, s_z, d_h, s_h, s_V):
        i, np, order, idx10, j, s_idx = declare('int', 6)
        basis = declare('matrix(10)')
        u_basis = declare('matrix(10)')
        v_basis = declare('matrix(10)')
        w_basis = declare('matrix(10)')
        for i in range(10):
            basis[i] = 0.0
            u_basis[i] = 0.0
            v_basis[i] = 0.0
            w_basis[i] = 0.0

        idx10 = 10 * d_idx
        order = self.order
        np = self.np
        for i in range(N_NBRS):
            s_idx = NBRS[i]
            XIJ = d_x[d_idx] - s_x[s_idx]
            YIJ = d_y[d_idx] - s_y[s_idx]
            ZIJ = d_z[d_idx] - s_z[s_idx]
            RIJ = sqrt(XIJ**2 + YIJ**2 + ZIJ**2)
            HIJ = 0.5 * (d_h[d_idx] + s_h[s_idx])
            get_basis_poynomials(order, XIJ, YIJ, ZIJ, self.dim, basis)
            WIJ = SPH_KERNEL.kernel(xij=[0.0, 0.0, 0.0], rij=RIJ, h=HIJ)

            tmpu = WIJ * s_u[s_idx] * s_V[s_idx]
            tmpv = WIJ * s_v[s_idx] * s_V[s_idx]
            tmpw = WIJ * s_w[s_idx] * s_V[s_idx]
            for j in range(np):
                u_basis[j] += basis[j] * tmpu
                v_basis[j] += basis[j] * tmpv
                w_basis[j] += basis[j] * tmpw

        for i in range(np):
            d_u[d_idx] += d_coeff[idx10+i] * u_basis[i]
            d_v[d_idx] += d_coeff[idx10+i] * v_basis[i]
            d_w[d_idx] += d_coeff[idx10+i] * w_basis[i]


class PressureApproxRKPM(Equation):
    def _get_helpers_(self):
        return [get_basis_poynomials]

    def __init__(self, dest, sources, dim=2):
        self.dim = dim
        self.order = 2
        self.np = 6

        super(PressureApproxRKPM, self).__init__(dest, sources)

    def initialize(self, d_idx, d_p):
        d_p[d_idx] = 0.0

    def loop_all(self, d_idx, d_p, s_p, SPH_KERNEL, N_NBRS, NBRS,
                 d_coeff, d_x, d_y, d_z, s_x, s_y, s_z, d_h, s_h, s_V):
        i, np, order, idx10, j, s_idx = declare('int', 6)
        basis = declare('matrix(10)')
        p_basis = declare('matrix(10)')
        for i in range(10):
            basis[i] = 0.0
            p_basis[i] = 0.0

        idx10 = 10 * d_idx
        order = self.order
        np = self.np
        for i in range(N_NBRS):
            s_idx = NBRS[i]
            XIJ = d_x[d_idx] - s_x[s_idx]
            YIJ = d_y[d_idx] - s_y[s_idx]
            ZIJ = d_z[d_idx] - s_z[s_idx]
            RIJ = sqrt(XIJ**2 + YIJ**2 + ZIJ**2)
            HIJ = 0.5 * (d_h[d_idx] + s_h[s_idx])
            get_basis_poynomials(order, XIJ, YIJ, ZIJ, self.dim, basis)
            WIJ = SPH_KERNEL.kernel(xij=[0.0, 0.0, 0.0], rij=RIJ, h=HIJ)

            tmp = WIJ * s_p[s_idx] * s_V[s_idx]
            for j in range(np):
                p_basis[j] += basis[j] * tmp

        for i in range(np):
            d_p[d_idx] += d_coeff[idx10+i] * p_basis[i]

def get_props():
    return ['V', {'name':'coeff', 'stride':10}]


def get_equations(dest, sources, derv=0, prop='p'):
    eqns = [
        Group(equations=[Volume(dest=d, sources=sources) for d in sources], real=True),
        Group(equations=[
            EvaluateReproducingCoefficients(
                dest=dest, sources=sources, dim=2)
        ], real=True),
    ]
    if prop == 'p':
        grp = Group(
            equations=[PressureApproxRKPM(dest=dest, sources=sources, dim=2)])
        eqns.append(grp)
    elif prop == 'u':
        grp = Group(
            equations=[VelocityApproxRKPM(dest=dest, sources=sources, dim=2)])
        eqns.append(grp)

    return eqns

def get_evaluator(dest, sources):
    from pysph.tools.sph_evaluator import SPHEvaluator

    sources_n = [s.name for s in sources]
    eqns = get_equations(dest.name, sources_n)
    print(eqns)
    eval = SPHEvaluator([dest]+sources, eqns, dim=2)
    eval.evaluate()