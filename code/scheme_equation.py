from compyle.api import declare
from pysph.sph.equation import Group, Equation
from pysph.sph.wc.linalg import (
    augmented_matrix, dot, gj_solve, identity, mat_vec_mult
)
from pysph.base.kernels import QuinticSpline

from math import pi
c0 = 10

# equations

def transpose(a=[1.0, 0.0], n=3, result=[0.0, 0.0]):
    i, j = declare('int', 2)
    for i in range(n):
        for j in range(n):
            result[n*i + j] = a[n*j + i]


def ten3_vec_mul(a=[1.0, 0.0], b=[1.0, 0.0], n=3, result=[0.0, 0.0]):
    i, j, k = declare('int', 3)
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += a[n*(n*i + j) + k] * b[k]
            result[n*i + j] = s


def qs_dwdq(rij=1.0, h=1.0):
    h1 = 1. / h
    q = rij * h1

    # get the kernel normalizing factor
    fac = 7.0 * h1 * h1 / (pi * 478.)

    tmp3 = 3. - q
    tmp2 = 2. - q
    tmp1 = 1. - q

    # compute the gradient
    if (rij > 1e-12):
        if (q > 3.0):
            val = 0.0

        elif (q > 2.0):
            val = -5.0 * tmp3 * tmp3 * tmp3 * tmp3

        elif (q > 1.0):
            val = -5.0 * tmp3 * tmp3 * tmp3 * tmp3
            val += 30.0 * tmp2 * tmp2 * tmp2 * tmp2
        else:
            val = -5.0 * tmp3 * tmp3 * tmp3 * tmp3
            val += 30.0 * tmp2 * tmp2 * tmp2 * tmp2
            val -= 75.0 * tmp1 * tmp1 * tmp1 * tmp1
    else:
        val = 0.0

    return val * fac

def qs_dwdq2(xij=[0.0, 0.0, 0.0], rij=1.0, h=1.0, d2wij=[0.0, 0.0, 0.0]):
    h1 = 1. / h
    q = rij * h1

    # get the kernel normalizing factor
    fac = 7.0 * h1 * h1 / (pi * 478.)

    tmp3 = 3. - q
    tmp2 = 2. - q
    tmp1 = 1. - q

    # compute the second gradient
    if (rij > 1e-12):
        if (q > 3.0):
            val = 0.0

        elif (q > 2.0):
            val = 20.0 * tmp3 * tmp3 * tmp3

        elif (q > 1.0):
            val = 20.0 * tmp3 * tmp3 * tmp3
            val -= 120.0 * tmp2 * tmp2 * tmp2
        else:
            val = 20.0 * tmp3 * tmp3 * tmp3
            val -= 120.0 * tmp2 * tmp2 * tmp2
            val += 300.0 * tmp1 * tmp1 * tmp1
    else:
        val = 0.0

    dwdq = qs_dwdq(rij, h)
    dw2dq2 = fac * val
    fac2 = 1.0 / ( rij**2 * h**2)
    if rij > 1e-14:
        t1 = fac2 * (dw2dq2 - dwdq / q) 
        d2wij[0] = t1 * xij[0] * xij[0] + dwdq/ (h**2 * q)
        d2wij[1] = t1 * xij[0] * xij[1]
        d2wij[2] = t1 * xij[1] * xij[1] + dwdq/ (h**2 * q)
    else:
        d2wij[0] = dw2dq2 / h**2 
        d2wij[1] = dw2dq2 / h**2 
        d2wij[2] = dw2dq2 / h**2 

class SetRhoc(Equation):
    def initialize(self, d_idx, d_rhoc, d_rho):
        d_rhoc[d_idx] = d_rho[d_idx]

class CopyLToGhost(Equation):
    def initialize(self, d_idx, d_tag, d_gid, d_L):
        idx, idx16, i = declare('int', 3)
        if d_tag[d_idx] == 2:
            idx = d_gid[d_idx] * 16
            idx16 = d_idx * 16
            for i in range(16):
                d_L[idx16 + i] = d_L[idx + i]


class DoubleDerivativeApprox(Equation):
    '''
    [1]S. P. Korzilius, W. H. A. Schilders, and M. J. H. Anthonissen, “An
    Improved CSPM Approach for Accurate Second-Derivative Approximations with
    SPH,” JAMP, vol. 05, no. 01, pp. 168–184, 2017, doi:
    10.4236/jamp.2017.51017.  
    '''

    def initialize(self, d_idx, d_ddu):
        i = declare('int')
        for i in range(3):
            d_ddu[3 * d_idx + i] = 0.0

    def _get_helpers_(self):
        return [qs_dwdq2, qs_dwdq]

    def loop(self, d_idx, d_ddu, XIJ, HIJ, RIJ, s_idx, s_m, s_rho, s_u, d_u, d_L):
        i = declare('int')
        L = declare('matrix(9)')
        v = declare('matrix(3)')
        d2wij = declare('matrix(3)')

        for i in range(3):
            v[i] = 0.0

        qs_dwdq2(XIJ, RIJ, HIJ, v)

        for i in range(3):
            d2wij[i] = 0.0
        for i in range(9):
            L[i] = d_L[9 * d_idx + i]

        mat_vec_mult(L, v, 3, d2wij)

        vj = s_m[s_idx] / s_rho[s_idx]
        tmp2 = (s_u[s_idx] - d_u[d_idx]) * vj

        d_ddu[3 * d_idx] += tmp2 * d2wij[0]
        d_ddu[3 * d_idx + 1] += tmp2 * d2wij[1]
        d_ddu[3 * d_idx + 2] += tmp2 * d2wij[2]


class DoubleDervMoment(Equation):
    '''
    [1]S. P. Korzilius, W. H. A. Schilders, and M. J. H. Anthonissen, “An
    Improved CSPM Approach for Accurate Second-Derivative Approximations with
    SPH,” JAMP, vol. 05, no. 01, pp. 168–184, 2017, doi:
    10.4236/jamp.2017.51017.  
    '''
    def initialize(self, d_idx, d_ddu):
        i = declare('int')
        for i in range(3):
            d_ddu[3 * d_idx + i] = 0.0

    def _get_helpers_(self):
        return [qs_dwdq2, qs_dwdq]

    def loop(self, d_idx, d_ddu, XIJ, HIJ, RIJ, s_idx, s_m, s_rho, d_gradv, d_L):
        i, idx9 = declare('int', 2)
        L = declare('matrix(9)')
        v = declare('matrix(3)')
        d2wij = declare('matrix(3)')
        idx9 = d_idx * 9

        qs_dwdq2(XIJ, RIJ, HIJ, v)

        for i in range(3):
            d2wij[i] = 0.0
        for i in range(9):
            L[i] = d_L[9 * d_idx + i]

        mat_vec_mult(L, v, 3, d2wij)

        vj = s_m[s_idx] / s_rho[s_idx]

        # negative sign is absorbed into XIJ from XJI
        xijdotgradv0 = (d_gradv[idx9] * XIJ[0] + 
                        d_gradv[idx9 + 1] * XIJ[1] +
                        d_gradv[idx9 + 2] * XIJ[2])

        d_ddu[3 * d_idx] += xijdotgradv0 * vj * d2wij[0]
        d_ddu[3 * d_idx + 1] += xijdotgradv0 * vj * d2wij[1]
        d_ddu[3 * d_idx + 2] += xijdotgradv0 * vj * d2wij[2]

class KorziliusPreStep(Equation):
    def _get_helpers_(self):
        return [transpose, mat_vec_mult, ten3_vec_mul, augmented_matrix, gj_solve, identity]

    def initialize(self, d_idx, d_L):
        i = declare('int')
        for i in range(9):
            d_L[9 * d_idx + i] = 0.0

    def post_loop(self, d_idx, d_L, d_gamma, d_gamma2):
        i, idx9 = declare('int', 2)

        temp_aug_L = declare('matrix(18)')
        L = declare('matrix(9)')
        L_inv = declare('matrix(9)')

        idx9 = d_idx * 9

        for i in range(18):
            temp_aug_L[i] = 0.0
        for i in range(9):
            L[i] = -0.5 * (1 - d_gamma2[d_idx]) * d_gamma[idx9 + i]
            L_inv[i] = 0.0
        
        identity(L_inv, 3)
        augmented_matrix(L, L_inv, 3, 3, 3, temp_aug_L)

        # If is_singular > 0 then matrix was singular
        is_singular = gj_solve(temp_aug_L, 3, 3, L_inv)

        for i in range(9):
            d_L[d_idx*9 + i] = L_inv[i]


class Gamma(Equation):
    '''
    [1]S. P. Korzilius, W. H. A. Schilders, and M. J. H. Anthonissen, “An
    Improved CSPM Approach for Accurate Second-Derivative Approximations with
    SPH,” JAMP, vol. 05, no. 01, pp. 168–184, 2017, doi:
    10.4236/jamp.2017.51017.  
    '''
    def initialize(self, d_idx, d_gamma):
        i = declare('int')
        for i in range(9):
            d_gamma[9*d_idx + i] = 0.0

    def _get_helpers_(self):
        return [qs_dwdq2, qs_dwdq]

    def loop(self, d_idx, d_gamma, XIJ, HIJ, RIJ, s_idx, s_m, s_rho):
        idx9 = declare('int')
        v = declare('matrix(3)')

        for i in range(3):
            v[i] = 0.0

        idx9 = d_idx * 9

        qs_dwdq2(XIJ, RIJ, HIJ, v)

        vj = s_m[s_idx] / s_rho[s_idx]

        d_gamma[idx9] +=  v[0] * XIJ[0] * XIJ[0] * vj
        d_gamma[idx9 + 1] += 2 * v[0] * XIJ[1] * XIJ[0] * vj
        d_gamma[idx9 + 2] += v[0] * XIJ[1] * XIJ[1] * vj

        d_gamma[idx9 + 3] += v[1] * XIJ[0] * XIJ[0] * vj
        d_gamma[idx9 + 4] += 2 * v[1] * XIJ[1] * XIJ[0] * vj
        d_gamma[idx9 + 5] += v[1] * XIJ[1] * XIJ[1] * vj

        d_gamma[idx9 + 6] += v[2] * XIJ[0] * XIJ[0] * vj
        d_gamma[idx9 + 7] += 2 * v[2] * XIJ[1] * XIJ[0] * vj
        d_gamma[idx9 + 8] += v[2] * XIJ[1] * XIJ[1] * vj


class Gamma22(Equation):
    def initialize(self, d_gamma2, d_idx):
        d_gamma2[d_idx] = 0.0

    def _get_helpers_(self):
        return [qs_dwdq2]

    def loop(self, d_idx, d_gamma2, XIJ, HIJ, RIJ, s_idx, s_m, s_rho, DWIJ, d_bt):
        i, idx9 = declare('int', 2)
        bt = declare('matrix(9)')
        v1 = declare('matrix(3)')
        v2 = declare('matrix(3)')
        idx9 = d_idx * 9

        vj = s_m[s_idx] / s_rho[s_idx]

        for i in range(9):
            bt[i] = d_bt[idx9 + i]
        for i in range(3):
            v1[i] = DWIJ[i]
            v2[i] = 0.0
        mat_vec_mult(bt, v1, 3, v2)

        d_gamma2[d_idx] += -(XIJ[0] * v2[0] + XIJ[1] *
                                  v2[1] + XIJ[2] * v2[2]) * vj



class FatehiViscosity(Equation):
    def __init__(self, dest, sources, nu, rho0, dim=2):
        r"""
        Parameters
        ----------
        nu : float
            kinematic viscosity
        """
        self.dim = dim
        self.nu = nu
        self.rho0 = rho0
        super(FatehiViscosity, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, s_rho, d_m, d_V, s_V,
             d_au, d_av, d_aw, s_m,
             R2IJ, EPS, DWIJ, VIJ, XIJ, RIJ, d_gradv, d_rhoc, s_rhoc):

        idx9 = declare('int')
        idx9 = 9 * d_idx
        # # averaged shear viscosity Eq. (6)
        etai = self.nu * d_rhoc[d_idx]
        etaj = self.nu * s_rhoc[s_idx]

        etaij = 2 * (etai * etaj)/(etai + etaj)

        if RIJ > 1e-14:
            xijdotdwij = XIJ[0]*DWIJ[0] + XIJ[1]*DWIJ[1] + XIJ[2]*DWIJ[2]

            xijdotgradv0 = d_gradv[idx9]*XIJ[0] + d_gradv[idx9+1]*XIJ[1] +  d_gradv[idx9+2]*XIJ[2]
            xijdotgradv1 = d_gradv[idx9+3]*XIJ[0] + d_gradv[idx9+4]*XIJ[1] +  d_gradv[idx9+5]*XIJ[2]
            xijdotgradv2 = d_gradv[idx9+6]*XIJ[0] + d_gradv[idx9+7]*XIJ[1] +  d_gradv[idx9+8]*XIJ[2]
            fac = s_m[s_idx] / (s_rho[s_idx] * d_rhoc[d_idx])
            tmp = fac * (2 * etaij) * (xijdotdwij/R2IJ)

            d_au[d_idx] += tmp * (VIJ[0] - xijdotgradv0)
            d_av[d_idx] += tmp * (VIJ[1] - xijdotgradv1)
            d_aw[d_idx] += tmp * (VIJ[2] - xijdotgradv2)


class FatehiViscosityCorrected(FatehiViscosity):
    def _get_helpers_(self):
        return [mat_vec_mult]

    def loop(self, d_idx, s_idx, d_rho, s_rho, d_m,
             d_au, d_av, d_aw, s_m,
             R2IJ, EPS, DWIJ, VIJ, XIJ, RIJ, d_gradv, d_L, d_rhoc, s_rhoc):
        i, d = declare('int', 2)
        L = declare('matrix(9)')
        v = declare('matrix(3)')
        dwij = declare('matrix(3)')
        d = self.dim


        for i in range(3):
            v[i] = DWIJ[i]
            dwij[i] = 0.0
        for i in range(9):
            L[i] = d_L[9 * d_idx + i]

        mat_vec_mult(L, v, 3, dwij)

        idx9 = declare('int')
        idx9 = 9 * d_idx
        # # averaged shear viscosity Eq. (6)
        etai = self.nu

        if RIJ > 1e-14:
            xijdotdwij = XIJ[0]*dwij[0] + XIJ[1]*dwij[1] + XIJ[2]*dwij[2]

            xijdotgradv0 = d_gradv[idx9]*XIJ[0] + d_gradv[idx9+1]*XIJ[1] +  d_gradv[idx9+2]*XIJ[2]
            xijdotgradv1 = d_gradv[idx9+3]*XIJ[0] + d_gradv[idx9+4]*XIJ[1] +  d_gradv[idx9+5]*XIJ[2]
            xijdotgradv2 = d_gradv[idx9+6]*XIJ[0] + d_gradv[idx9+7]*XIJ[1] +  d_gradv[idx9+8]*XIJ[2]
            fac = s_m[s_idx] / (s_rho[s_idx])
            tmp = fac * (2 * etai) * (xijdotdwij/R2IJ)

            d_au[d_idx] += tmp * (VIJ[0] - xijdotgradv0)
            d_av[d_idx] += tmp * (VIJ[1] - xijdotgradv1)
            d_aw[d_idx] += tmp * (VIJ[2] - xijdotgradv2)


class CreateFatehiOrder3Tensor(Equation):
    def initialize(self, d_idx, d_L3):
        i, idx27 = declare('int', 2)
        idx27 = d_idx * 27
        for i in range(27):
            d_L3[idx27 + i] = 0.0

    def loop(self, d_idx, XIJ, DWIJ, d_L3, s_m, s_rho, s_idx, R2IJ):
        i, j, k, idx27 = declare('int', 4)

        idx27 = d_idx * 27
        if R2IJ > 1e-14:
            v_j = s_m[s_idx] / s_rho[s_idx]/R2IJ

            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        d_L3[idx27 + 3*(3*i + j) + k] += v_j * XIJ[i] * DWIJ[j] * XIJ[k]


class CreateFatehiOrder2Tensor(Equation):
    def initialize(self, d_idx, d_L2):
        i, idx9 = declare('int', 2)
        idx9 = d_idx * 9
        for i in range(9):
            d_L2[idx9 + i] = 0.0

    def loop(self, d_idx, XIJ, DWIJ, d_L2, s_m, s_rho, s_idx, R2IJ):
        i, j, idx9 = declare('int', 3)
        idx9 = d_idx * 9

        v_j = s_m[s_idx] / s_rho[s_idx]
        for i in range(3):
            for j in range(3):
                d_L2[idx9 + 3*i + j] += v_j * XIJ[i] * DWIJ[j]


class CreateFatehiVector(Equation):
    def initialize(self, d_idx, d_L1):
        i, idx3 = declare('int', 2)
        idx3 = d_idx * 3
        for i in range(3):
            d_L1[idx3 + i] = 0.0

    def loop(self, d_idx, XIJ, DWIJ, R2IJ, s_idx, s_m, s_rho, d_L1):
        i, idx3 = declare('int', 2)
        idx3 = d_idx * 3

        v_j = s_m[s_idx] / s_rho[s_idx] * R2IJ
        for i in range(3):
            d_L1[idx3 + i] +=  v_j * DWIJ[i]


class GradientCorrectionPreStepFatehi(Equation):
    def __init__(self, dest, sources, dim=2):
        self.dim = dim
        super(GradientCorrectionPreStepFatehi, self).__init__(dest, sources)

    def _get_helpers_(self):
        return [augmented_matrix, gj_solve, identity, dot, mat_vec_mult]

    def initialize(self, d_idx, d_bt):
        i = declare('int')
        for i in range(9):
            d_bt[9 * d_idx + i] = 0.0

    def loop_all(self, d_idx, d_bt, s_m, s_rho, d_x, d_y, d_z, d_h, s_x,
                 s_y, s_z, s_h, SPH_KERNEL, NBRS, N_NBRS):
        x = d_x[d_idx]
        y = d_y[d_idx]
        z = d_z[d_idx]
        h = d_h[d_idx]
        i, j, s_idx, n, d = declare('int', 5)
        d = self.dim
        xij = declare('matrix(3)')
        dwij = declare('matrix(3)')
        temp_aug_m2 = declare('matrix(18)')
        m2inv = declare('matrix(9)')
        m2 = declare('matrix(9)')

        for i in range(9):
            m2[i] = 0.0
            m2inv[i] = 0.0
        for i in range(18):
            temp_aug_m2[i] = 0.0
        for i in range(3):
            xij[i] = 0.0
            dwij[i] = 0.0

        n = self.dim
        for k in range(N_NBRS):
            s_idx = NBRS[k]
            xij[0] = x - s_x[s_idx]
            xij[1] = y - s_y[s_idx]
            xij[2] = z - s_z[s_idx]
            hij = (h + s_h[s_idx]) * 0.5
            r = sqrt(xij[0] * xij[0] + xij[1] * xij[1] + xij[2] * xij[2])
            SPH_KERNEL.gradient(xij, r, hij, dwij)
            V = s_m[s_idx] / s_rho[s_idx]
            if r > 1.0e-12:
                for i in range(n):
                    for j in range(n):
                        xj = xij[i]
                        m2[3 * i + j] -= V * dwij[j] * xj

        identity(m2inv, d)
        augmented_matrix(m2, m2inv, d, d, 3, temp_aug_m2)

        # If is_singular > 0 then matrix was singular
        is_singular = gj_solve(temp_aug_m2, d, d, m2inv)

        for i in range(d):
            for j in range(d):
                d_bt[d_idx*9 + 3*i + j] = m2inv[d*i + j]


class EvaluateFatehiCorrection(Equation):
    def __init__(self, dest, sources, dim=2):
        self.dim = dim
        super(EvaluateFatehiCorrection, self).__init__(dest, sources)

    def _get_helpers_(self):
        return [transpose, mat_vec_mult, ten3_vec_mul, augmented_matrix, gj_solve, identity]

    def post_loop(self, d_idx, d_L, d_L1, d_L2, d_L3, d_bt):
        i, j, idx9, d = declare('int', 4)
        bt_T = declare('matrix(9)')
        bt = declare('matrix(9)')
        v1 = declare('matrix(3)')
        v2 = declare('matrix(3)')

        temp_aug_L = declare('matrix(18)')
        ten2 = declare('matrix(9)')
        L = declare('matrix(9)')
        L_inv = declare('matrix(9)')
        ten27 = declare('matrix(27)')

        d = self.dim
        idx9 = d_idx * 9

        for i in range(18):
            temp_aug_L[i] = 0.0
        for i in range(27):
            ten27[i] = d_L3[d_idx*27 + i]
        for i in range(9):
            bt[i] = d_bt[idx9 + i]
            bt_T[i] = 0.0
            ten2[i] = 0.0
            L[i] = 0.0
            L_inv[i] = 0.0
        for i in range(3):
            v1[i] = d_L1[d_idx*3 + i]
            v2[i] = 0.0
        mat_vec_mult(bt, v1, 3, v2)

        ten3_vec_mul(ten27, v2, 3, ten2)


        for i in range(9):
            L[i] = -(d_L2[d_idx*9 + i] + ten2[i])

        identity(L_inv, d)
        augmented_matrix(L, L_inv, d, d, 3, temp_aug_L)

        # If is_singular > 0 then matrix was singular
        is_singular = gj_solve(temp_aug_L, d, d, L_inv)

        for i in range(d):
            for j in range(d):
                d_L[d_idx*9 + 3*i + j] = L_inv[d*i + j]


class LiuCorrectionPreStep(Equation):
    # Liu et al 2005
    def __init__(self, dest, sources, dim=2):
        self.dim = dim

        super(LiuCorrectionPreStep, self).__init__(dest, sources)

    def _get_helpers_(self):
        return [gj_solve, augmented_matrix, identity]

    def initialize(self, d_idx, d_L):
        i, j = declare('int', 2)

        for i in range(4):
            for j in range(4):
                d_L[16*d_idx + j+4*i] = 0.0

    def loop(self, d_idx, s_idx, d_h, s_h, s_x, s_y, s_z, d_x, d_y, d_z, s_rho,
             s_m, WIJ, XIJ, DWIJ, d_L):
        Vj = s_m[s_idx] / s_rho[s_idx]
        i16 = declare('int')
        i16 = 16*d_idx

        d_L[i16+0] += WIJ * Vj

        d_L[i16+1] += -XIJ[0] * WIJ * Vj
        d_L[i16+2] += -XIJ[1] * WIJ * Vj
        d_L[i16+3] += -XIJ[2] * WIJ * Vj

        d_L[i16+4] += DWIJ[0] * Vj
        d_L[i16+8] += DWIJ[1] * Vj
        d_L[i16+12] += DWIJ[2] * Vj

        d_L[i16+5] += -XIJ[0] * DWIJ[0] * Vj
        d_L[i16+6] += -XIJ[1] * DWIJ[0] * Vj
        d_L[i16+7] += -XIJ[2] * DWIJ[0] * Vj

        d_L[i16+9] += - XIJ[0] * DWIJ[1] * Vj
        d_L[i16+10] += -XIJ[1] * DWIJ[1] * Vj
        d_L[i16+11] += -XIJ[2] * DWIJ[1] * Vj

        d_L[i16+13] += -XIJ[0] * DWIJ[2] * Vj
        d_L[i16+14] += -XIJ[1] * DWIJ[2] * Vj
        d_L[i16+15] += -XIJ[2] * DWIJ[2] * Vj

    def post_loop(self, d_idx, d_L):
        i, j, n = declare('int', 3)
        n = self.dim + 1
        # Note that we allocate enough for a 3D case but may only use a
        # part of the matrix.
        l, linv = declare('matrix(16)', 2)
        tempaug = declare('matrix(32)')

        for i in range(32):
            tempaug[i] = 0.0
        for i in range(16):
            l[i] = 0.0
            linv[i] = 0.0

        for i in range(i):
            l[i] = d_L[16 * d_idx + i]

        identity(linv, n)
        augmented_matrix(l, linv, n, n, 4, tempaug)
        # print(tempaug)
        error_code = gj_solve(tempaug, n, n, linv)

        # print(linv, l)
        for i in range(n):
            for j in range(n):
                d_L[d_idx*16 + n*i + j] = linv[n*i + j]


class ViscosityCleary(Equation):
    def __init__(self, dest, sources, nu, rho0):
        r"""
        Parameters
        ----------
        nu : float
            kinematic viscosity
        """

        self.nu = nu
        self.rho0 = rho0
        super(ViscosityCleary, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, s_rho, d_m,
             d_au, d_av, d_aw, s_m,
             R2IJ, EPS, DWIJ, VIJ, XIJ, RIJ, s_rhoc, d_rhoc):

        # # averaged shear viscosity Eq. (6)
        etai = self.nu * d_rho[d_idx]
        etaj = self.nu * s_rho[s_idx]

        etaij = 2 * (etai * etaj)/(etai + etaj)

        if RIJ > 1e-14:
            xijdotdwij = XIJ[0]*DWIJ[0] + XIJ[1]*DWIJ[1] + XIJ[2]*DWIJ[2]
            tmp = s_m[s_idx] / (s_rho[s_idx] * d_rho[d_idx])
            tmp = tmp * (2 * etaij) * (xijdotdwij/R2IJ)

            d_au[d_idx] += tmp * VIJ[0]
            d_av[d_idx] += tmp * VIJ[1]
            d_aw[d_idx] += tmp * VIJ[2]

class CorrectedDensity(Equation):
    def initialize(self, d_idx, d_rho, d_wij):
        d_wij[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_wij, WIJ, s_m, s_rho):
        d_wij[d_idx] += WIJ * s_m[s_idx] / s_rho[s_idx]

    def post_loop(self, d_idx, d_wij, d_rho):
        if d_wij[d_idx] > 1e-14:
            d_rho[d_idx] /= d_wij[d_idx]

class MomentumEquationSymm(Equation):
    # violeu p 5.130
    def __init__(self, dest, sources,
                 gx=0.0, gy=0.0, gz=0.0):

        self.gx = gx
        self.gy = gy
        self.gz = gz
        super(MomentumEquationSymm, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw, d_dt_cfl):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, d_cs,
             d_p, d_au, d_av, d_aw, s_m,
             s_rho, s_cs, s_p, VIJ,
             XIJ, HIJ, R2IJ, RHOIJ1, EPS,
             DWIJ, WIJ, WDP, d_dt_cfl, d_m, d_rhoc):

        tmp = (s_p[s_idx] + d_p[d_idx])/(d_rho[d_idx] * s_rho[s_idx])

        d_au[d_idx] += -s_m[s_idx] * (tmp) * DWIJ[0]
        d_av[d_idx] += -s_m[s_idx] * (tmp) * DWIJ[1]
        d_aw[d_idx] += -s_m[s_idx] * (tmp) * DWIJ[2]

    def post_loop(self, d_idx, d_au, d_av, d_aw, d_dt_force):
        d_au[d_idx] += self.gx
        d_av[d_idx] += self.gy
        d_aw[d_idx] += self.gz


class MomentumEquationSymmMomConv(Equation):
    # violeu p 5.130
    def __init__(self, dest, sources,
                 gx=0.0, gy=0.0, gz=0.0, dim=2):

        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.dim = dim
        super(MomentumEquationSymmMomConv, self).__init__(dest, sources)

    def _get_helpers_(self):
        return [mat_vec_mult]

    def initialize(self, d_idx, d_au, d_av, d_aw, d_dt_cfl):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, d_cs, d_p, d_au, d_av, d_aw, s_m,
             s_rho, s_cs, s_p, VIJ, XIJ, HIJ, R2IJ, RHOIJ1, EPS, DWIJ, WIJ,
             WDP, d_dt_cfl, d_m, d_rhoc, s_L, d_L):

        i, d = declare('int', 2)
        L = declare('matrix(16)')
        v, wdwij = declare('matrix(4)', 2)
        dwij1, dwij2 = declare('matrix(3)', 2)
        d = self.dim

        v[0] = WIJ
        wdwij[0] = 0.0
        for i in range(3):
            v[i+1] = DWIJ[i]
            wdwij[i+1] = 0.0
        for i in range(16):
            L[i] = d_L[16 * d_idx + i]

        mat_vec_mult(L, v, d+1, wdwij)
        for i in range(3):
            dwij1[i] = wdwij[i+1]

        wdwij[0] = 0.0
        for i in range(3):
            v[i+1] = -DWIJ[i]
            wdwij[i+1] = 0.0
        for i in range(16):
            L[i] = s_L[16 * s_idx + i]

        mat_vec_mult(L, v, d+1, wdwij)
        for i in range(3):
            dwij2[i] = wdwij[i+1]

        tmp = -s_m[s_idx]/(d_rho[d_idx] * s_rho[s_idx]) * (s_p[s_idx] + d_p[d_idx])

        d_au[d_idx] += 0.5 * tmp *(dwij1[0] - dwij2[0])
        d_av[d_idx] += 0.5 * tmp *(dwij1[1] - dwij2[1])
        d_aw[d_idx] += 0.5 * tmp *(dwij1[2] - dwij2[2])

    def post_loop(self, d_idx, d_au, d_av, d_aw, d_dt_force):
        d_au[d_idx] += self.gx
        d_av[d_idx] += self.gy
        d_aw[d_idx] += self.gz

class MomentumEquationSymmModified(Equation):
    # violeu p 5.130
    def __init__(self, dest, sources,
                 gx=0.0, gy=0.0, gz=0.0):

        self.gx = gx
        self.gy = gy
        self.gz = gz
        super(MomentumEquationSymmModified, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw, d_dt_cfl, d_dwij):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0
        d_dwij[3*d_idx] = 0.0
        d_dwij[3*d_idx + 1] = 0.0
        d_dwij[3*d_idx + 2] = 0.0

    def loop(self, d_idx, s_idx, d_rho, d_cs,
             d_p, d_au, d_av, d_aw, s_m,
             s_rho, s_cs, s_p, VIJ,
             XIJ, HIJ, R2IJ, RHOIJ1, EPS,
             DWIJ, WIJ, WDP, d_dt_cfl, d_m, d_dwij, d_rhoc):

        tmp = (s_p[s_idx] - d_p[d_idx])/(d_rhoc[d_idx] * s_rho[s_idx])

        d_au[d_idx] += -s_m[s_idx] * (tmp) * DWIJ[0]
        d_av[d_idx] += -s_m[s_idx] * (tmp) * DWIJ[1]
        d_aw[d_idx] += -s_m[s_idx] * (tmp) * DWIJ[2]

        tmp = s_m[s_idx] / s_rho[s_idx]
        d_dwij[3*d_idx] += tmp * DWIJ[0]
        d_dwij[3*d_idx + 1] += tmp * DWIJ[1]
        d_dwij[3*d_idx + 2] += tmp * DWIJ[2]

    def post_loop(self, d_idx, d_au, d_av, d_aw, d_dt_force, d_rho, d_dwij, d_p):
        d_au[d_idx] += self.gx
        d_av[d_idx] += self.gy
        d_aw[d_idx] += self.gz

        # p =  2 * d_p[d_idx] / d_rho[d_idx]
        # p =  2 * d_p[d_idx] / abs(d_p[d_idx]) / d_rho[d_idx]
        p =  2 * 10.0 / d_rho[d_idx]
        d_au[d_idx] += p * d_dwij[3*d_idx]
        d_av[d_idx] += p * d_dwij[3*d_idx+1]
        d_aw[d_idx] += p * d_dwij[3*d_idx+2]



class MomentumEquationAntiSymm(Equation):
    #Violeu p 331
    def __init__(self, dest, sources,
                 gx=0.0, gy=0.0, gz=0.0):

        self.gx = gx
        self.gy = gy
        self.gz = gz
        super(MomentumEquationAntiSymm, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw, d_dt_cfl):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, d_cs,
             d_p, d_au, d_av, d_aw, s_m,
             s_rho, s_cs, s_p, VIJ,
             XIJ, HIJ, R2IJ, RHOIJ1, EPS,
             DWIJ, WIJ, WDP, d_dt_cfl, d_m, d_rhoc):

        tmp = (s_p[s_idx] - d_p[d_idx])/(d_rhoc[d_idx] * d_rho[s_idx])

        d_au[d_idx] += -s_m[s_idx] * (tmp) * DWIJ[0]
        d_av[d_idx] += -s_m[s_idx] * (tmp) * DWIJ[1]
        d_aw[d_idx] += -s_m[s_idx] * (tmp) * DWIJ[2]

    def post_loop(self, d_idx, d_au, d_av, d_aw, d_dt_force):
        d_au[d_idx] += self.gx
        d_av[d_idx] += self.gy
        d_aw[d_idx] += self.gz

class EvaluateFirstMoment(Equation):
    def initialize(self, d_idx, d_xijdwij):
        d_xijdwij[d_idx] = 0.0

    def loop_all(self, d_idx, s_idx, XIJ, DWIJ, d_xijdwij, s_m, s_rho):
        d_xijdwij[d_idx] += (XIJ[0]*DWIJ[0] + XIJ[1]*DWIJ[1] + XIJ[2]*DWIJ[2]) * s_m[s_idx] / s_rho[s_idx]


class ZeroOrderCorrection(Equation):
    def loop(self, d_idx, d_xijdwij, DWIJ):
        if abs(d_xijdwij[d_idx]) > 1e-14:
            DWIJ[0] /= d_xijdwij[d_idx]
            DWIJ[1] /= d_xijdwij[d_idx]
            DWIJ[2] /= d_xijdwij[d_idx]


# function
def get_standard_ce(eqns, corr):
    from tsph_with_pst import ContinuityEquation
    from pysph.sph.wc.kernel_correction import GradientCorrection
    eqns.append(
        Group(equations=[
            ContinuityEquation(dest="dest", sources=["dest"]),
        ]))
    if corr == 'bonet':
        gradeq = GradientCorrection(dest="dest", sources=["dest"])
        eqns[-1].equations.insert(0, gradeq)
    elif corr == 'liu':
        from kgf_sph import FirstOrderCorrection
        gradeq = FirstOrderCorrection(dest="dest", sources=["dest"])
        eqns[-1].equations.insert(0, gradeq)
    elif corr == 'zo':
        gradeq = ZeroOrderCorrection(dest="dest", sources=["dest"])
        eqns[-1].equations.insert(0, gradeq)


def get_delta_ce(eqns, corr):
    from pysph.sph.basic_equations import ContinuityEquation
    from pysph.sph.wc.basic import (
        ContinuityEquationDeltaSPHPreStep, ContinuityEquationDeltaSPH)
    eqns.append(
        Group(equations=[
            ContinuityEquationDeltaSPHPreStep(dest="dest", sources=["dest"])
        ]))
    eqns.append(
        Group(equations=[
            ContinuityEquation(dest="dest", sources=["dest"]),
            ContinuityEquationDeltaSPH(dest="dest", sources=["dest"], c0=c0)
        ]))


def get_wcsph_me(eqns, corr):
    from pysph.sph.wc.transport_velocity import MomentumEquationPressureGradient
    from pysph.sph.wc.kernel_correction import GradientCorrection
    eqns.append(
        Group(equations=[
            MomentumEquationPressureGradient(dest="dest", sources=["dest"], pb=0.0)
        ]))
    if corr == 'bonet':
        gradeq = GradientCorrection(dest="dest", sources=["dest"])
        eqns[-1].equations.insert(0, gradeq)
    elif corr == 'liu':
        from kgf_sph import KGFCorrection
        gradeq = KGFCorrection(dest="dest", sources=["dest"])
        eqns[-1].equations.insert(0, gradeq)
    elif corr == 'zo':
        gradeq = ZeroOrderCorrection(dest="dest", sources=["dest"])
        eqns[-1].equations.insert(0, gradeq)


def get_pji_me(eqns, corr):
    from pysph.sph.wc.kernel_correction import GradientCorrection
    from tsph_with_pst import MomentumEquationSecondOrder
    eqns.append(
        Group(equations=[
            SetRhoc(dest="dest", sources=None)
        ]))
    eqns.append(
        Group(equations=[
            MomentumEquationSecondOrder(dest="dest", sources=["dest"], rho0=1.0)
        ]))
    if corr == 'bonet':
        gradeq = GradientCorrection(dest="dest", sources=["dest"])
        eqns[-1].equations.insert(0, gradeq)
    elif corr == 'liu':
        from kgf_sph import FirstOrderCorrection
        gradeq = FirstOrderCorrection(dest="dest", sources=["dest"])
        eqns[-1].equations.insert(0, gradeq)
    elif corr == 'zo':
        gradeq = ZeroOrderCorrection(dest="dest", sources=["dest"])
        eqns[-1].equations.insert(0, gradeq)


def get_pji2_me(eqns, corr):
    from pysph.sph.wc.kernel_correction import GradientCorrection
    eqns.append(
        Group(equations=[
            MomentumEquationAntiSymm(dest="dest", sources=["dest"])
        ]))
    if corr == 'bonet':
        gradeq = GradientCorrection(dest="dest", sources=["dest"])
        eqns[-1].equations.insert(0, gradeq)
    elif corr == 'liu':
        from kgf_sph import FirstOrderCorrection
        gradeq = FirstOrderCorrection(dest="dest", sources=["dest"])
        eqns[-1].equations.insert(0, gradeq)
    elif corr == 'zo':
        gradeq = ZeroOrderCorrection(dest="dest", sources=["dest"])
        eqns[-1].equations.insert(0, gradeq)


def get_symm_me(eqns, corr):
    from pysph.sph.wc.kernel_correction import GradientCorrection
    eqns.append(
        Group(equations=[
            MomentumEquationSymm(dest="dest", sources=["dest"])
        ]))
    if corr == 'bonet':
        gradeq = GradientCorrection(dest="dest", sources=["dest"])
        eqns[-1].equations.insert(0, gradeq)
    elif corr == 'liu':
        from kgf_sph import FirstOrderCorrection
        gradeq = FirstOrderCorrection(dest="dest", sources=["dest"])
        eqns[-1].equations.insert(0, gradeq)
    elif corr == 'zo':
        gradeq = ZeroOrderCorrection(dest="dest", sources=["dest"])
        eqns[-1].equations.insert(0, gradeq)


def get_symm_mom_conv(eqns, corr):
    from pysph.sph.wc.kernel_correction import GradientCorrection
    eqns.append(
        Group(equations=[
            LiuCorrectionPreStep(dest="dest", sources=["dest"])
        ]))
    eqns.append(
        Group(equations=[
            CopyLToGhost(dest="dest", sources=None)
        ], real=False))
    eqns.append
    eqns.append(
        Group(equations=[
            MomentumEquationSymmMomConv(dest="dest", sources=["dest"])
        ]))


def get_symm2_me(eqns, corr):
    from pysph.sph.wc.kernel_correction import GradientCorrection
    eqns.append(
        Group(equations=[
            MomentumEquationSymmModified(dest="dest", sources=["dest"])
        ]))
    if corr == 'bonet':
        gradeq = GradientCorrection(dest="dest", sources=["dest"])
        eqns[-1].equations.insert(0, gradeq)
    elif corr == 'liu':
        from kgf_sph import FirstOrderCorrection
        gradeq = FirstOrderCorrection(dest="dest", sources=["dest"])
        eqns[-1].equations.insert(0, gradeq)
    elif corr == 'zo':
        gradeq = ZeroOrderCorrection(dest="dest", sources=["dest"])
        eqns[-1].equations.insert(0, gradeq)


def get_tvf_me(eqns, corr):
    from pysph.sph.wc.transport_velocity import MomentumEquationPressureGradient
    from pysph.sph.wc.kernel_correction import GradientCorrection
    eqns.append(
        Group(equations=[
            MomentumEquationPressureGradient(dest="dest", sources=["dest"], pb=0.0)
        ]))
    if corr == 'bonet':
        gradeq = GradientCorrection(dest="dest", sources=["dest"])
        eqns[-1].equations.insert(0, gradeq)
    elif corr == 'liu':
        from kgf_sph import FirstOrderCorrection
        gradeq = FirstOrderCorrection(dest="dest", sources=["dest"])
        eqns[-1].equations.insert(0, gradeq)
    elif corr == 'zo':
        gradeq = ZeroOrderCorrection(dest="dest", sources=["dest"])
        eqns[-1].equations.insert(0, gradeq)


def get_standard_visc(eqns, corr, eq=None):
    from pysph.sph.wc.transport_velocity import MomentumEquationViscosity
    from pysph.sph.wc.kernel_correction import GradientCorrection
    if eq is None:
        eqns.append(
            Group(equations=[
                MomentumEquationViscosity(dest="dest", sources=["dest"], nu=1.0)
            ]))
    elif eq == 'cleary':
        eqns.append(
            Group(equations=[
                ViscosityCleary(dest="dest", sources=["dest"], nu=1.0, rho0=1.0)
            ]))
    if corr == 'bonet':
        gradeq = GradientCorrection(dest="dest", sources=["dest"])
        eqns[-1].equations.insert(0, gradeq)
    elif corr == 'liu':
        from kgf_sph import FirstOrderCorrection
        gradeq = FirstOrderCorrection(dest="dest", sources=["dest"])
        eqns[-1].equations.insert(0, gradeq)
    elif corr == 'zo':
        gradeq = ZeroOrderCorrection(dest="dest", sources=["dest"])
        eqns[-1].equations.insert(0, gradeq)


def get_visc_coupled(eqns, corr, eq=None):
    from pysph.sph.wc.kernel_correction import GradientCorrection
    from tsph_with_pst import VelocityGradient, DivGrad
    eqns.append(
        Group(equations=[
            GradientCorrection(dest="dest", sources=["dest"]),
            VelocityGradient(dest="dest", sources=["dest"], rho0=1.0)
        ], update_nnps=True))
    if eq == 'coupled':
        eqns.append(
            Group(equations=[
                DivGrad(dest="dest", sources=["dest"], nu=1.0, rho0=1.0)
            ]))
    elif eq == 'fatehi':
        eqns.append(
            Group(equations=[
                FatehiViscosity(dest="dest", sources=["dest"], nu=1.0, rho0=1.0)
            ]))
    if corr == 'bonet':
        gradeq = GradientCorrection(dest="dest", sources=["dest"])
        eqns[-1].equations.insert(0, gradeq)
    elif corr == 'liu':
        # eqns.pop(0)
        from kgf_sph import FirstOrderCorrection
        gradeq = FirstOrderCorrection(dest="dest", sources=["dest"])
        eqns[-1].equations.insert(0, gradeq)
    elif corr == 'zo':
        gradeq = ZeroOrderCorrection(dest="dest", sources=["dest"])
        eqns[-1].equations.insert(0, gradeq)
    elif corr == 'fatehic':
        eqns.pop(-1)
        eqns.append(
        Group(equations=[
            GradientCorrectionPreStepFatehi(dest="dest", sources=["dest"], dim=2),
            CreateFatehiOrder3Tensor(dest="dest", sources=["dest"]),
            CreateFatehiOrder2Tensor(dest="dest", sources=["dest"]),
            CreateFatehiVector(dest="dest", sources=["dest"]),
            EvaluateFatehiCorrection(dest="dest", sources=["dest"], dim=2),
        ], update_nnps=True))
        eqns.append(
            Group(equations=[
                FatehiViscosityCorrected(dest="dest", sources=["dest"], nu=1.0, rho0=1.0)
            ]))


def get_visc_korzilius(eqns, corr, eq=None):
    from pysph.sph.wc.kernel_correction import GradientCorrection
    from tsph_with_pst import VelocityGradient
    eqns.append(
        Group(equations=[
            GradientCorrectionPreStepFatehi(dest="dest", sources=["dest"], dim=2),
            GradientCorrection(dest="dest", sources=["dest"]),
            VelocityGradient(dest="dest", sources=["dest"], rho0=1.0),
        ], update_nnps=True))
    eqns.append(
        Group(equations=[
            Gamma(dest="dest", sources=["dest"]),
            Gamma22(dest="dest", sources=["dest"]),
            KorziliusPreStep(dest="dest", sources=["dest"]),
        ], update_nnps=True))
    eqns.append(
        Group(equations=[
            DoubleDerivativeApprox(dest="dest", sources=["dest"]),
            DoubleDervMoment(dest="dest", sources=["dest"]),
        ], update_nnps=True))


def get_equation(app):
    eqns_name = app.use_sph
    name = eqns_name.split('_')
    from pysph.sph.wc.transport_velocity import SummationDensity
    eqns = []
    eqns.append(
        Group(equations=[
            SummationDensity(dest='dest', sources=['dest']),
            ], update_nnps=True))
    if len(name) > 1:
        if name[-1] == 'bonet' or name[1] == 'fatehi' or name[1] == 'coupled':
            from pysph.sph.wc.kernel_correction import GradientCorrectionPreStep
            eqns.append(
                Group(equations=[
                    GradientCorrectionPreStep(dest='dest', sources=['dest']),
                    ], update_nnps=True))
        elif name[-1] == 'liu':
            from kgf_sph import FirstOrderPreStep
            eqns.append(
                Group(equations=[
                    FirstOrderPreStep(dest='dest', sources=['dest']),
                    ], update_nnps=True))
        elif name[-1] == 'zo':
            eqns.append(
                Group(equations=[
                    EvaluateFirstMoment(dest='dest', sources=['dest']),
                    ], update_nnps=True))

    if name[0] == 'ce':
        if name[-1] == 'delta':
            get_delta_ce(eqns, name[-1])
        else:
            get_standard_ce(eqns, name[-1])
    elif name[0] == 'me':
        if len(name) < 2:
            get_wcsph_me(eqns, name[-1])
        elif name[1] == 'tvf':
            get_tvf_me(eqns, name[1])
        elif name[1] == 'pji':
            get_pji_me(eqns, name[-1])
        elif name[1] == 'pji2':
            get_pji2_me(eqns, name[-1])
        elif name[1] == 'symm':
            get_symm_me(eqns, name[-1])
        elif name[1] == 'symm2':
            get_symm2_me(eqns, name[-1])
        elif name[1] == 'mc':
            get_symm_mom_conv(eqns, name[-1])
        else:
            get_wcsph_me(eqns, name[-1])

    elif name[0] == 'visc':
        if len(name) < 2:
            get_standard_visc(eqns, name[-1])
        elif name[1] == 'cleary':
            get_standard_visc(eqns, name[-1], eq='cleary')
        elif name[1] == 'coupled':
            get_visc_coupled(eqns, name[-1], eq='coupled')
        elif name[1] == 'korzilius':
            get_visc_korzilius(eqns, name[-1])
        elif name[1] == 'fatehi':
            if len(name) > 2:
                if name[-1] == 'fatehi':
                    get_visc_coupled(eqns, 'fatehic', eq='fatehi')
                else:
                    get_visc_coupled(eqns, name[-1], eq='fatehi')
            else:
                get_visc_coupled(eqns, name[-1], eq='fatehi')

        else:
            get_standard_visc(eqns, name[-1])

    return eqns