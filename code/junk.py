from pysph.base.utils import get_particle_array
from pysph.sph.acceleration_eval import AccelerationEval
from pysph.base.kernels import CubicSpline
from pysph.base.nnps import LinkedListNNPS as NNPS
from pysph.sph.sph_compiler import SPHCompiler
from pysph.sph.equation import Equation, Group
from compyle.api import declare
from pysph.sph.basic_equations import SummationDensity
from pysph.sph.wc.kernel_correction import GradientCorrectionPreStep
from pysph.sph.wc.density_correction import gj_solve

import numpy as np
import matplotlib.pyplot as plt

class GradientCorrection(Equation):
    r"""**Kernel Gradient Correction**

    From [BonetLok1999], equations (42) and (45)

    .. math::
            \nabla \tilde{W}_{ab} = L_{a}\nabla W_{ab}

    .. math::
            L_{a} = \left(\sum \frac{m_{b}}{\rho_{b}} \nabla W_{ab}
            \mathbf{\otimes}x_{ba} \right)^{-1}
    """

    def _get_helpers_(self):
        return [gj_solve]

    def __init__(self, dest, sources, dim=2, tol=0.1):
        self.dim = dim
        self.tol = tol
        super(GradientCorrection, self).__init__(dest, sources)

    def loop(self, d_idx, d_m_mat, DWIJ, HIJ):
        i, j, n, nt = declare('int', 4)
        n = self.dim
        nt = n + 1
        # Note that we allocate enough for a 3D case but may only use a
        # part of the matrix.
        temp = declare('matrix(12)')
        res = declare('matrix(3)')
        eps = 1.0e-04 * HIJ
        for i in range(n):
            for j in range(n):
                temp[nt * i + j] = d_m_mat[9 * d_idx + 3 * i + j]
            # Augmented part of matrix
            temp[nt*i + n] = DWIJ[i]

        gj_solve(temp, n, 1, res)

        print('d_idx = ', d_idx)
        print('DWIJ = ', [DWIJ[i] for i in range(3)])

        res_mag = 0.0
        dwij_mag = 0.0
        for i in range(n):
            res_mag += abs(res[i])
            dwij_mag += abs(DWIJ[i])
        change = abs(res_mag - dwij_mag)/(dwij_mag + eps)
        if change < self.tol:
            for i in range(n):
                DWIJ[i] = res[i]
        
        print('res = ', [res[i] for i in range(3)])
        print('DWIJ = ', [DWIJ[i] for i in range(3)])
        print('change = ', change)
        print('************')

def _make_accel_eval(pa, dim, equations, cache_nnps=False):
    arrays = [pa]
    kernel = CubicSpline(dim=dim)
    a_eval = AccelerationEval(
        particle_arrays=arrays, equations=equations, kernel=kernel
    )
    comp = SPHCompiler(a_eval, integrator=None)
    comp.compile()
    nnps = NNPS(dim=kernel.dim, particles=arrays, cache=cache_nnps)
    nnps.update()
    a_eval.set_nnps(nnps)
    return a_eval

def print_prop_with_stride(prop, stride, shape, txt='Deformation gradient:'):
    prop = np.round(prop, 2)
    print(txt)
    N = len(prop)
    for i in range(N//stride):
        print(prop[i*stride:(i+1)*stride].reshape(shape))

def print_change_in_coords(
    x, y, z, x_prime, y_prime, z_prime, show_3d=False
):
    print("Change in coordinates: ")
    n = len(x)
    for i in range(n):
        if not show_3d:
            t = f"({x[i]:.2f}, {y[i]:.2f}) --> "
            t += f"({x_prime[i]:.2f}, {y_prime[i]:.2f})"
        else:
            t = f"({x[i]:.2f}, {y[i]:.2f}, {z[i]:.2f}) --> "
            t += f"({x_prime[i]:.2f}, {y_prime[i]:.2f}, {z_prime[i]:.2f})"
        print(t)

def make_pa(n=5):
    dim = 2
    dx = 1.0 / (n - 1)
    x = y = np.linspace(0, 1, n)
    z = np.zeros_like(x)
    m = np.ones_like(x)
    h = np.ones_like(x) * 1.05
    x_prime = np.exp(-x)
    y_prime = np.exp(-y)
    z_prime = 0.0

    n_round = 2
    x_prime = np.round(x_prime, n_round)
    y_prime = np.round(y_prime, n_round)
    z_prime = np.round(z_prime, n_round)

    pa = get_particle_array(
        name='fluid', x=x, y=y, z=z,
        h=h, m=m,
        x_prime=x_prime, y_prime=y_prime, z_prime=z_prime
    )

    # Ensure properties are present
    props = {
    'deform_grad': 9,
    'deform_grad_exact': 9,
    'm_mat': 9,
    'ftle': 1,
    }
    for prop in props.keys():
        if prop not in pa.properties:
            pa.add_property(prop, stride=props[prop])

    # Exact deformation gradient
    exact = []
    for i in range(n):
        if dim == 2:
            coord = [x[i], y[i], np.inf]
        else:
            coord = [x[i], y[i], z[i]]
        mat = np.diag(-np.exp(-np.array(coord)))
        exact.append(mat.flatten())
    exact = np.array(exact)
    pa.deform_grad_exact = exact.flatten()
    
    return pa, dim

class DeformationGradientEquation(Equation):
    def __init__(self, dest, sources, dim=2):
        self.dim = dim
        super().__init__(dest, sources)

    def initialize(self, d_idx, d_deform_grad):
        i9, i = declare('int', 2)
        i9 = 9*d_idx
            
        for i in range(9):
            d_deform_grad[i9 + i] = 0.0
        
        d_deform_grad[i9] = 1.0
        d_deform_grad[i9 + 4] = 1.0
        if self.dim == 3:
            d_deform_grad[i9 + 8] = 1.0

    def loop(
        self, d_idx, s_idx, d_deform_grad, d_x, d_y, d_z,
        d_x_prime, d_y_prime, d_z_prime,
        DWIJ,
        s_x_prime, s_y_prime, s_z_prime, s_m, s_rho
    ):
        # Volume
        Vj = s_m[s_idx]/s_rho[s_idx]
        x_prime_ji = declare('matrix(3)')
        x_prime_ji[0] = s_x_prime[s_idx] - d_x_prime[d_idx]
        x_prime_ji[1] = s_y_prime[s_idx] - d_y_prime[d_idx]
        x_prime_ji[2] = s_z_prime[s_idx] - d_z_prime[d_idx]

        i9, i, j = declare('int', 3)
        i9 = 9*d_idx

        # Tensor product of x_prime_ij and DWIJ
        for i in range(3):
            for j in range(3):
                d_deform_grad[i9 + 3*i + j] += x_prime_ji[i]*DWIJ[j]*Vj
        
        print("(d_idx, s_idx): ", (d_idx, s_idx))
        print("d_x: ", [d_x[d_idx], d_y[d_idx], d_z[d_idx]])
        print("d_x_prime: ", [d_x_prime[d_idx], d_y_prime[d_idx], d_z_prime     [d_idx]])
        print("s_x_prime: ", [s_x_prime[s_idx], s_y_prime[s_idx], s_z_prime[s_idx]])
        print("x_prime_ji: ", x_prime_ji)
        print("DWIJ:", [DWIJ[i] for i in range(3)])
        print("Vj: ", Vj)
        print("d_deform_grad: ")
        print("\t", [d_deform_grad[i9 + i] for i in range(3)])
        print("\t", [d_deform_grad[i9 + i] for i in range(3, 6)])
        print("\t", [d_deform_grad[i9 + i] for i in range(6, 9)])
        print("-"*10)

def main():
    pa, dim = make_pa(n=2)
    print_prop_with_stride(pa.deform_grad, 9, (3, 3))
    print_change_in_coords(
        x=pa.x, y=pa.y, z=pa.z,
        x_prime=pa.x_prime, y_prime=pa.y_prime, z_prime=pa.z_prime,
        show_3d=False
    )

    equations = [
        Group(
            equations=[
                SummationDensity(dest='fluid', sources=['fluid']),
            ], real=False
        ),
            Group(
                equations=[
                    GradientCorrectionPreStep(
                        dest='fluid', sources=['fluid'], dim=dim
                    )
                ], real=False
            ),
        Group(
            equations=[
                GradientCorrection(
                        dest='fluid', sources=['fluid'], dim=dim, tol=0.2
                ),
                DeformationGradientEquation(
                    dest='fluid', sources=['fluid'], dim=dim
                ),
            ], real=True
        ),
    ]

    
    a_eval = _make_accel_eval(pa=pa, dim=dim, equations=equations)
    
    n_temp = 40
    print("\nComputing acceleration")
    print("="*n_temp)
    a_eval.compute(0.1, 0.1)
    print("="*n_temp)
    print("Done computing acceleration\n")
    
    print_prop_with_stride(pa.deform_grad, 9, (3, 3))
    print_prop_with_stride(
        pa.deform_grad_exact, 9, (3, 3),
        txt='Deformation gradient (EXACT):'
    )

if __name__ == '__main__':
    main()