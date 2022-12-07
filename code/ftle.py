r"""
Calculation of Backward & Forward-in-time Finite-Time Lyapunov Exponent (FTLE)
#####################
References
-----------
    .. [Sun2016] P. N. Sun, A. Colagrossi, S. Marrone, and A. M. Zhang,
    “Detection of Lagrangian Coherent Structures in the SPH framework,” Comput.
    Methods Appl. Mech. Eng., vol. 305, pp. 849-868, 2016,
    doi: 10.1016/j.cma.2016.03.027.
"""
# Library imports.
import numpy as np
from pysph.tools.sph_evaluator import SPHEvaluator
from pysph.base.kernels import WendlandQuinticC4
from pysph.solver.utils import load
from pysph.base.utils import get_particle_array
from pysph.sph.equation import Equation, Group
from pysph.sph.basic_equations import SummationDensity
from pysph.sph.wc.kernel_correction import (
    GradientCorrectionPreStep, GradientCorrection,
)
import cython
from compyle.api import declare
from textwrap import dedent

from pysph.base.nnps import LinkedListNNPS as NNPS
from pysph.sph.acceleration_eval import AccelerationEval
from pysph.sph.sph_compiler import SPHCompiler
from pysph.sph.wc.linalg import gj_solve, augmented_matrix
from math import log, sqrt

FTLE_TYPES = ['forward', 'backward']
FLTE_METHODS = ['postprocessing']

def extract_counter(fname: str):
    """
    Extract the counter at the end of the filename after the last underscore
    and before the extension.
    Example: 'output_000000.npz' --> '000000'

    Parameters
    ----------
    fname : str
        Filename.
    
    Returns
    -------
    counter : int
        Counter at the end of the filename.
    """
    try:
        counter_str = fname.split('_')[-1].split('.')[0]
    except:
        msg = f"Filename {fname} is not in the correct format. "
        msg += "Valid format: 'dir/fname_000000.npz', "
        msg += "'dir1/dir2/fname_<counter>.hdf5'"
        raise ValueError(msg)
    return int(counter_str)

def rename_fnames_according_to_time(fname0:str, fname1:str):
    """
    Rename the filenames according to the time instance, by reading the counter
    from the filenames and returning the filenames in ascending order of time.

    Parameters
    ----------
    fname0 : str
        First filename.
    fname1 : str
        Second filename.

    Returns
    -------
    fname0 : str
        Filename of the first time instance.
    fname1 : str
        Filename of the second time instance.
    """
    counter0 = extract_counter(fname0)
    counter1 = extract_counter(fname1)
    if counter0 > counter1:
        return fname1, fname0
    return fname0, fname1

class DeformationGradientEquation(Equation):
    def initialize(self, d_idx, d_deform_grad):
        i9, i = declare('int', 2)
        i9 = 9*d_idx

        for i in range(9):
            d_deform_grad[i9 + i] = 0.0

    def loop(
        self, d_idx, s_idx, d_deform_grad, d_x, d_y, d_z, DWIJ,
        s_x_prime, s_y_prime, s_z_prime, s_m, s_rho
    ):
        # Volume
        Vj = s_m[s_idx]/s_rho[s_idx]
        x_prime_ij = declare('matrix(3)')
        x_prime_ij[0] = s_x_prime[s_idx] - d_x[d_idx]
        x_prime_ij[1] = s_y_prime[s_idx] - d_y[d_idx]
        x_prime_ij[2] = s_z_prime[s_idx] - d_z[d_idx]

        i9, i, j = declare('int', 3)
        i9 = 9*d_idx

        # Tensor product of x_prime_ij and DWIJ
        for i in range(3):
            for j in range(3):
                d_deform_grad[i9 + 3*i + j] += x_prime_ij[i]*DWIJ[j]*Vj

class LyapunovExponentEquation(Equation):
    def __init__(self, dest, sources, t0, tf, dim, ftle_type):
        self.t0 = t0
        self.tf = tf

        delta_t = abs(tf - t0)
        self.delta_t_inv = 1.0/delta_t

        self.dim = dim

        if ftle_type not in FTLE_TYPES:
            raise ValueError(
                f"{ftle_type} is not a valid FTLE type. Valid types "
                f"are {FTLE_TYPES}"
            )    
        self.ftle_type = ftle_type
        super(LyapunovExponentEquation, self).__init__(dest, sources)

    def _cython_code_(self):
        code = dedent("""
        cimport cython
        from pysph.base.linalg3 cimport eigen_decomposition
        from pysph.base.linalg3 cimport transform_diag_inv
        """)
        return code
    
    def initialize(self, d_idx, d_ftle):
        d_ftle[d_idx] = 0.0
    
    def loop(self, d_idx, d_deform_grad, d_ftle):
        i9, i, j, k = declare('int', 4)
        i9 = 9*d_idx

        # Compute matrix multiplication of deform_grad.T and deform_grad
        C_ij = declare('matrix((3,3))')
        for i in range(3):
            for j in range(3):
                C_ij[i][j] = 0.0
                for k in range(3):
                    f_transpose_i = d_deform_grad[i9 + 3*k + i]
                    f_j = d_deform_grad[i9 + 3*j + k]
                    C_ij[i][j] += f_transpose_i*f_j
        
        if self.dim == 2:
            # Overwrite the last row and column of C_ij with zeros
            for i in range(3):
                C_ij[i][2] = 0.0
                C_ij[2][i] = 0.0

        # Matrix of Eigenvectors (columns)
        R = declare('matrix((3,3))')

        # Eigenvalues
        V = declare('matrix((3,))')

        # TODO: Profile this function call to identify if it is a bottleneck

        # Compute eigenvalues and eigenvectors
        eigen_decomposition(C_ij, R, cython.address(V[0]))

        # Sort eigenvalues in ascending order
        V.sort()

        if self.ftle_type == 'forward':
            i = 2
        else:
            if self.dim == 2:
                i = 0
            else:
                i = 1
        d_ftle[d_idx] = self.delta_t_inv * log(sqrt(V[i]))



class FTLyapunovExponent(object):
    def __init__(
        self, dim:int, t0, pa_0, tf, pa_f,
        method:str='postprocessing', kernel=None, domain_manager=None,
    ):
        self.dim = dim

        self.t0 = t0
        self.pa_0 = pa_0

        self.tf = tf
        self.pa_f = pa_f

        if method not in FLTE_METHODS:
            raise ValueError(
                f"{method} is not a valid FTLE method. Valid methods "
                f"are {FLTE_METHODS}"
            )    
        self.method = method

        # Setup particle arrays
        self._setup_particle_arrays()
        self.arrays = [self.pa_0, self.pa_f]

        # Solver related attributes
        self.nnps = None
        self.equations = None
        self.func_eval = None
        self.domain_manager = domain_manager
        if kernel is None:
            kernel = WendlandQuinticC4(dim=dim)
        self.kernel = kernel
    
    # Class methods
    @classmethod
    def from_pysph_files(
        cls, dim:int, fname_i:str, fname_f:str, kernel=None, 
        domain_manager=None
    ):
        fname_i, fname_f = rename_fnames_according_to_time(fname_i, fname_f)

        def _read_pysph_data(fname:str):
            data = load(fname)
            t = data['solver_data']['t']
            pa = data['arrays']['fluid']
            return t, pa
        
        t0, pa_0 = _read_pysph_data(fname_i)
        t1, pa_f = _read_pysph_data(fname_f)

        return cls(
            dim=dim,
            t0=t0, pa_0=pa_0, 
            tf=t1, pa_f=pa_f,
            method='postprocessing',
            kernel=kernel,
            domain_manager=domain_manager,
        )
    
    @classmethod
    def from_ndarray(
        cls, dim:int, t0:float, x0:np.ndarray, y0:np.ndarray, z0:np.ndarray,
        m0:np.ndarray, rho0:np.ndarray, h0:np.ndarray,
        tf:float, xf:np.ndarray, yf:np.ndarray, zf:np.ndarray,
        mf:np.ndarray, rhof:np.ndarray, hf:np.ndarray,
        kernel=None, domain_manager=None,
    ):
        pa_0 = get_particle_array(
            name='initial', x=x0, y=y0, z=z0, m=m0, rho=rho0, h=h0
        )
        pa_f = get_particle_array(
            name='final', x=xf, y=yf, z=zf, m=mf, rho=rhof, h=hf
        )
        return cls(
            dim=dim,
            t0=t0, pa_0=pa_0,
            tf=tf, pa_f=pa_f,
            method='postprocessing',
            kernel=kernel,
            domain_manager=domain_manager,
        )

    @classmethod
    def from_example(
        cls, dim:int, nx:int, flow_type:str,
    ):
        pi = np.pi
        sin, cos = np.sin, np.cos

        _x = np.arange(-1, 1+1./nx, 1./nx)
        if dim == 2:
            X, Y = np.meshgrid(_x, _x)
            R2 = X**2 + Y**2
            flow_types = ['parabolic', 'spiral']
            if flow_type not in flow_types:
                raise ValueError(
                    f"{flow_type} is not a valid flow type. Valid flow types "
                    f"are {flow_types}"
                )
            
            if flow_type == 'parabolic':
                x = 1.5*X
                y = X**2 + Y
            elif flow_type == 'spiral':
                x = X + cos(2*pi*R2)
                y = Y + sin(2*pi*R2)
            
            Z = z = np.zeros_like(x)

        else:
            raise NotImplementedError("Only 2D is implemented.")
        
        m = np.ones_like(x)
        rho = np.ones_like(x)
        h = np.ones_like(x)

        pa_0 = get_particle_array(
            name='initial', x=X, y=Y, z=Z, m=m, rho=rho, h=h
        )
        pa_f = get_particle_array(
            name='final', x=x, y=y, z=z, m=m, rho=rho, h=h
        )

        return cls(
            dim=dim,
            t0=0., pa_0=pa_0,
            tf=1., pa_f=pa_f,
            method='postprocessing'
        )

    # Private methods
    def _setup_particle_arrays(self):
        initial_props = self.pa_0.properties
        initial_output_props = self.pa_0.output_property_arrays
        # Rename particle arrays
        self.pa_0.set_name('initial')
        self.pa_f.set_name('final')

        # Ensure properties are present
        props = {
            'deform_grad': 9,
            'm_mat': 9,
            'ftle': 1,
        }
        for prop in props.keys():
            if prop not in self.pa_0.properties:
                self.pa_0.add_property(prop, stride=props[prop])
            if prop not in self.pa_f.properties:
                self.pa_f.add_property(prop, stride=props[prop])
        
        # Copy properties from pa_0 to pa_f and vice versa --> Termed as
        # prime properties
        prime_props_origin = ['x', 'y', 'z', 'm', 'rho', 'h']
        prime_props_target = [
            'x_prime', 'y_prime', 'z_prime', 'm_prime', 'rho_prime', 'h_prime'
        ]
        for prop_o, prop_t in zip(prime_props_origin, prime_props_target):
            self.pa_0.add_property(prop_t, stride=1)
            # Copy from pa_f[prop_o] to pa_0[prop_t]
            self.pa_0.get(prop_t, only_real_particles=False)[:] = \
                self.pa_f.get(prop_o, only_real_particles=False)

            # Copy from pa_0[prop_o] to pa_f[prop_t]
            self.pa_f.add_property(prop_t, stride=1)
            self.pa_f.get(prop_t, only_real_particles=False)[:] = \
                self.pa_0.get(prop_o, only_real_particles=False)

        # Output properties
        output_props = initial_output_props + ['deform_grad', 'ftle']
        self.pa_0.set_output_arrays(output_props)
        self.pa_f.set_output_arrays(output_props)

    def _get_equations(self, ftle_type):
        if ftle_type == 'forward':
            pa_name = 'initial'
        elif ftle_type == 'backward':
            pa_name = 'final'
    
        equations = [
            Group(
                equations=[
                    SummationDensity(dest=pa_name, sources=[pa_name]),
                ], real=False
            ),
            Group(
                equations=[
                    GradientCorrectionPreStep(
                        dest=pa_name, sources=[pa_name], dim=self.dim
                    )
                ], real=False
            ),
            Group(
                equations=[
                    GradientCorrection(
                        dest=pa_name, sources=[pa_name], dim=self.dim
                    ),
                    DeformationGradientEquation(
                        dest=pa_name, sources=[pa_name]
                    ),
                    LyapunovExponentEquation(
                        dest=pa_name, sources=[pa_name], t0=self.t0,
                        tf=self.tf, dim=self.dim, ftle_type=ftle_type
                    )
                ], real=True
            ),
        ]
        
        return equations
    
    def _compile_acceleration_eval(self, mode, backend):
        self.func_eval = AccelerationEval(
            particle_arrays=self.arrays, equations=self.equations,
            kernel=self.kernel, mode=mode, backend=backend
        )
        print(f"{mode = }")
        compiler = SPHCompiler(
            acceleration_evals=self.func_eval, integrator=None
        )
        compiler.compile()

    def _create_nnps(self):
        self.nnps = NNPS(
            dim=self.dim, particles=self.arrays,
            radius_scale=self.kernel.radius_scale, domain=self.domain_manager,
            cache=True
        )
        self.func_eval.set_nnps(self.nnps)
    
    # Public methods
    def compute(self, ftle_type, mode='mpi', backend='cython'):
        if ftle_type not in FTLE_TYPES:
            raise ValueError(
                f"{ftle_type} is not a valid FTLE type. Valid types "
                f"are {FTLE_TYPES}"
            )

        self.equations = self._get_equations(ftle_type)
        self._compile_acceleration_eval(mode=mode, backend=backend)
        self._create_nnps()
        self.func_eval.compute(t=0.0, dt=0.1) # Passing junk arguments

        if ftle_type == 'forward':
            result = self.pa_0.get('ftle').copy()
        elif ftle_type == 'backward':
            result = self.pa_f.get('ftle').copy()
        
        return result


if __name__ == '__main__':
    import time
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dim', type=int, default=2,
        help='Dimension of the problem'
    )
    parser.add_argument(
        '--nx', type=int, default=10,
        help='Number of particles along x'
    )
    parser.add_argument(
        '--openmp', action='store_true',
        help='Use OpenMP'
    )
    parser.add_argument(
        '--backend', type=str, default='cython',
        help="Backend to use. Options: 'opencl', 'cython', 'cuda'"
    )

    args = parser.parse_args()
    dim = args.dim
    nx = args.nx
    openmp = args.openmp
    if openmp:
        mode = 'mpi'
    else:
        mode = 'serial'
    backend = args.backend



    ftle_ob = FTLyapunovExponent.from_example(dim=dim, nx=nx, flow_type='parabolic')
    # Time forward
    t0 = time.time()
    ftle_ob.compute(ftle_type='forward', mode=mode, backend=backend)
    t1 = time.time()
    print(f"Computed forward FTLE in {t1-t0} seconds")
    # Time backward
    t0 = time.time()
    ftle_ob.compute(ftle_type='backward', mode=mode)
    t1 = time.time()
    print(f"Computed backward FTLE in {t1-t0} seconds")
