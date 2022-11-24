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
from compyle.api import declare

from pysph.base.nnps import LinkedListNNPS as NNPS
from pysph.sph.acceleration_eval import AccelerationEval
from pysph.sph.sph_compiler import SPHCompiler
from pysph.sph.wc.linalg import gj_solve, augmented_matrix

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
    counter : str
        Counter.
    """
    return fname.split('_')[-1].split('.')[0]

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
        return fname0, fname1
    return fname1, fname0

class DeformationGradientForward(Equation):
    def initialize(self, d_idx, d_F):
        i9 = declare('int')
        i9 = 9*d_idx

        for i in range(9):
            d_F[i9 + i] = 0.0

    def l
    


#TODO: Equation to calculate the Cauchy-Green strain tensor.
#TODO: Equation to calculate eigenvalues of the Cauchy-Green strain tensor.
#TODO: Compyle eigenvalue


FTLE_TYPES = ['forward', 'backward']
FLTE_METHODS = ['postprocessing']

class FTLE(object):
    def __init__(
        self, dim:int, pa_0, pa_f,
        method:str='postprocessing', kernel=None, domain_manager=None,
        mode=None, backend=None
    ):
        self.dim = dim

        self.pa_0 = pa_0
        self.pa_f = pa_f

        if method not in FLTE_METHODS:
            raise ValueError(
                f"{method} is not a valid FTLE method. Valid methods "
                f"are {FLTE_METHODS}"
            )    
        self.method = method

        # Setup particle arrays
        self.pa_0, self.pa_f = self._setup_particle_arrays()
        self.arrays = [self.pa_0, self.pa_f]

        # Solver related attributes
        self.nnps = None
        self.equations = None
        self.func_eval = None
        self.domain_manager = domain_manager
        if kernel is None:
            kernel = WendlandQuinticC4(dim=dim)
        self.kernel = kernel
        self.mode = mode
        self.backend = backend
    
    # Class methods
    @classmethod
    def from_pysph_files(
        cls, dim:int, fname_i:str, fname_f:str, kernel=None, 
        domain_manager=None, mode=None, backend=None
    ):
        fname_i, fname_f = rename_fnames_according_to_time(fname_i, fname_f)

        def _read_pysph_data(fname:str):
            data = load(fname)
            t = data['solver_data']['t']
            pa = data['arrays']['fluid']
            return t, pa
        
        t0, pa0 = _read_pysph_data(fname_i)
        t1, pa1 = _read_pysph_data(fname_f)

        return cls(
            dim=dim,
            pa_init=pa0,
            pa_final=pa1,
            method='postprocessing'
            kernel=kernel,
            domain_manager=domain_manager,
            mode=mode,
            backend=backend
        )
    
    @classmethod
    def from_ndarray(
        cls, t0, x0:np.ndarray, y0:np.ndarray, z0:np.ndarray,
        
        
         y_i:np.ndarray, z_i:np.ndarray,
        m_i:np.ndarray, rho_i:np.ndarray, h_i:np.ndarray,



    ):

    @classmethod
    def from_example(
        cls, dim:int, nx:int, flow_type:str,
    ):
        pi = np.pi
        sin, cos = np.sin, np.cos

        _x = np.arange(0, 1+1./nx, 1./nx)
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
            
            z = np.zeros_like(x)

        else:
            raise NotImplementedError("Only 2D is implemented.")
        
        m = np.ones_like(x)
        rho = np.ones_like(x)
        h = np.ones_like(x)

        return cls(
            dim=dim,
            t0=0, x0=X, y0=Y, z0=z, m0=m, rho0=rho, h0=h,
            t1=1, x1=x, y1=y, z1=z, m1=m, rho1=rho, h1=h,
            method='postprocessing'
        )

    
    # Private methods
    def _setup_particle_arrays(self):
        props = {
            'deform_grad': 9,
            'm_mat': 9,
            'ftle': 1,
        }
        for prop in props.keys():
            if prop not in self.pa_i.properties:
                self.pa_i.add_property(prop, stride=props[prop])
            if prop not in self.pa_f.properties:
                self.pa_f.add_property(prop, stride=props[prop])

    def _get_equations(self, ftle_type):
        if ftle_type == 'forward':
            pass
        elif ftle_type == 'backward':
            pass
            #TODO
        return None
    
    def _compile_acceleration_eval(self):
        self.func_eval = AccelerationEval(
            particle_arrays=self.arrays, equations=self.equations,
            kernel=self.kernel, mode=self.mode, backend=self.backend
        )
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
    def compute(self, ftle_type):
        if ftle_type not in FTLE_TYPES:
            raise ValueError(
                f"{ftle_type} is not a valid FTLE type. Valid types "
                f"are {FTLE_TYPES}"
            )

        self.equations = self._get_equations(ftle_type, self.t0, self.tf)
        self._compile_acceleration_eval()
        self._create_nnps()
        self.func_eval.compute(t=0.0, dt=0.1) # Passing junk arguments

        if ftle_type == 'forward':
            result = self.pa_i.get('ftle').copy()
        elif ftle_type == 'backward':
            result = self.pa_f.get('ftle').copy()
        
        result.shape = self.shape
        return result.squeeze()