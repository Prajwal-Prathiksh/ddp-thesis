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


FTLE_TYPES = ['forward', 'backward']
FLTE_METHODS = ['postprocessing']

class FTLE(object):
    def __init__(
        self, dim:int, t0:float, x0:np.ndarray, y0:np.ndarray, z0:np.ndarray,
        m0:np.ndarray, rho0:np.ndarray, h0:np.ndarray, 
        t1:float, x1:np.ndarray, y1:np.ndarray, z1:np.ndarray,
        m1:np.ndarray, rho1:np.ndarray, h1:np.ndarray, 
        ftle_type:str='backward', method:str='postprocessing',
    ):
        self.dim = dim

        self.t0 = t0
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.m0 = m0
        self.rho0 = rho0
        self.h0 = h0
    
        self.t1 = t1
        self.x1 = x1
        self.y1 = y1
        self.z1 = z1
        self.m1 = m1
        self.rho1 = rho1
        self.h1 = h1

        
        if ftle_type not in FTLE_TYPES:
            raise ValueError(
                f"{ftle_type} is not a valid FTLE type. Valid types "
                f"are {FTLE_TYPES}"
            )
        self.ftle_type = ftle_type

        if method not in FLTE_METHODS:
            raise ValueError(
                f"{method} is not a valid FTLE method. Valid methods "
                f"are {FLTE_METHODS}"
            )    
        self.method = method
    
    # Class methods
    @classmethod
    def from_pysph_files(
        cls, dim:int, fname0:str, fname1:str, ftle_type:str='backward'
    ):
        fname0, fname1 = rename_fnames_according_to_time(fname0, fname1)

        def _read_pysph_data(fname:str):
            data = load(fname)
            t = data['solver_data']['t']

            fluid_data = data['arrays']['fluid']
            x = fluid_data.get('x')
            y = fluid_data.get('y')
            z = fluid_data.get('z')

            m, rho = fluid_data.get('m'), fluid_data.get('rho')

            h = fluid_data.get('h')

            return t, x, y, z, m, rho, h

        t0, x0, y0, z0, m0, rho0, h0 = _read_pysph_data(fname0)
        t1, x1, y1, z1, m1, rho1, h1 = _read_pysph_data(fname1)

        return cls(
            dim=dim,
            t0=t0, x0=x0, y0=y0, z0=z0, m0=m0, rho0=rho0, h0=h0,
            t1=t1, x1=x1, y1=y1, z1=z1, m1=m1, rho1=rho1, h1=h1,
            ftle_type=ftle_type,
            method='postprocessing'
        )

    @classmethod
    def from_example(cls, dim:int, ftle_type:str='backward'):
        pi = np.pi
        pass


    
    


def calculate_ftle_backward(
    fname1: str, fname2: str, dim: int, h: float, particle_name: str = 'fluid',
    kernel: object = None, domain_manager: object = None, **kwargs
):
    """
    Calculate the backward-in-time finite-time Lyapunov exponent (FTLE) from
    two time instances of a flow field.

    Parameters
    ----------
    fname1 : str
        Filename of the first time instance.
    fname2 : str
        Filename of the second time instance.
    dim : int
        Dimension of the flow field.
    h : float
        Smoothing length.
    particle_name : str, optional
        Name of the particle group. The default is 'fluid'.
    kernel : object, optional
        Kernel object. The default is WendlandQuinticC4.
    domain_manager : object, optional
        Domain manager object. The default is None.
    **kwargs : dict
        Keyword arguments for the SPHEvaluator.

    Returns
    -------
    ftle : np.ndarray
        Finite-time Lyapunov exponent.
    """
    if kernel is None:
        kernel = WendlandQuinticC4(dim=dim)

    # Load data
    data1 = load(fname1)
    data2 = load(fname2)
    data1, data2 = rename_fnames_according_to_time(data1, data2)

    # Time instances
    t0 = data1['solver_data']['t']  # Initial time
    t = data2['solver_data']['t']  # Final time

    # Get positions and volumes
    # Initial states
    x0 = data1[particle_name]['x']
    y0 = data1[particle_name]['y']
    z0 = data1[particle_name]['z']
    m0 = data1[particle_name]['m']
    rho0 = data1[particle_name]['rho']

    # Final states
    x = data2[particle_name]['x']
    y = data2[particle_name]['y']
    z = data2[particle_name]['z']
    m = data2[particle_name]['m']
    rho = data2[particle_name]['rho']

    # f(x) = sph_sum(Xij)
    # Create particle array
    bit_pa = get_particle_array(
        name='bit_pa', x=x, y=y, z=z, V=V, h=h, m=m, rho=rho,
        xprime=x0, yprime=y0, zprime=z0, mprime=m0, rhoprime=rho0
    )
    bit_pa.add_property('F', stride=9)

    # Equation to solve
    equations = [
        # Group1: GradientCorrectionPreStep
        # Group2:
        #   GradientCorrection
        #   DeformationGradientEquation --> f(xprime_ij, DWIJ, mj, rhoj)
    ]

    # Create SPH evaluator
    bit_ftle_eval = SPHEvaluator(
        arrays=[bit_pa], equations=equations, dim=dim, kernel=kernel,
        domain_manager=domain_manager, **kwargs
    )


def calculate_ftle_forward():
    pass


def calculate_ftle(
    fname1: str, fname2: str, dim: int = 2, particle_name: str = 'fluid',
    method: str = 'backward', kernel: object = None
):
    """
    Calculate the finite-time Lyapunov exponent (FTLE) from two time instances
    of a flow field.

    Parameters
    ----------
    fname1 : str
        Filename of the first time instance.
    fname2 : str
        Filename of the second time instance.
    particle_name : str, optional
        Name of the particle group. The default is 'fluid'.
    method : str, optional
        Method to calculate the FTLE. The default is 'backward'.
    kernel : object, optional
        Kernel object. The default is WendlandQuinticC4.

    Returns
    -------
    ftle : np.ndarray
        Finite-time Lyapunov exponent.
    """
    if kernel is None:
        kernel = WendlandQuinticC4(dim=dim)

    if method == 'backward':
        return calculate_ftle_backward(
            fname1, fname2, dim, particle_name, kernel
        )
    elif method == 'forward':
        return calculate_ftle_forward(
            fname1, fname2, dim, particle_name, kernel
        )
    else:
        raise ValueError(
            "Invalid method. Choose from 'backward' or 'forward'."
        )
