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
from pysph.tools.sph_evaluator import SPHEvaluator
from pysph.base.kernels import WendlandQuinticC4
from pysph.solver.utils import load
from pysph.base.utils import get_particle_array

# Method 1 (Postprocessing-based)
# Forward-in-time FTLE and Backward-in-time FTLE


def rename_fnames_according_to_time(data1: dict, data2: dict):
    """
    Rename the filenames of the two time instances of a flow field according
    to the time of the flow field.

    Parameters
    ----------
    data1 : dict
        Data of the first file.
    data2 : dict
        Data of the second file.

    Returns
    -------
    data1 : dict
        Data of the first time instance.
    data2 : dict
        Data of the second time instance.
    """
    if data1['solver_data']['t'] < data2['solver_data']['t']:
        return data1, data2
    else:
        return data2, data1


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
