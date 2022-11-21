r"""
A Sinusoidal Velocity Profile
##############################
"""
# Library imports
import os
import numpy as np
from pysph.base.nnps import DomainManager
from pysph.base.utils import get_particle_array
from pysph.solver.application import Application
from pysph.sph.integrator import Integrator
from pysph.base.kernels import (
    CubicSpline, WendlandQuinticC2_1D, WendlandQuintic, WendlandQuinticC4_1D,
    WendlandQuinticC4, WendlandQuinticC6_1D, WendlandQuinticC6,
    Gaussian, SuperGaussian, QuinticSpline
)
from pysph.solver.solver import Solver
from pysph.sph.equation import Equation, Group
from pysph.sph.basic_equations import SummationDensity
from pysph.tools.interpolator import (
    SPHFirstOrderApproximationPreStep, SPHFirstOrderApproximation
)
from pysph.sph.wc.kernel_correction import (
    GradientCorrectionPreStep, GradientCorrection
)

# Local imports
from turbulence_tools import TurbulentFlowApp

#TODOL Add a more robust/realistic test case?

# Kernel choices
KERNEL_CHOICES = [
    'CubicSpline', 'WendlandQuinticC2', 'WendlandQuinticC4',
    'WendlandQuinticC6', 'Gaussian', 'SuperGaussian', 'QuinticSpline'
]

# Interpolating method choices
INTERPOLATING_METHOD_CHOICES = ['sph', 'shepard', 'order1', 'order1BL']

def get_kernel_cls(name: str, dim: int):
    """
        Return the kernel class corresponding to the name initialized with the
        dimension.

        Parameters
        ----------
        name : str
            Name of the kernel class.
        dim : int
            Dimension of the kernel.

        Returns
        -------
        kernel_cls : class
            Kernel class (dim).
    """
    if dim not in [1, 2, 3]:
        raise ValueError("Dimension must be 1, 2 or 3.")
    mapper = {
        'CubicSpline': CubicSpline,
        'WendlandQuinticC2': WendlandQuinticC2_1D if dim == 1
        else WendlandQuintic,
        'WendlandQuinticC4': WendlandQuinticC4_1D if dim == 1
        else WendlandQuinticC4,
        'WendlandQuinticC6': WendlandQuinticC6_1D if dim == 1
        else WendlandQuinticC6,
        'Gaussian': Gaussian,
        'SuperGaussian': SuperGaussian,
        'QuinticSpline': QuinticSpline
    }
    if name not in mapper:
        raise ValueError("Kernel name not recognized")
    return mapper[name](dim=dim)


def perturb_signal(perturb_fac: float, *args: np.ndarray):
    """
    Perturb the given signal/s by a uniform random number between [0, 1) scaled
    by the given factor.
    The random number is seeded by numpy.random.seed(1) to ensure
    reproducibility.

    Parameters
    ----------
    perturb_fac : float
        Factor by which the signal is to be perturbed by a uniform random
        number.
    *args : np.ndarray
        Signals to be perturbed.

    Returns
    -------
    perturbed_signals : list
        Perturbed signals.
    """
    if perturb_fac > 0.:
        np.random.seed(1)
        return [
            arg + perturb_fac * np.random.random(arg.shape) for arg in args
        ]
    else:
        return args


class DummyIntegrator(Integrator):
    def one_timestep(self):
        pass


class SinVelocityProfile(TurbulentFlowApp):
    """
    Particles having a sinusoidal velocity profile.
    """

    def add_user_options(self, group):
        group.add_argument(
            "--perturb", action="store", type=float, dest="perturb", default=0,
            help="Random perturbation of initial particles as a fraction "
            "of dx (setting it to zero disables it, the default)."
        )
        group.add_argument(
            "--nx", action="store", type=int, dest="nx", default=101,
            help="Number of points along x direction."
        )
        group.add_argument(
            "--hdx", action="store", type=float, dest="hdx", default=1.0,
            help="Ratio h/dx."
        )
        group.add_argument(
            "--dim", action="store", type=int, dest="dim", default=2,
            help="Dimension of the problem."
        )
        group.add_argument(
            "--i-nx", action="store", type=int, dest="i_nx", default=None,
            help="Number of interpolation points along x direction. If not "
            "specified, it is set to nx."
        )
        group.add_argument(
            "--i-kernel", action="store", type=str, dest="i_kernel",
            default='WendlandQuinticC2', choices=KERNEL_CHOICES,
            help="Interpolation kernel."
        )
        group.add_argument(
            "--i-method", action="store", type=str, dest="i_method",
            default='sph', choices=INTERPOLATING_METHOD_CHOICES,
            help="Interpolating method."
        )

    def consume_user_options(self):
        self.perturb = self.options.perturb
        self.nx = self.options.nx
        self.hdx = self.options.hdx
        self.dim = self.options.dim

        i_nx = self.options.i_nx
        self.i_nx = self.nx if i_nx is None else i_nx
        self.i_kernel = self.options.i_kernel
        self.i_kernel_cls = get_kernel_cls(self.i_kernel, self.dim)
        self.i_method = self.options.i_method

        self.dx = dx = 1. / self.nx
        self.volume = dx**self.dim

        self.L = 1.
        self.rho0 = 1.

    def create_domain(self):
        print("create_domain: domain created")
        if self.dim == 1:
            dm = DomainManager(
                xmin=0, xmax=self.L, periodic_in_x=True
            )
        elif self.dim == 2:
            dm = DomainManager(
                xmin=0, xmax=self.L, ymin=0, ymax=self.L, periodic_in_x=True,
                periodic_in_y=True
            )
        elif self.dim == 3:
            dm = DomainManager(
                xmin=0, xmax=self.L, ymin=0, ymax=self.L, zmin=0, zmax=self.L,
                periodic_in_x=True, periodic_in_y=True, periodic_in_z=True
            )

        return dm

    def create_particles(self):
        # Create the particles
        dx = self.dx

        _x = np.arange(dx / 2, self.L, dx)
        twopi = 2 * np.pi
        cos, sin = np.cos, np.sin
        if self.dim == 1:
            x = perturb_signal(self.perturb, _x)[0]
            y = z = 0.
            u0 = - cos(twopi * x)
            v0 = w0 = 0.
        elif self.dim == 2:
            x, y = np.meshgrid(_x, _x)
            x, y = perturb_signal(self.perturb, x, y)
            z = 0.
            u0 = - cos(twopi * x) * sin(twopi * y)
            v0 = sin(twopi * x) * cos(twopi * y)
            w0 = 0.
        elif self.dim == 3:
            x, y, z = np.meshgrid(_x, _x, _x)
            x, y, z = perturb_signal(self.perturb, x, y, z)
            u0 = - cos(twopi * x) * sin(twopi * y) * sin(twopi * z)
            v0 = sin(twopi * x) * cos(twopi * y) * sin(twopi * z)
            w0 = - sin(twopi * x) * sin(twopi * y) * cos(twopi * z)
        else:
            raise ValueError("Dimension should be 1, 2 or 3.")

        vmag = np.sqrt(u0**2 + v0**2 + w0**2)

        # Initialize
        m = self.volume * self.rho0
        h = self.hdx * dx

        # Create the arrays
        pa = get_particle_array(
            name='fluid', x=x, y=y, z=z, m=m, h=h,
            u=u0, v=v0, w=w0, rho=self.rho0, vmag=vmag
        )
        pa.set_output_arrays(
            [
                'x', 'y', 'z', 'u', 'v', 'w', 'vmag',
                'rho', 'm', 'h', 'pid', 'gid', 'tag'
            ]
        )

        print("Created %d particles" % pa.get_number_of_particles())
        return [pa]

    def create_solver(self):
        dim = self.dim
        dt = 1
        tf = dt * 1.1

        kernel = CubicSpline(dim=dim)

        integrator = DummyIntegrator()

        solver = Solver(
            kernel=kernel, dim=dim, integrator=integrator, dt=dt, tf=tf
        )
        solver.set_print_freq(1)
        solver.set_max_steps(0)
        print("create_solver: solver created")
        return solver

    def create_equations(self):
        print("create_equations: equations created")
        return []

    # The following are all related to post-processing.
    def get_exact_energy_spectrum(self):
        dim = self.dim

        N = int(1 + np.ceil(np.sqrt(dim*self.i_nx**2)/2))
        Ek_exact = np.zeros(N, dtype=np.float64)

        if dim == 1:
            Ek_exact[1] = 0.125
        elif dim == 2:
            Ek_exact[1] = 0.125
        elif dim == 3:
            Ek_exact[2] = 0.09375

        return Ek_exact            

    def post_process(self, info_fname):
        info = self.read_info(info_fname)

        dim = self.dim
        if len(self.output_files) == 0:
            return

        from energy_spectrum import EnergySpectrum

        espec_ob = EnergySpectrum.from_pysph_file(
            fname=self.output_files[0],
            dim=dim,
            L=self.L,
            i_nx=self.i_nx,
            kernel=self.i_kernel_cls,
            domain_manager=self.create_domain(),
            method=self.i_method,
            U0=1.
        )
        espec_ob.compute()
        fname = os.path.join(self.output_dir, 'energy_spectrum_log.png')
        espec_ob.plot_scalar_Ek(
            savefig=True,
            fname=fname,
            plot_type='log'
        )
        espec_ob.plot_scalar_Ek(
            savefig=True,
            fname=fname.replace('_log', '_stem'),
            plot_type='stem'
        )
        fname = os.path.join(self.output_dir, 'EK_spectrum_shiftted.png')
        espec_ob.plot_vector_Ek(
            savefig=True,
            fname=fname,
            shift_fft=True
        )
        espec_ob.plot_vector_Ek(
            savefig=True,
            fname=fname.replace('_shiftted', ''),
            shift_fft=False
        )


if __name__ == '__main__':
    turb_app = SinVelocityProfile()
    turb_app.run()
    turb_app.dump_enery_spectrum(iter_idx=0)
