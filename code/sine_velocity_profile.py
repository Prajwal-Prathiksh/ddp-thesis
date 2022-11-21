r"""
A Sinusoidal Velocity Profile
##############################
"""
# Library imports
import os
import numpy as np
from pysph.base.nnps import DomainManager
from pysph.base.utils import get_particle_array
from pysph.sph.integrator import Integrator
from pysph.base.kernels import CubicSpline
from pysph.solver.solver import Solver

# Local imports
from energy_spectrum import EnergySpectrum
from turbulence_tools import TurbulentFlowApp, get_kernel_cls

# TODO: Add a more robust/realistic test case?


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
    """
    A sinusoidal velocity profile problem.
    """

    def one_timestep(self):
        """
        Do nothing.
        """
        pass


class SinVelocityProfile(TurbulentFlowApp):
    """
    Particles having a sinusoidal velocity profile.
    """

    def add_user_options(self, group):
        """
        Add user options to the given group.
        """
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

    def consume_user_options(self):
        """
        Store user options as variables.
        """
        self.perturb = self.options.perturb
        self.nx = self.options.nx
        self.hdx = self.options.hdx
        self.dim = self.options.dim

        i_nx = self.options.i_nx
        self.i_nx = self.options.i_nx
        self.i_kernel = self.options.i_kernel
        self.i_kernel_cls = get_kernel_cls(self.i_kernel, self.dim)
        self.i_method = self.options.i_method

        self.dx = dx = 1. / self.nx
        self.volume = dx**self.dim

        self.L = 1.
        self.rho0 = 1.

    def create_domain(self):
        """
        Create the domain.
        """
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
        """
        Create the particles.
        """
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
        pa.add_property('m_mat', stride=9)
        pa.set_output_arrays(
            [
                'x', 'y', 'z', 'u', 'v', 'w', 'vmag',
                'rho', 'm', 'h', 'pid', 'gid', 'tag'
            ]
        )

        print("Created %d particles" % pa.get_number_of_particles())
        return [pa]

    def create_solver(self):
        """
        Create the solver.
        """
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
        return solver

    def create_equations(self):
        """
        Do nothing.
        """
        return []

    # The following are all related to post-processing.
    def get_exact_energy_spectrum(self):
        """
        Get the exact energy spectrum.
        """
        dim = self.dim

        N = int(1 + np.ceil(np.sqrt(dim * self.i_nx**2) / 2))
        Ek_exact = np.zeros(N, dtype=np.float64)

        if dim == 1:
            Ek_exact[1] = 0.125
        elif dim == 2:
            Ek_exact[1] = 0.125
        elif dim == 3:
            Ek_exact[2] = 0.09375

        return Ek_exact

    def post_process(self, info_fname: str):
        """
        Post-process the data.
        This method is called after the simulation is complete.

        Parameters:
        -----------
        info_fname : str
            The name of the info file.
        """
        info = self.read_info(info_fname)

        dim = self.dim
        if len(self.output_files) == 0:
            return

        method = self.options.i_method
        if method not in ['sph', 'shepard', 'order1']:
            method = 'order1'
        espec_ob = EnergySpectrum.from_pysph_file(
            fname=self.output_files[0],
            dim=dim,
            L=self.L,
            i_nx=self.i_nx,
            kernel=self.i_kernel_cls,
            domain_manager=self.create_domain(),
            method=method,
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
    turb_app.dump_enery_spectrum(dim=turb_app.dim, L=turb_app.L, iter_idx=0)
