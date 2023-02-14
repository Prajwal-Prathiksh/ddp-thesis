r"""
A Sinusoidal Velocity Profile
Author: K T Prajwal Prathiksh
###
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
    return args


def get_flow_field(
    dim: int, dx: float, L: float, perturb_fac: float = 0,
    n_freq: int = 1, decay_rate: float = 2.
):
    r"""
    Get the flow field for the given domain parameters.
    The velocity field is a sinusoidal velocity profile with a given number of
    frequencies (N) and a decay rate (gamma) for the amplitudes of the
    frequencies in the velocity field.

    Note: A warning is raised if the number of frequencies is greater than half
    the number of grid points in the domain (N > L / (2 * dx)) - Nyquist
    criterion.

    1D:
    .. math::
        u(x) = - \sum_{i=1}^{N} i^{-\gamma} \cos(2\pi i x)

    2D:
    .. math::
        u(x, y) = - \sum_{i=1}^{N} i^{-\gamma} \cos(2\pi i x) \sin(2\pi i y)
        v(x, y) = \sum_{i=1}^{N} i^{-\gamma} \sin(2\pi i x) \cos(2\pi i y)

    3D:
    .. math::
        u(x, y, z) = - \sum_{i=1}^{N} i^{-\gamma} \cos(2\pi i x) \sin(2\pi i y)
            \sin(2\pi i z)
        v(x, y, z) = \sum_{i=1}^{N} i^{-\gamma} \sin(2\pi i x) \cos(2\pi i y)
            \sin(2\pi i z)
        w(x, y, z) = - \sum_{i=1}^{N} i^{-\gamma} \sin(2\pi i x) \sin(2\pi i y)
            \cos(2\pi i z)

    where,
    .. math::
        x, y, z \in range\{dx/2, L, dx\}
        N \in \mathbb{Z^+}
        \gamma \in \mathbb{R^+}

    Parameters
    ----------
    dim : int
        Dimension of the flow field.
    dx : float
        Grid spacing.
    L : float
        Domain length.
    perturb_fac : float, optional
        Factor by which the signal is to be perturbed by a uniform random
        number. The default is 0.
    n_freq : int, optional
        Number of frequencies in the velocity field. Default is 1.
    decay_rate : float, optional
        Decay rate of the amplitudes of the frequencies in the velocity field.
        Default is 2.

    Returns
    -------
    (x, y, z, u, v, w) : tuple(np.ndarray)
    """
    _x = np.arange(dx / 2, L, dx)

    if n_freq >= len(_x) / 2:
        raise Warning(
            'Number of frequencies is greater than half the number of grid '
            'points in the domain (N >= L / (2 * dx)).'
        )

    decay_rate = np.abs(decay_rate)
    twopi = 2 * np.pi
    cos, sin = np.cos, np.sin
    npsum = np.sum

    freq_range = range(1, n_freq + 1)

    if dim == 1:
        x = perturb_signal(perturb_fac, _x)[0]
        y = z = 0.
        u0 = -npsum(
            [i ** -decay_rate * cos(twopi * i * x) for i in freq_range],
            axis=0
        )
        v0 = w0 = 0.
    elif dim == 2:
        x, y = np.meshgrid(_x, _x)
        x, y = perturb_signal(perturb_fac, x, y)
        z = 0.
        u0 = -npsum(
            [i ** -decay_rate * cos(twopi * i * x) * sin(twopi * i * y)
             for i in freq_range],
            axis=0
        )
        v0 = npsum(
            [i ** -decay_rate * sin(twopi * i * x) * cos(twopi * i * y)
             for i in freq_range],
            axis=0
        )
        w0 = 0.
    elif dim == 3:
        x, y, z = np.meshgrid(_x, _x, _x)
        x, y, z = perturb_signal(perturb_fac, x, y, z)
        u0 = -npsum(
            [i ** -decay_rate * cos(twopi * i * x) * sin(twopi * i * y) *
             sin(twopi * i * z) for i in freq_range],
            axis=0
        )
        v0 = npsum(
            [i ** -decay_rate * sin(twopi * i * x) * cos(twopi * i * y) *
             sin(twopi * i * z) for i in freq_range],
            axis=0
        )
        w0 = -npsum(
            [i ** -decay_rate * sin(twopi * i * x) * sin(twopi * i * y) *
             cos(twopi * i * z) for i in freq_range],
            axis=0
        )
    else:
        raise ValueError("Dimension should be 1, 2 or 3.")
    return x, y, z, u0, v0, w0


class DummyIntegrator(Integrator):
    """
    A sinusoidal velocity profile problem.
    """

    def one_timestep(self, t, dt):
        """
        Do nothing.
        """
        return None


class SinVelocityProfile(TurbulentFlowApp):
    """
    Particles having a sinusoidal velocity profile.
    """

    def __init__(self, *args, **kw):
        """
        Initialize the problem.
        """
        super().__init__(*args, **kw)
        self.perturb = None
        self.nx = None
        self.hdx = None
        self.dim = None

        self.i_nx = None
        self.i_kernel = None
        self.i_kernel_cls = None
        self.i_method = None

        self.dx = None
        self.volume = None
        self.L = None
        self.rho0 = None

    def add_user_options(self, group):
        """
        Add user options to the given group.
        """
        group.add_argument(
            "--perturb", action="store", type=float, dest="perturb",
            default=0.,
            help="Random perturbation of initial particles as a fraction "
            "of dx (setting it to zero disables it, the default)."
        )
        group.add_argument(
            "--nx", action="store", type=int, dest="nx", default=101,
            help="Number of points along x direction."
        )
        group.add_argument(
            "--hdx", action="store", type=float, dest="hdx", default=1.2,
            help="Ratio h/dx."
        )
        group.add_argument(
            "--dim", action="store", type=int, dest="dim", default=2,
            help="Dimension of the problem."
        )
        group.add_argument(
            "--n-freq", action="store", type=int, dest="n_freq", default=None,
            help="Number of frequencies in the velocity field. Default is "
            "half the number of nx."
        )
        group.add_argument(
            "--decay-rate", action="store", type=float, dest="decay_rate",
            default=1.,
            help="Decay rate of the amplitude of the frequencies in the "
            "sinusoidal velocity profile. Therefore, the decay rate of the "
            "energy spectrum should be (decay_rate*2)."
        )
        group.add_argument(
            "--make-plots", action="store_true", dest="make_plots",
            default=False, help="Make plots."
        )

    def consume_user_options(self):
        """
        Store user options as variables.
        """
        self.perturb = self.options.perturb
        self.nx = self.options.nx
        self.hdx = self.options.hdx
        self.dim = self.options.dim
        self.n_freq = self.options.n_freq
        if self.n_freq is None:
            self.n_freq = self.nx // 2
        self.decay_rate = abs(self.options.decay_rate)

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
        x, y, z, u0, v0, w0 = get_flow_field(
            dim=self.dim, dx=dx, L=self.L, perturb_fac=self.perturb,
            n_freq=self.n_freq, decay_rate=self.decay_rate
        )
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

        print('-'*40)
        print(f'Dimension: {self.dim}')
        print(f'Perturbation: {self.perturb}')
        print(f"Number of frequencies: {self.n_freq}")
        print(f"Decay rate: {self.decay_rate}")
        print(f"Created {pa.get_number_of_particles()} particles.")
        print('-'*40)

        # Save un-perturbed velocity field for comparison
        x, y, z, u0, v0, w0 = get_flow_field(
            dim=self.dim, dx=dx, L=self.L, perturb_fac=0,
            n_freq=self.n_freq, decay_rate=self.decay_rate
        )
        self.save_initial_vel_field(
            dim=self.dim, x=x, y=y, z=z, m=m, h=h, u=u0, v=v0, w=w0, L=self.L,
            dx=dx
        )

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
    def get_exact_ek(self):
        """
        Get the exact energy spectrum.

        Returns
        -------
        ek : array
            The exact energy spectrum.
        """
        dim = self.dim

        N = self.get_length_of_ek(dim=dim)
        k_i = np.arange(1, self.n_freq + 1, dtype=np.float64)
        if dim < 3:
            ek = k_i**(-2 * self.decay_rate) / 8
        else:
            ek = 3 * k_i**(-2 * self.decay_rate) / 32

        ek = np.insert(ek, 0, 0)
        if len(ek) < N:
            ek = np.append(ek, np.zeros(N - len(ek)))
        else:
            ek = ek[:N]

        return ek

    def get_expected_ek_slope(self):
        """
        Get the slope of the energy spectrum.

        Returns
        -------
        slope : float
        """
        return -2 * self.decay_rate

    def post_process(self, info_fname: str):
        """
        Post-process the data.
        This method is called after the simulation is complete.

        Parameters:
        -----------
        info_fname : str
            The name of the info file.
        """
        _ = self.read_info(info_fname)

        dim = self.dim
        if len(self.output_files) == 0:
            return

        # Turbulence specific post-processing
        self.compute_interpolated_vel_field(f_idx_list=[0], dim=dim, L=self.L)
        self.compute_ek(f_idx_list=[0], dim=dim, L=self.L, U0=1.)
        self.compute_ek_from_initial_vel_field(dim=dim, L=self.L, U0=1.)

        if not self.options.make_plots:
            return

        # Make plots
        self.plot_ek(
            f_idx=0, plot_type='loglog', exact=True,
            wo_interp_initial=True,
        )
        # self.plot_ek_fit(
        #     f_idx=0, plot_type='loglog', tol=1e-8,
        #     exact=True, no_interp=True
        # )

        method = self.options.i_method
        if method not in ['sph', 'shepard', 'order1']:
            method = 'order1'
        espec_ob = EnergySpectrum.from_pysph_ofile(
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
        k_fit, ek_fit, fit_params = espec_ob.get_ek_fit()
        fname = os.path.join(self.output_dir, 'energy_spectrum_log.png')
        espec_ob.plot_scalar_ek(
            savefig=True,
            fname=fname,
            plot_type='log'
        )
        fname = os.path.join(self.output_dir, 'EK_spectrum.png')
        espec_ob.plot_vector_ek(
            savefig=True,
            fname=fname,
            shift_fft=False
        )


if __name__ == '__main__':
    turb_app = SinVelocityProfile()
    turb_app.run()
    turb_app.post_process(turb_app.info_filename)
