r"""    
A Sinusoidal Velocity Profile
##############################
"""

r"""
Taylor Green vortex flow (5 minutes).
######################################
"""

import os
import numpy as np
from numpy import pi, sin, cos, exp

from pysph.base.nnps import DomainManager
from pysph.base.utils import get_particle_array
from pysph.base.kernels import QuinticSpline
from pysph.solver.application import Application

from pysph.sph.equation import Group, Equation


def perturb_signal(perturb_fac:float, *args:np.ndarray):
    """
    Perturb the given signal/s by a uniform random number between [0, 1) scaled by the given factor.
    The random number is seeded by numpy.random.seed(1) to ensure reproducibility.

    Parameters
    ----------
    perturb_fac : float
        Factor by which the signal is to be perturbed by a uniform random number.
    *args : np.ndarray
        Signals to be perturbed.

    Returns
    -------
    perturbed_signals : list
        Perturbed signals.
    """
    if perturb_fac > 0.:
        np.random.seed(1)
        return [arg + perturb_fac * np.random.random(arg.shape) for arg in args]
    else:
        return args

class SinVelocityProfile(Application):

    def add_user_options(self, group):
        group.add_argument(
            "--perturb", action="store", type=float, dest="perturb", default=0,
            help="Random perturbation of initial particles as a fraction "
            "of dx (setting it to zero disables it, the default)."
        )
        group.add_argument(
            "--nx", action="store", type=int, dest="nx", default=50,
            help="Number of points along x direction. (default 50)"
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
            "--nx-i", action="store", type=int, dest="nx_i", default=50,
            help="Number of interpolation points along x direction. (default 50)."
        )

    def consume_user_options(self):
        self.perturb = self.options.perturb
        self.nx = self.options.nx
        self.hdx = self.options.hdx
        self.dim = self.options.dim
        self.nx_i = self.options.nx_i
        
        self.dx = dx = 1. / self.nx
        self.volume = dx**self.dim

        self.L = 1.
        self.rho0 = 1.

    def create_domain(self):
        return DomainManager(
            xmin=0, xmax=self.L,
            ymin=0, ymax=self.L,
            zmin=0, zmax=self.L,
            periodic_in_x=True,
            periodic_in_y=True,
            periodic_in_z=True
        )

    def create_particles(self):
        # Create the particles
        dx = self.dx

        _x = np.arange(dx/2, self.L, dx)
        twopi = 2 * np.pi
        cos, sin = np.cos, np.sin
        if self.dim == 1:
            x = perturb_signal(self.perturb, _x)[0]
            u0 = -cos(twopi * x)
            v0 = w0 = 0.
        elif self.dim == 2:
            x, y = np.meshgrid(_x, _x)
            x, y = perturb_signal(self.perturb, x, y)
            u0 = cos(twopi * x) * sin(twopi * y) * - 1.
            v0 = sin(twopi * x) * cos(twopi * y)
            w0 = 0.
        elif self.dim == 3:
            x, y, z = np.meshgrid(_x, _x, _x)
            x, y, z = perturb_signal(self.perturb, x, y, z)
            u0 = cos(twopi * x) * sin(twopi * y) * sin(twopi * z) * - 1.
            v0 = sin(twopi * x) * cos(twopi * y) * sin(twopi * z)
            w0 = sin(twopi * x) * sin(twopi * y) * cos(twopi * z)
        else:
            raise ValueError("Dimension should be 1, 2 or 3.")

        # Initialize
        m = self.volume * self.rho0
        h = self.hdx * dx

        # Create the arrays
        pa = get_particle_array(
            name='particles', x=x, y=y, m=m, h=h,
            u=u0, v=v0, w=w0, rho=self.rho0
        )
        print("Created %d particles" % pa.get_number_of_particles())
        return [pa]

    # The following are all related to post-processing.
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
            nx_i=self.nx_i,
            kernel=None,
            domain_manager=self.create_domain(),
            U0=1.
        )

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
        espec_ob.plot_EK(
            savefig=True,
            fname=fname,
            shift_fft=True
        )
        espec_ob.plot_EK(
            savefig=True,
            fname=fname.replace('_shiftted', ''),
            shift_fft=False
        )

        # Sane npz file
        fname = os.path.join(self.output_dir, 'results.npz')
        np.savez(
            fname,
            t=espec_ob.t,
            Ek=espec_ob.Ek,
            EK_U=espec_ob.EK_U,
            EK_V=espec_ob.EK_V,
            EK_W=espec_ob.EK_W,
        )

    def customize_output(self):
        self._mayavi_config('''
        b = particle_arrays['fluid']
        b.scalar = 'vmag'
        ''')


if __name__ == '__main__':
    app = SinVelocityProfile()
    app.run()
    app.post_process(app.info_filename)