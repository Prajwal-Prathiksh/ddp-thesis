r"""
Triperiodic Beltrami Flow
Author: K T Prajwal Prathiksh
###
References
----------
    .. [Antuono2020] M. Antuono, “Tri-periodic fully three-dimensional analytic 
    solutions for the Navier-Stokes equations,” J. Fluid Mech., vol. 890, 2020, 
    doi: 10.1017/jfm.2020.126.

    .. [Colagrossi2021] A. Colagrossi, “Smoothed particle hydrodynamics 
    method from a large eddy simulation perspective . Generalization to a 
    quasi-Lagrangian model Smoothed particle hydrodynamics method from a 
    large eddy simulation perspective . Generalization to a 
    quasi-Lagrangian model,” vol. 015102, no. December 2020, 2021,
    doi: 10.1063/5.0034568.
"""

import os
import numpy as np
from numpy import pi, cos

from pysph.base.nnps import DomainManager
from pysph.base.utils import get_particle_array
from pysph.base.kernels import QuinticSpline
from pysph.sph.scheme import SchemeChooser

from tsph_with_pst import TSPHScheme
from tg_config import ext_force_antuono2020
from turbulence_tools import TurbulentFlowApp

# domain and constants
U = 1.0
rho0 = 1.0
c0 = 10 * U
p0 = c0**2 * rho0

class TriperiodicBeltrami(TurbulentFlowApp):
    def add_user_options(self, group):
        group.add_argument(
            "--perturb", action="store", type=float, dest="perturb", default=0,
            help="Random perturbation of initial particles as a fraction "
            "of dx (setting it to zero disables it, the default)."
        )
        group.add_argument(
            "--length", action="store", type=float, dest="length", default=1.0,
            help="Length of the domain (default 1.0)."
        )
        group.add_argument(
            "--nx", action="store", type=int, dest="nx", default=30,
            help="Number of points along x direction. (default 50)"
        )
        group.add_argument(
            "--re", action="store", type=float, dest="re", default=50_000,
            help="Reynolds number (defaults to 100)."
        )
        group.add_argument(
            "--hdx", action="store", type=float, dest="hdx", default=1.0,
            help="Ratio h/dx."
        )
        group.add_argument(
            "--pb-factor", action="store", type=float, dest="pb_factor",
            default=1.0,
            help="Use fraction of the background pressure (default: 1.0)."
        )
        group.add_argument(
            "--c0-fac", action="store", type=float, dest="c0_fac",
            default=10.0,
            help="default factor is 10"
        )
        corrections = ['', 'mixed', 'gradient', 'crksph', 'kgf', 'order1']
        group.add_argument(
            "--kernel-correction", action="store", type=str,
            dest='kernel_correction',
            default='', help="Type of Kernel Correction", choices=corrections
        )
        group.add_argument(
            "--no-periodic", action="store_false",
            dest="no_periodic",
            help="Make periodic domain"
        )

    def consume_user_options(self):
        nx = self.options.nx
        re = self.options.re
        self.L = self.options.length
        self.U = U

        self.c0 = self.options.c0_fac * U
        self.nu = nu = U * self.L / re

        self.dx = dx = self.L / nx
        self.volume = dx ** 3
        self.hdx = self.options.hdx

        h0 = self.hdx * self.dx
        if self.options.scheme.endswith('isph'):
            dt_cfl = 0.25 * dx / U
        else:
            dt_cfl = 0.25 * dx / (self.c0 + U)
        dt_viscous = 0.125 * dx**2 / nu
        dt_force = 0.25 * 1.0

        self.dt = min(dt_cfl, dt_viscous, dt_force)
        self.tf = 2.0
        self.kernel_correction = self.options.kernel_correction
        self.no_periodic = self.options.no_periodic

    def pre_step(self, solver):
        pa = self.particles[0]
        x, y, z = pa.x, pa.y, pa.z
        L, U = self.L, self.U
        nu = self.nu

        dt, t = solver.dt, solver.t

        fx, fy, fz = ext_force_antuono2020(
            x=x, y=y, z=z, t=t, nu=nu, L=L, U=U
        )
        pa.u[:] += fx * dt
        pa.v[:] += fy * dt
        pa.w[:] += fz * dt
    
    def post_step(self, solver):
        if self.options.scheme == 'tsph' or self.options.scheme == 'tisph':
            self.scheme.scheme.post_step(self.particles, self.domain)
    
    def configure_scheme(self):
        h0 = self.hdx * self.dx
        pfreq = 100000000
        kernel = QuinticSpline(dim=3)

        if self.options.scheme == 'tsph':
            self.scheme.configure(
                hdx=self.hdx, nu=self.nu, h0=h0, gx=0.0,
                periodic=self.no_periodic
            )
        
        tf = self.options.final_time
        if tf is None:
            tf = self.tf
        times = np.linspace(0, tf, 100)
        self.scheme.configure_solver(
            kernel=kernel, tf=self.tf, dt=self.dt, output_at_times=times, pfreq=pfreq
        )
    
    def create_scheme(self):
        h0 = None
        hdx = None
        nu = None
        tsph = TSPHScheme(
            ['fluid'], [], dim=3, rho0=rho0, c0=c0, h0=h0,
            hdx=hdx, nu=nu, gamma=7.0, kernel_corr=True,
            method='sd', scm='wcsph', pst_freq=10
        )
        s = SchemeChooser(default='tsph', tsph=tsph)
        return s
        
    def create_domain(self):
        if self.options.no_periodic:
            return DomainManager(
                xmin=0, xmax=self.L,
                ymin=0, ymax=self.L,
                zmin=0, zmax=self.L,
                periodic_in_x=True, periodic_in_y=True, periodic_in_z=True
            )
    
    def create_particles(self):
        # Create particles
        dx = self.dx
        _x = np.arange(dx/2, self.L, dx)
        x, y, z = np.meshgrid(_x, _x, _x)

        # Initialize
        V0 = self.volume
        m = V0 * rho0

        h = self.hdx * dx
        re = self.options.re
        u0, v0, w0 = 0.0, 0.0, 0.0
        p0 = 1.
        
        color0 = cos(2*pi*x) * cos(4*pi*y) * cos(6*pi*z)
        rhoc = 0.0
        rho = rho0

        if self.options.eos == 'linear':
            print('linear')
            rhoc = (p0 / self.c0**2 + 1)
        elif self.options.eos == 'tait':
            print('tait')
            rhoc = (p0 * 7.0 / self.c0**2 + 1)**(1./7.0)

        # create the arrays
        fluid = get_particle_array(
            name='fluid', x=x, y=y, z=z, m=m, h=h, u=u0, v=v0, rho=rho, rhoc=rhoc, 
            p=p0, color=color0, V0=V0
        )
        self.scheme.setup_properties([fluid])
        nfp = fluid.get_number_of_particles()

        print(f"Taylor green vortex problem :: nfluid = {nfp}, dt = {self.dt}")
        
        fluid.add_output_arrays(['au', 'av', 'aw'])

        fluid.gid[:] = np.arange(nfp)
        fluid.add_constant('c0', self.c0)
        # fluid.add_property('rhoc')

        return [fluid]

    # The following are all related to post-processing.
    def _get_post_process_props(self, array):
        x, y, m, u, v, p, au, av, rhoc, rho = array.get(
            'x', 'y', 'm', 'u', 'v', 'p',
            'au', 'av', 'rhoc', 'rho'
        )
        return x, y, m, u, v, p, au, av, rhoc, rho

    def _add_extra_props(self, array):
        extra = ['pavg', 'nnbr']
        for prop in extra:
            if prop not in array.properties:
                array.add_property(prop)
        array.add_output_arrays(extra)

    def _get_sph_evaluator(self, array):
        if not hasattr(self, '_sph_eval'):
            from pysph.sph.wc.edac import ComputeAveragePressure
            from pysph.tools.sph_evaluator import SPHEvaluator
            equations = [
                ComputeAveragePressure(dest='fluid', sources=['fluid'])
            ]
            dm = self.create_domain()
            sph_eval = SPHEvaluator(
                arrays=[array], equations=equations, dim=2,
                kernel=QuinticSpline(dim=2), domain_manager=dm
            )
            self._sph_eval = sph_eval
        return self._sph_eval

    def post_process(self, info_fname):
        self.read_info(info_fname)
        if len(self.output_files) == 0:
            return

        from pysph.solver.utils import iter_output
        from pysph.solver.utils import load
        from pysph.tools.interpolator import Interpolator

        files = self.output_files
        t, ke, decay, lm = [], [], [], []

        for sd, array in iter_output(files[0:], 'fluid'):
            _t = sd['t']
            t.append(_t)
            x, y, m, u, v, p, au, av, rhoc, rho =\
                self._get_post_process_props(array)
            if self.options.scheme == 'tsph' or self.options.scheme == 'tdsph':
                if self.options.scm == 'wcsph' or self.options.scm == 'fatehi':
                    if self.options.eos == 'linear':
                        p = self.c0**2*(rhoc - 1)
                    elif self.options.eos == 'tait':
                        p = self.c0**2/7*(rhoc**7 - 1)


            vmag2 = u**2 + v**2
            vmag = np.sqrt(vmag2)
            ke.append(0.5 * np.sum(m * vmag2))
            lm.append(sum(m*au))

            vmag_max = vmag.max()
            decay.append(vmag_max)

        # Convert to numpy arrays
        t, ke, decay, lm = list(map(np.asarray, (t, ke, decay, lm)))

        fname = os.path.join(self.output_dir, 'results.npz')
        np.savez(
            fname, t=t, ke=ke, decay=decay, lm=lm
        )

        import matplotlib
        matplotlib.use('Agg')

        from matplotlib import pyplot as plt
        plt.clf()
        plt.grid()
        plt.semilogy(t, decay, label="computed")
        plt.xlabel('t')
        plt.ylabel('max velocity')
        plt.title(f'Re={self.options.re}, U={self.U}')
        plt.legend()
        fig = os.path.join(self.output_dir, "decay.png")
        plt.savefig(fig, dpi=300)

        plt.clf()
        plt.grid()
        plt.plot(t, ke, label="computed")
        plt.xlabel('t')
        plt.ylabel('kinetic energy')
        plt.title(f'Re={self.options.re}, U={self.U}')
        plt.legend()
        fig = os.path.join(self.output_dir, "ke.png")
        plt.savefig(fig, dpi=300)

        plt.clf()
        plt.plot(t, lm, label="total linear mom")
        plt.grid()
        plt.xlabel('t')
        plt.ylabel(r'Total linear mom')
        plt.title(f'Re={self.options.re}, U={self.U}')
        fig = os.path.join(self.output_dir, "mom.png")
        plt.savefig(fig, dpi=300)

    def customize_output(self):
        self._mayavi_config('''
        b = particle_arrays['fluid']
        b.scalar = 'vmag'
        ''')

if __name__ == '__main__':
    app = TriperiodicBeltrami()
    app.run()
    app.post_process(app.info_filename)
    app.ek_post_processing(
        dim=3, L=app.L, U0=U, f_idx=-1,
        compute_without_interp=True
    )
    # app.plot_ek(f_idx=-1)
    app.plot_ek_fit(f_idx=-1, tol=1e-20, k_n=4)
