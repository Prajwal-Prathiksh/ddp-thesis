"""Taylor Green vortex flow (5 minutes).
"""

import os
import numpy as np
from numpy import pi, cos

from pysph.base.nnps import DomainManager
from pysph.base.utils import get_particle_array
from pysph.base.kernels import QuinticSpline
from pysph.solver.application import Application

from tg_config import exact_solution

# domain and constants
L = 1.0
U = 1.0
rho0 = 1.0
c0 = 10 * U
p0 = c0**2 * rho0


class TaylorGreen(Application):
    def add_user_options(self, group):
        group.add_argument(
            "--init", action="store", type=str, default=None,
            help="Initialize particle positions from given file."
        )
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
            "--re", action="store", type=float, dest="re", default=100,
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
            "--remesh", action="store", type=int, dest="remesh", default=0,
            help="Remeshing frequency (setting it to zero disables it)."
        )
        remesh_types = ['m4', 'sph']
        group.add_argument(
            "--remesh-eq", action="store", type=str, dest="remesh_eq",
            default='m4', choices=remesh_types,
            help="Remeshing strategy to use."
        )
        group.add_argument(
            "--shift-freq", action="store", type=int, dest="shift_freq",
            default=0,
            help="Particle position shift frequency.(set zero to disable)."
        )
        shift_types = ['simple', 'fickian', 'ipst', 'mod_fickian', 'delta_plus']
        group.add_argument(
            "--shift-kind", action="store", type=str, dest="shift_kind",
            default='simple', choices=shift_types,
            help="Use of fickian shift in positions."
        )
        group.add_argument(
            "--shift-parameter", action="store", type=float,
            dest="shift_parameter", default=None,
            help="Constant used in shift, range for 'simple' is 0.01-0.1"
            "range 'fickian' is 1-10."
        )
        group.add_argument(
            "--shift-correct-vel", action="store_true",
            dest="correct_vel", default=False,
            help="Correct velocities after shifting (defaults to false)."
        )
        group.add_argument(
            "--no-periodic", action="store_false",
            dest="no_periodic",
            help="Make periodic domain"
        )


    def consume_user_options(self):
        nx = self.options.nx
        re = self.options.re

        self.c0 = self.options.c0_fac * U
        self.nu = nu = U * L / re

        self.dx = dx = L / nx
        self.volume = dx * dx
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
        from tg_config import prestep
        prestep(self, solver)

    def post_step(self, solver):
        if self.options.scheme == 'tsph' or self.options.scheme == 'tisph':
            if self.options.remesh == 0:
                self.scheme.scheme.post_step(self.particles, self.domain)

    def configure_scheme(self):
        from tg_config import configure_scheme
        configure_scheme(self, p0)

    def create_scheme(self):
        from tg_config import create_scheme
        return create_scheme(rho0, c0, p0)

    def create_equations(self):
        from tg_config import create_equation
        return create_equation(self)

    def create_domain(self):
        if self.options.no_periodic:
            return DomainManager(
                xmin=0, xmax=L, ymin=0, ymax=L, periodic_in_x=True,
                periodic_in_y=True
            )

    def create_particles(self):
        # create the particles
        dx = self.dx
        x, y, h = None, None, None
        filename = '%d_tg.npz'%self.options.nx
        # filename = '%s_%d_%.2f.npz'%(self.options.kernel, self.options.nx, self.hdx)
        dirname = os.path.dirname(os.path.abspath(__file__))
        print(dirname)
        datafile = os.path.join(os.path.dirname(dirname), 'data', filename)
        print(datafile)
        if os.path.exists(datafile) and (not(self.options.scheme=='ewcsph' or self.options.scheme=='rsph')):
            data = np.load(datafile)
            x = data['x']
            y = data['y']
            # h = data['h']
            print(x, y)
        else:
            _x = np.arange(dx / 2, L, dx)
            x, y = np.meshgrid(_x, _x)

        # Initialize
        m = self.volume * rho0
        V0 = self.volume
        if h is None:
            h = self.hdx * dx
        re = self.options.re
        b = -8.0*pi*pi / re
        u0, v0, p0 = exact_solution(U=U, b=b, t=0, x=x, y=y)
        color0 = cos(2*pi*x) * cos(4*pi*y)
        rhoc = 0.0
        rho = rho0
        if self.options.eos == 'linear':
            print('linear')
            rhoc = (p0 / self.c0**2 + 1)
        elif self.options.eos == 'tait':
            print('tait')
            rhoc = (p0 * 7.0 / self.c0**2 + 1)**(1./7.0)

        if self.options.scheme == 'ewcsph':
            print('ewcsph')
            rhoc = (p0 / self.c0**2 + 1)
            # rho = (p0 / self.c0**2 + 1)
            rho = (p0 * 7.0 / self.c0**2 + 1)**(1./7.0)
            # rhoc = (p0 * 7.0 / self.c0**2 + 1)**(1./7.0)
        # create the arrays
        fluid = get_particle_array(name='fluid', x=x, y=y, m=m, h=h, u=u0,
                                   v=v0, rho=rho, rhoc=rhoc, p=p0, color=color0, V0=V0)

        self.scheme.setup_properties([fluid])

        print("Taylor green vortex problem :: nfluid = %d, dt = %g" % (
            fluid.get_number_of_particles(), self.dt))

        from tg_config import configure_particles
        configure_particles(self, fluid)

        nfp = fluid.get_number_of_particles()
        fluid.gid[:] = np.arange(nfp)
        fluid.add_constant('c0', self.c0)
        fluid.add_property('rhoc')

        return [fluid]

    def create_tools(self):
        from tg_config import create_tools
        return create_tools(self)

    # The following are all related to post-processing.
    def _get_post_process_props(self, array):
        """Return x, y, m, u, v, p.
        """
        x, y, m, u, v, p, au, av, T, rhoc, rho = array.get(
            'x', 'y', 'm', 'u', 'v', 'p',
            'au', 'av', 'T', 'rhoc', 'rho'
        )
        return x, y, m, u, v, p, au, av, T, rhoc, rho

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
        decay_rate = -8.0 * np.pi**2 / self.options.re

        files = self.output_files
        t, ke, ke_ex, decay, linf, l1, p_l1, lm, am = [], [], [], [], [], [], [], [], []

        for sd, array in iter_output(files[0:], 'fluid'):
            _t = sd['t']
            t.append(_t)
            x, y, m, u, v, p, au, av, T, rhoc, rho = self._get_post_process_props(array)
            if self.options.scheme == 'tsph' or self.options.scheme == 'tdsph':
                if self.options.scm == 'wcsph' or self.options.scm == 'fatehi':
                    if self.options.eos == 'linear':
                        p = self.c0**2*(rhoc - 1)
                    elif self.options.eos == 'tait':
                        p = self.c0**2/7*(rhoc**7 - 1)
            if self.options.scheme == 'dpsph':
                if self.options.eos == 'linear':
                    p = self.c0**2*(rhoc - 1)
                elif self.options.eos == 'tait':
                    p = self.c0**2/7*(rhoc**7 - 1)

            if self.options.scheme == 'ewcsph':
                if self.options.method == 'soc':
                    p = self.c0**2*(rhoc - 1)
                    # p = self.c0**2/7*(rhoc**7 - 1)
                else:
                    # p = self.c0**2*(rho - 1)
                    p = self.c0**2/7*(rho**7 - 1)
            u_e, v_e, p_e = exact_solution(U, decay_rate, _t, x, y)
            vmag2 = u**2 + v**2
            vmag = np.sqrt(vmag2)
            ke.append(0.5 * np.sum(m * vmag2))
            vmag2_e = u_e**2 + v_e**2
            vmag_e = np.sqrt(vmag2_e)
            ke_ex.append(0.5 * np.sum(m * vmag2_e))
            lin_mom = sum(m*au)
            ang_mom = sum(T)
            lm.append(lin_mom)
            am.append(ang_mom)

            vmag_max = vmag.max()
            decay.append(vmag_max)
            theoretical_max = U * np.exp(decay_rate * _t)
            linf.append(abs((vmag_max - theoretical_max) / theoretical_max))

            l1_err = np.average(np.abs(vmag - vmag_e))
            avg_vmag_e = np.average(np.abs(vmag_e))
            # scale the error by the maximum velocity.
            l1.append(l1_err / avg_vmag_e)

            p_e_max = np.abs(p_e).max()
            p_error = np.average(np.abs(p - p_e)) / p_e_max
            p_l1.append(p_error)

        t, ke, ke_ex, decay, l1, linf, p_l1, lm, am = list(map(
            np.asarray, (t, ke, ke_ex, decay, l1, linf, p_l1, lm, am))
        )
        print(l1, p_l1)
        decay_ex = U * np.exp(decay_rate * t)
        fname = os.path.join(self.output_dir, 'results.npz')
        np.savez(
            fname, t=t, ke=ke, ke_ex=ke_ex, decay=decay, linf=linf, l1=l1,
            p_l1=p_l1, decay_ex=decay_ex, lm=lm, am=am
        )

        import matplotlib
        matplotlib.use('Agg')

        from matplotlib import pyplot as plt
        plt.clf()
        plt.grid()
        plt.semilogy(t, decay_ex, label="exact")
        plt.semilogy(t, decay, label="computed")
        plt.xlabel('t')
        plt.ylabel('max velocity')
        plt.legend()
        fig = os.path.join(self.output_dir, "decay.png")
        plt.savefig(fig, dpi=300)

        plt.clf()
        plt.plot(t, linf)
        plt.grid()
        plt.xlabel('t')
        plt.ylabel(r'$L_\infty$ error')
        fig = os.path.join(self.output_dir, "linf_error.png")
        plt.savefig(fig, dpi=300)

        plt.clf()
        plt.plot(t, l1, label="error")
        plt.grid()
        plt.xlabel('t')
        plt.ylabel(r'$L_1$ error')
        fig = os.path.join(self.output_dir, "l1_error.png")
        plt.savefig(fig, dpi=300)

        plt.clf()
        plt.plot(t, p_l1, label="error")
        plt.grid()
        plt.xlabel('t')
        plt.ylabel(r'$L_1$ error for $p$')
        fig = os.path.join(self.output_dir, "p_l1_error.png")
        plt.savefig(fig, dpi=300)

        plt.clf()
        plt.plot(t, lm, label="total linear mom")
        plt.grid()
        plt.xlabel('t')
        plt.ylabel(r'Total linear mom')
        fig = os.path.join(self.output_dir, "mom.png")
        plt.savefig(fig, dpi=300)

        plt.clf()
        plt.plot(t, am, label="total angular mom")
        plt.grid()
        plt.xlabel('t')
        plt.ylabel(r'Total angular mom')
        fig = os.path.join(self.output_dir, "ang_mom.png")
        plt.savefig(fig, dpi=300)

    def customize_output(self):
        self._mayavi_config('''
        b = particle_arrays['fluid']
        b.scalar = 'vmag'
        ''')


if __name__ == '__main__':
    app = TaylorGreen()
    app.run()
    app.post_process(app.info_filename)
