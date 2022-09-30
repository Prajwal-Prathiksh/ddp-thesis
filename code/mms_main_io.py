"""
We assume both inlet and outlet is moving with 1m/s speed normal to the
wall
Both inlet and outlet remains planar
"""

import os
import numpy as np
from numpy import pi, cos

from pysph.base.nnps import DomainManager
from pysph.base.utils import get_particle_array
from pysph.base.kernels import QuinticSpline
from pysph.solver.application import Application
from pysph.sph.basic_equations import SummationDensity
from sympy.geometry import parabola
from config_mms import get_props

# domain and constants
L = 1.0
U = 1.0
rho0 = 1.0
c0 = 40
p0 = c0**2 * rho0


class MSMethodIO(Application):
    def add_user_options(self, group):
        group.add_argument(
            "--nx", action="store", type=int, dest="nx", default=50,
            help="Number of points along x direction. (default 50)"
        )
        group.add_argument(
            "--re", action="store", type=float, dest="re", default=100,
            help="Reynolds number (defaults to 100)."
        )

        group.add_argument(
            "--nu", action="store", type=float, dest="nu", default=-1.0,
            help="Kinematic viscosity"
        )
        group.add_argument(
            "--hdx", action="store", type=float, dest="hdx", default=1.2,
            help="Ratio h/dx."
        )
        group.add_argument(
            "--mms", action="store", type=str, dest="mms",
            default='mms1',
            help="mms type 'mms1', 'mms2'..."
        )
        group.add_argument(
            "--perturb", action="store", type=str, dest="perturb",
            default='up',
            help="perturbation type 'up', 'pack' and 'p' "
        )
        group.add_argument(
            "--bctype", action="store", type=str, dest="bctype",
            default='mms',
            help="boundary type 'adami' "
        )
        group.add_argument(
            "--bc", action="store", type=str, dest="bc",
            default='u_slip',
            help="boundary type 'u_slip', 'u_no_slip' and 'p_solid'"
        )
        group.add_argument(
            "--shape", action="store", type=str, dest="shape",
            default='line',
            help="Shape of the boundary 'sin' or 'line'"
        )
        group.add_argument(
            "--dim", action="store", type=int, dest="dim",
            default=2,
            help="Dimension of the domain"
        )
        group.add_argument(
            "--nl", action="store", type=int, dest="nl",
            default=10,
            help="Number of layer of solid particles"
        )


    def consume_user_options(self):
        nx = self.options.nx
        re = self.options.re

        self.c0 = c0
        if self.options.nu < -1e-14:
            self.nu = U * L / re
        else:
            self.nu = self.options.nu

        self.dim = self.options.dim
        self.dx = dx = L / nx
        self.volume = dx**self.dim
        self.hdx = self.options.hdx
        self.nl = self.options.nl

        h0 = self.hdx * self.dx
        if self.options.scheme.endswith('isph'):
            dt_cfl = 0.25 * dx / U
        else:
            dt_cfl = 0.25 * dx / (self.c0 + U)
        if self.nu > 1e-14:
            dt_viscous = 0.125 * dx**2 / self.nu
        else:
            dt_viscous = 10000.0
        dt_force = 0.25 * 1.0

        self.dt = min(dt_cfl, dt_viscous, dt_force)
        self.tf = 2.0
        self.mms = self.options.mms
        self.perturb = self.options.perturb
        self.bctype = self.options.bctype
        self.bc = bc = self.options.bc
        inlet = outlet = 'mms'
        if bc.split('_')[-1] == 'inlet':
            inlet = bc
        else:
            outlet = bc
        self.bcs = {'mms':['wall'], inlet:['inlet'], outlet:['outlet']}
        self.bid_eval = None

    def pre_step(self, solver):
        from config_solid_bc import bc_pre_step
        self.bid_eval = bc_pre_step(self.bid_eval, solver, self.bctype,
                                    self.particles, self.domain)

    def post_step(self, solver):
        from config_mms import config_eq
        # We rewrite the properties with MS as we do not want the BC to get
        # affected by the particle shifiting thus bc='mms'
        self.scheme.scheme.post_step(self.particles,
                                     self.domain,
                                     config_eq,
                                     mms=self.mms,
                                     bc='mms')


    def configure_scheme(self):
        from config_mms import configure_scheme_io
        configure_scheme_io(self, rho0, p0, c0)

    def create_scheme(self):
        from config_mms import create_scheme_io
        return create_scheme_io(self, rho0, p0, c0)

    def create_equations(self):
        from config_mms import create_equations_io
        return create_equations_io(self, rho0, p0)

    def create_particles(self):
        # create the particles
        from particle_arrays import create_particles_io
        from config_io import get_bc_require, set_bc_props
        mirror_inlet, mirror_outlet = get_bc_require(
            self.bctype, self.bc)
        particles = create_particles_io(self, rho0, L, mirror_inlet, mirror_outlet)
        self.scheme.setup_properties(particles, clean=False)
        set_bc_props(self.bctype, particles)

        print("problem :: nfluid = %d, dt = %g" % (
            particles[0].get_number_of_particles(), self.dt))

        props = ['ioid', 'disp']
        for pa in particles:
            pa.add_constant('c0', self.c0)
            for prop in props:
                pa.add_property(prop)

        if self.bctype == 'hybrid':
            pa.v[:] = 0.0
            for pa in particles:
                for i in range(6):
                    pa.pag[i::6] = pa.p.copy()
                    pa.uag[i::6] = pa.u.copy()
                    pa.rag[i::6] = pa.rhoc.copy()
                pa.pta[:] = pa.p.copy()
                pa.uta[:] = pa.u.copy()
                pa.rta[:] = pa.rhoc.copy()
        return particles

    def create_inlet_outlet(self, particle_arrays):
        iom = self.iom
        io = iom.get_inlet_outlet(particle_arrays)
        return io

    # The following are all related to post-processing.
    def _get_post_process_props(self, array):
        """Return x, y, m, u, v, p.
        """

        x, y, z, m, u, v, w, p, au, av, rhoc, rho = array.get(
            'x', 'y', 'z', 'm', 'u', 'v', 'w', 'p',
            'au', 'av', 'rhoc', 'rho'
        )
        return x, y, z, m, u, v, w, p, au, av, rhoc, rho

    def _no_interpolate(self, arrays):
        from pysph.tools.sph_evaluator import SPHEvaluator
        arr = []
        names = []
        for key in arrays:
            arr.append(arrays[key])
            names.append(arrays[key].name)
        eq = SummationDensity(dest='fluid', sources=names)
        eval = SPHEvaluator(arr, equations=[eq], dim=self.dim, kernel=QuinticSpline(dim=self.dim))
        eval.evaluate()
        x, y, z, m, u, v, w, p, au, av, rhoc, rho = self._get_post_process_props(arr[0])
        vol = m/rho
        cond = y > -1.0 # condition to test boundary

        p = self.c0**2*(rhoc - 1.0)

        return x[cond], y[cond], z[cond], u[cond], v[cond], w[cond], p[cond], vol[cond]

    def post_process(self, info_fname):
        self.read_info(info_fname)
        if len(self.output_files) == 0:
            return

        from pysph.solver.utils import iter_output
        from pysph.solver.utils import load
        from pysph.tools.interpolator import Interpolator
        decay_rate = -8.0 * np.pi**2 / self.options.re

        files = self.output_files
        t, ke, ke_ex, decay, linf, l1, l2, p_l1, p_linf, p_l2, lm, am = [], [], [], [], [], [], [], [], [], [], [], []

        for sd, arrays in iter_output(files[0:]):
            _t = sd['t']
            t.append(_t)
            x, y, z, u, v, w, p, vol = self._no_interpolate(arrays)

            u_e, v_e, w_e, rhoc_e, p_e = get_props(x, y, z, _t, self.c0, self.mms)
            vmag2 = u**2 + v**2 + w**2
            vmag = np.sqrt(vmag2)
            vmag2_e = u_e**2 + v_e**2 + w_e**2
            vmag_e = np.sqrt(vmag2_e)

            theoretical_max = 1.0
            linf.append(max(abs(vmag - vmag_e) / theoretical_max))
            l1_err = np.sum(np.abs(vmag - vmag_e)*vol)
            l2_err = np.sqrt(np.sum((vmag - vmag_e)**2*vol))
            avg_vmag_e = np.sum(vmag_e*vol)
            l1.append(l1_err / avg_vmag_e)
            l2.append(l2_err / avg_vmag_e)

            p_e_max = 1.
            p_error = np.sum(np.abs(p - p_e)*vol) / p_e_max
            p_error2 = np.sqrt(np.sum((p - p_e)**2*vol) / p_e_max)
            p_l1.append(p_error)
            p_l2.append(p_error2)
            p_linf.append(max(abs(p-p_e))/p_e_max)

        t, decay, l1, linf, l2, p_l1, p_linf, p_l2, lm, am = list(map(
            np.asarray, (t, decay, l1, linf, l2, p_l1, p_linf, p_l2, lm, am))
        )
        print(linf, p_linf)
        decay_ex = U * np.exp(decay_rate * t)
        fname = os.path.join(self.output_dir, 'results.npz')
        np.savez(
            fname, t=t, decay=decay, linf=linf, l1=l1, p_l1=p_l1,
            decay_ex=decay_ex, lm=lm, am=am, p_linf=p_linf, l2=l2, p_l2=p_l2
        )

        import matplotlib
        matplotlib.use('Agg')

        from matplotlib import pyplot as plt
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

    def customize_output(self):
        self._mayavi_config('''
        b = particle_arrays['fluid']
        b.scalar = 'u'
        ''')

if __name__ == '__main__':
    app = MSMethodIO()
    app.run()
    app.post_process(app.info_filename)
