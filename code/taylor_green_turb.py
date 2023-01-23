r"""
Taylor Green vortex flow (5 minutes).
######################################
"""
# Library imports
import os
import numpy as np
from numpy import pi, sin, cos, exp

from pysph.base.nnps import DomainManager
from pysph.base.utils import get_particle_array
from pysph.base.kernels import QuinticSpline
from pysph.solver.application import Application

from pysph.sph.equation import Group, Equation
from pysph.sph.scheme import TVFScheme, WCSPHScheme, SchemeChooser
from pysph.sph.wc.edac import ComputeAveragePressure, EDACScheme
from pysph.sph.iisph import IISPHScheme

from pysph.sph.wc.kernel_correction import (
    GradientCorrectionPreStep, GradientCorrection,
    MixedKernelCorrectionPreStep, MixedGradientCorrection
)
from pysph.sph.wc.crksph import CRKSPHPreStep, CRKSPH, CRKSPHScheme
from pysph.sph.wc.gtvf import GTVFScheme
from pysph.sph.wc.pcisph import PCISPHScheme
from pysph.sph.wc.shift import ShiftPositions
from pysph.sph.isph.sisph import SISPHScheme
from pysph.sph.isph.isph import ISPHScheme

from tsph_with_pst_scheme import TSPHScheme

# Local imports
from turbulence_tools import TurbulentFlowApp

# Domain and constants
L = 1.0
U = 1.0
rho0 = 1.0
c0 = 10 * U
p0 = c0**2 * rho0


def m4p(x=0.0):
    """From the paper by Chaniotis et al. (JCP 2002).
    """
    if x < 0.0:
        return 0.0
    elif x < 1.0:
        return 1.0 - 0.5 * x * x * (5.0 - 3.0 * x)
    elif x < 2.0:
        return (1 - x) * (2 - x) * (2 - x) * 0.5
    else:
        return 0.0


class M4(Equation):
    '''An equation to be used for remeshing.
    '''

    def initialize(self, d_idx, d_prop):
        d_prop[d_idx] = 0.0

    def _get_helpers_(self):
        return [m4p]

    def loop(self, s_idx, d_idx, s_temp_prop, d_prop, d_h, XIJ):
        xij = abs(XIJ[0] / d_h[d_idx])
        yij = abs(XIJ[1] / d_h[d_idx])
        d_prop[d_idx] += m4p(xij) * m4p(yij) * s_temp_prop[s_idx]


def exact_solution(U, b, t, x, y):
    factor = U * exp(b * t)

    u = -cos(2 * pi * x) * sin(2 * pi * y)
    v = sin(2 * pi * x) * cos(2 * pi * y)
    p = -0.25 * (cos(4 * pi * x) + cos(4 * pi * y))

    return factor * u, factor * v, factor * factor * p


class TaylorGreen(TurbulentFlowApp):

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
            "--pb-factor", action="store", type=float, dest="pb_factor",
            default=1.0,
            help="Use fraction of the background pressure (default: 1.0)."
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

    def consume_user_options(self):
        nx = self.options.nx
        re = self.options.re

        self.dim = 2
        self.i_nx = nx

        self.nu = nu = U * L / re

        self.dx = dx = L / nx
        self.volume = dx * dx
        self.hdx = self.options.hdx

        self.h0 = h0 = self.hdx * self.dx
        if self.options.scheme.endswith('isph'):
            dt_cfl = 0.25 * h0 / U
        else:
            dt_cfl = 0.25 * h0 / (c0 + U)
        dt_viscous = 0.125 * h0**2 / nu
        dt_force = 0.25 * 1.0

        self.dt = min(dt_cfl, dt_viscous, dt_force)
        self.tf = 1

    def configure_scheme(self):
        scheme = self.scheme
        h0 = self.hdx * self.dx
        pfreq = 100
        kernel = QuinticSpline(dim=2)
        if self.options.scheme == 'tvf':
            scheme.configure(pb=self.options.pb_factor * p0, nu=self.nu, h0=h0)
        elif self.options.scheme == 'wcsph':
            scheme.configure(hdx=self.hdx, nu=self.nu, h0=h0)
        elif self.options.scheme == 'edac':
            scheme.configure(h=h0, nu=self.nu, pb=self.options.pb_factor * p0)
        elif self.options.scheme.endswith('isph'):
            pfreq = 10
            scheme.configure(nu=self.nu)
        elif self.options.scheme == 'crksph':
            scheme.configure(h0=h0, nu=self.nu)
        elif self.options.scheme == 'gtvf':
            scheme.configure(pref=p0, nu=self.nu, h0=h0)
        elif self.options.scheme == 'tsph':
            scheme.configure()
        scheme.configure_solver(kernel=kernel, tf=self.tf, dt=self.dt,
                                pfreq=pfreq)

    def create_scheme(self):
        h0 = None
        hdx = None
        wcsph = WCSPHScheme(
            ['fluid'], [], dim=2, rho0=rho0, c0=c0, h0=h0,
            hdx=hdx, nu=None, gamma=7.0, alpha=0.0, beta=0.0
        )
        tvf = TVFScheme(
            ['fluid'], [], dim=2, rho0=rho0, c0=c0, nu=None,
            p0=p0, pb=None, h0=h0
        )
        edac = EDACScheme(
            ['fluid'], [], dim=2, rho0=rho0, c0=c0, nu=None,
            pb=p0, h=h0
        )
        iisph = IISPHScheme(
            fluids=['fluid'], solids=[], dim=2, nu=None,
            rho0=rho0, has_ghosts=True
        )
        crksph = CRKSPHScheme(
            fluids=['fluid'], dim=2, nu=None,
            rho0=rho0, h0=h0, c0=c0, p0=0.0
        )
        gtvf = GTVFScheme(
            fluids=['fluid'], solids=[], dim=2, rho0=rho0, c0=c0,
            nu=None, h0=None, pref=None
        )
        pcisph = PCISPHScheme(
            fluids=['fluid'], dim=2, rho0=rho0, nu=None
        )
        sisph = SISPHScheme(
            fluids=['fluid'], solids=[], dim=2, nu=None, rho0=rho0,
            c0=c0, alpha=0.0, has_ghosts=True, pref=p0,
            rho_cutoff=0.2, internal_flow=True, gtvf=True
        )
        isph = ISPHScheme(
            fluids=['fluid'], solids=[], dim=2, nu=None, rho0=rho0, c0=c0,
            alpha=0.0
        )
        tsph_with_pst = TSPHScheme(
            fluids=['fluid'], solids=[], dim=2, rho0=rho0, hdx=hdx, p0=p0,
            nu=None
        )
        s = SchemeChooser(
            default='wcsph', wcsph=wcsph, tvf=tvf, edac=edac, iisph=iisph,
            crksph=crksph, gtvf=gtvf, pcisph=pcisph, sisph=sisph, isph=isph,
            tsph_with_pst=tsph_with_pst
        )
        return s

    def create_domain(self):
        return DomainManager(
            xmin=0, xmax=L, ymin=0, ymax=L, periodic_in_x=True,
            periodic_in_y=True
        )

    def create_fluid(self):
        # create the particles
        dx = self.dx
        _x = np.arange(dx / 2, L, dx)
        x, y = np.meshgrid(_x, _x)
        if self.options.init is not None:
            fname = self.options.init
            from pysph.solver.utils import load
            data = load(fname)
            _f = data['arrays']['fluid']
            x, y = _f.x.copy(), _f.y.copy()

        if self.options.perturb > 0:
            np.random.seed(1)
            factor = dx * self.options.perturb
            x += np.random.random(x.shape) * factor
            y += np.random.random(x.shape) * factor

        # Initialize
        m = self.volume * rho0
        h = self.hdx * dx
        re = self.options.re
        b = -8.0 * pi * pi / re
        u0, v0, p0 = exact_solution(U=U, b=b, t=0, x=x, y=y)
        color0 = cos(2 * pi * x) * cos(4 * pi * y)

        # create the arrays
        fluid = get_particle_array(name='fluid', x=x, y=y, m=m, h=h, u=u0,
                                   v=v0, rho=rho0, p=p0, color=color0, c0=c0)
        
        self.save_initial_vel_field(
            dim=2, u=u0, v=v0, w=0., L=L, dx=self.dx
        )
        return fluid

    def create_particles(self):
        fluid = self.create_fluid()

        self.scheme.setup_properties([fluid])

        print("Taylor green vortex problem :: nfluid = %d, dt = %g" % (
            fluid.get_number_of_particles(), self.dt))

        # volume is set as dx^2
        if self.options.scheme == 'sisph':
            nfp = fluid.get_number_of_particles()
            fluid.gid[:] = np.arange(nfp)
            fluid.add_output_arrays(['gid'])
        if self.options.scheme == 'tvf':
            fluid.V[:] = 1. / self.volume
        if self.options.scheme == 'iisph':
            # These are needed to update the ghost particle properties.
            nfp = fluid.get_number_of_particles()
            fluid.orig_idx[:] = np.arange(nfp)
            fluid.add_output_arrays(['orig_idx'])
        if self.options.scheme == 'isph':
            gid = np.arange(fluid.get_number_of_particles(real=False))
            fluid.add_property('gid')
            fluid.gid[:] = gid[:]
            fluid.add_property('dpos', stride=3)
            fluid.add_property('gradv', stride=9)

        return [fluid]

    # The following are all related to post-processing.
    def _get_post_process_props(self, array):
        """Return x, y, m, u, v, p.
        """
        if 'pavg' not in array.properties or \
           'pavg' not in array.output_property_arrays:
            self._add_extra_props(array)
            sph_eval = self._get_sph_evaluator(array)
            sph_eval.update_particle_arrays([array])
            sph_eval.evaluate()

        x, y, m, u, v, p, pavg = array.get(
            'x', 'y', 'm', 'u', 'v', 'p', 'pavg'
        )
        return x, y, m, u, v, p - pavg

    def _add_extra_props(self, array):
        extra = ['pavg', 'nnbr']
        for prop in extra:
            if prop not in array.properties:
                array.add_property(prop)
        array.add_output_arrays(extra)

    def _get_sph_evaluator(self, array):
        if not hasattr(self, '_sph_eval'):
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
        info = self.read_info(info_fname)
        if len(self.output_files) == 0:
            return

        from pysph.solver.utils import iter_output
        decay_rate = -8.0 * np.pi**2 / self.options.re

        files = self.output_files
        t, ke, ke_ex, decay, linf, l1, p_l1 = [], [], [], [], [], [], []
        for sd, array in iter_output(files, 'fluid'):
            _t = sd['t']
            t.append(_t)
            x, y, m, u, v, p = self._get_post_process_props(array)
            u_e, v_e, p_e = exact_solution(U, decay_rate, _t, x, y)
            vmag2 = u**2 + v**2
            vmag = np.sqrt(vmag2)
            ke.append(0.5 * np.sum(m * vmag2))
            vmag2_e = u_e**2 + v_e**2
            vmag_e = np.sqrt(vmag2_e)
            ke_ex.append(0.5 * np.sum(m * vmag2_e))

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

        t, ke, ke_ex, decay, l1, linf, p_l1 = list(map(
            np.asarray, (t, ke, ke_ex, decay, l1, linf, p_l1))
        )
        decay_ex = U * np.exp(decay_rate * t)

        # Plots
        import matplotlib
        matplotlib.use('Agg')

        from matplotlib import pyplot as plt
        plt.clf()
        plt.semilogy(t, decay_ex, label="exact")
        plt.semilogy(t, decay, label="computed")
        plt.xlabel('t')
        plt.ylabel('max velocity')
        plt.legend()
        fig = os.path.join(self.output_dir, "decay.png")
        plt.savefig(fig, dpi=300)

        plt.clf()
        plt.plot(t, linf)
        plt.xlabel('t')
        plt.ylabel(r'$L_\infty$ error')
        fig = os.path.join(self.output_dir, "linf_error.png")
        plt.savefig(fig, dpi=300)

        plt.clf()
        plt.plot(t, l1, label="error")
        plt.xlabel('t')
        plt.ylabel(r'$L_1$ error')
        fig = os.path.join(self.output_dir, "l1_error.png")
        plt.savefig(fig, dpi=300)

        plt.clf()
        plt.plot(t, p_l1, label="error")
        plt.xlabel('t')
        plt.ylabel(r'$L_1$ error for $p$')
        fig = os.path.join(self.output_dir, "p_l1_error.png")
        plt.savefig(fig, dpi=300)
        
        # self.plot_ek_evolution()

    def customize_output(self):
        self._mayavi_config('''
        b = particle_arrays['fluid']
        b.scalar = 'vmag'
        ''')


if __name__ == '__main__':
    turb_app = TaylorGreen()
    turb_app.run()
    # turb_app.ek_post_processing(
    #     dim=2, L=L, U0=1., f_idx=0, compute_without_interp=True
    # )
    # turb_app.ek_post_processing(dim=2, L=L, U0=1., f_idx=-1)
    turb_app.post_process(turb_app.info_filename)
