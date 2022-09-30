"""
2D vortex
"""

import os
import numpy as np
from numpy import pi, cos

from pysph.base.nnps import DomainManager
from pysph.base.utils import get_particle_array
from pysph.base.kernels import QuinticSpline
from pysph.solver.application import Application
from pysph.sph.basic_equations import SummationDensity
from pysph.sph.equation import Group
from sympy.geometry import parabola
from config_mms import get_props

# domain and constants
U = 1.0
B = 2.0
rho0 = 1.0
c0 = 40
p0 = c0**2 * rho0


class Pulse(Application):
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
        group.add_argument(
            "--lt", action="store", type=float, dest="lt",
            default=2.0,
            help="Number of layer of solid particles"
        )


    def consume_user_options(self):
        nx = self.options.nx
        re = self.options.re
        L = self.L = self.options.lt

        self.c0 = c0
        self.nu = self.options.nu

        self.dim = self.options.dim
        self.dx = dx = 1.0 / nx
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
        from pysph.base.kernels import QuinticSpline
        from config_io import set_fluid_names, get_stepper
        from config_io import get_inlet_outlet_manager
        from io_bc.common import MirrorStep

        kernel = QuinticSpline(dim=2)
        h = self.dx * self.hdx

        scheme = app.scheme.scheme
        set_fluid_names(self.bctype, scheme)
        iom = get_inlet_outlet_manager(self, xi=0.0, xo=self.L, is_inlet=True)

        app.iom = iom

        app.scheme.configure(hdx=self.hdx, nu=self.nu)

        extra_steppers = get_stepper(self.bctype, self.bc)
        extra_steppers['mirror_inlet'] = MirrorStep()
        print(extra_steppers)

        app.scheme.configure_solver(kernel=kernel,
                                    tf=self.tf,
                                    dt=self.dt,
                                    extra_steppers=extra_steppers)

    def create_scheme(self):
        from pysph.sph.scheme import SchemeChooser
        from tsph_with_pst_scheme import TSPHSchemeMain

        tsph = TSPHSchemeMain(
            ['fluid'], [], ios=['inlet', 'outlet'], dim=2, rho0=rho0,
            hdx=None, nu=None, gamma=7.0, kernel_corr=True
        )

        s = SchemeChooser(
            default='tsph', tsph=tsph
        )

        return s

    def create_equations(self):
        from config_mms import create_equations_io
        eqns = self.scheme.get_equations()
        self.get_inlet_eqns(eqns)

        if self.bctype == 'mirror':
            self.get_outlet_mirror(eqns)
        elif self.bctype == 'mirror_new':
            self.get_outlet_mirror_new(eqns)
        elif self.bctype == 'hybrid':
            self.get_outlet_hybrid(eqns)
        print(eqns)
        return eqns

    def get_inlet_eqns(self, eqns):
        from io_bc.mirror import velocity_eq, pressure_eq
        g0, g1, g2, g3 = pressure_eq('inlet', 'mirror_inlet', xn=-1.0, xo=0.0, sources=['fluid'])

        eqns[0].equations.extend(g0)
        eqns.insert(1, Group(equations=g1, update_nnps=True))
        eqns.insert(2, Group(equations=g2, update_nnps=True))
        eqns.insert(3, Group(equations=g3, update_nnps=True))

    def get_outlet_mirror(self, eqns):
        from io_bc.mirror import velocity_eq, pressure_eq
        g0, g1, g2, g3 = velocity_eq('outlet', 'mirror_outlet', xn=1.0, xo=self.options.lt, sources=['fluid'])
        _g0, _g1, _g2, _g3 = pressure_eq('outlet', 'mirror_outlet', xn=1.0, xo=self.options.lt, sources=['fluid'])

        g0.pop()
        g2.extend(_g2[1:])
        g3.extend(_g3)

        eqns[0].equations.extend(g0)
        eqns[1].equations.extend(g1)
        eqns[2].equations.extend(g2)
        eqns[3].equations.extend(g3)

    def get_outlet_hybrid(self, eqns):
        from io_bc.hybrid import eq1, EvaluateOutletVelocity, EvaluateOutletPressure
        g0 = eq1('outlet', u0=U)

        g1 = [EvaluateOutletVelocity(dest='outlet', sources=None),
              EvaluateOutletPressure(dest='outlet', sources=None)]

        eqns[1].equations.extend(g0)
        eqns[2].equations.extend(g1)

    def get_outlet_mirror_new(self, eqns):
        from io_bc.mirror_new import (
            velocity_eq, pressure_eq, EvaluateVelocityOnGhostOutlet,
            EvaluatePressureOnGhostOutlet)
        g0, g1, g2 = velocity_eq('outlet', 'mirror_outlet', xn=1.0, xo=self.options.lt, sources=['fluid'], u0=U)
        _g0, _g1, _g2 = pressure_eq('outlet', 'mirror_outlet', xn=1.0, xo=self.options.lt, sources=['fluid'], u0=U)

        g0.pop()
        g2.extend(_g2[1:])
        g3 = [
           EvaluateVelocityOnGhostOutlet(dest='outlet', sources=['mirror_outlet']),
           EvaluatePressureOnGhostOutlet(dest='outlet', sources=['mirror_outlet'])
        ]

        eqns[1].equations.extend(g0)
        eqns[2].equations.extend(g1)
        eqns[3].equations.extend(g2)
        eqns[4].equations.extend(g3)

    def initialize_vortex(self, x, y):
        u = np.zeros_like(x) + U
        v = np.zeros_like(x)
        w = np.zeros_like(x)
        p = 0.5 + np.exp(-10*(x-1)**2)
        rhoc = p/c0**2 + rho0
        return u, v, w, rhoc, p

    def create_ios(self, mirror_outlet):

        from pysph.base.utils import get_particle_array
        from pysph.tools.geometry import remove_overlap_particles

        L = self.L
        dx = self.dx
        nl = self.nl * dx
        xf, yf = np.mgrid[dx / 2:L:dx, dx / 2:B:dx]
        xf, yf = [np.ravel(t) for t in (xf, yf)]

        xs, ys = np.mgrid[dx / 2 - nl:L + nl:dx, dx / 2:B:dx]
        xs, ys = [np.ravel(t) for t in (xs, ys)]
        cond = ((ys < 0) | (ys > L))

        # Initialize
        m = self.volume * rho0
        V0 = self.volume
        h = self.hdx * dx
        rho = rho0

        uf, vf, wf, rhocf, pf = self.initialize_vortex(xf, yf)
        fluid = get_particle_array(name='fluid', x=xf, y=yf, m=m, h=h, rho=rho,
                                rhoc=rhocf, u=uf, v=vf, w=wf, p=pf, xn=0, yn=0, zn=0)

        condi = (xs < 0.5) & (~cond)
        ui, vi, wi, rhoci, pi = self.initialize_vortex(xs[condi], ys[condi])
        inlet = get_particle_array(name='inlet', x=xs[condi], y=ys[condi], m=m,
                                h=h, rho=rho, rhoc=rhoci, u=ui, v=vi,
                                w=wi, p=pi, xn=-1.0, yn=0.0, zn=0.0)

        condo = (xs > 0.5) & (~cond)
        uo, vo, wo, rhoco, po = self.initialize_vortex(xs[condo], ys[condo])
        outlet = get_particle_array(name='outlet', x=xs[condo], y=ys[condo], m=m,
                                    h=h, rho=rho, rhoc=rhoco, u=uo, v=vo,
                                    w=wo, p=po, xn=1.0, yn=0.0, zn=0.0)

        remove_overlap_particles(inlet, fluid, dx, dim=2)
        remove_overlap_particles(outlet, fluid, dx, dim=2)
        particles = [fluid, inlet, outlet]


        for pa in particles:
            pa.add_constant('uref', 1.0)
            nfp = pa.get_number_of_particles()
            pa.gid[:] = np.arange(nfp)

        ''' mirror to the inlet particles created inside the fluid remains fixed
        during the simulation'''
        x = inlet.x
        y = inlet.y
        z = inlet.z
        xgm, ygm, zgm = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)

        xgm = -x
        ygm = y
        zgm = z

        usgm, vsgm, wsgm, rhocsgm, psgm = self.initialize_vortex(x, y)
        mirror_inlet = get_particle_array(
            name='mirror_inlet', x=xgm, y=ygm, m=m, h=h, rho=rho0,
            rhoc=rhocsgm, u=usgm, v=vsgm, w=wsgm, p=psgm)
        particles.append(mirror_inlet)

        if mirror_outlet == True:
            ''' mirror to the inlet particles created inside the fluid remains fixed
            during the simulation'''
            x = outlet.x
            y = outlet.y
            z = outlet.z
            xgm, ygm, zgm = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)

            xgm = 2*L - x
            ygm = y
            zgm = z

            usgm, vsgm, wsgm, rhocsgm, psgm = self.initialize_vortex(x, y)
            mirror_outlet = get_particle_array(
                name='mirror_outlet', x=xgm, y=ygm, m=m, h=h, rho=rho0,
                rhoc=rhocsgm, u=-usgm, v=-vsgm, w=-wsgm, p=psgm)
            particles.append(mirror_outlet)

        return particles

    def create_particles(self):
        # create the particles
        from particle_arrays import create_particles_io
        from config_io import get_bc_require, set_bc_props
        mirror_inlet, mirror_outlet = get_bc_require(
            self.bctype, self.bc)
        particles = self.create_ios(mirror_outlet)
        self.scheme.setup_properties(particles, clean=False)
        set_bc_props(self.bctype, particles)

        print("problem :: nfluid = %d, dt = %g" % (
            particles[0].get_number_of_particles(), self.dt))

        props = ['ioid', 'disp']
        for pa in particles:
            pa.add_constant('c0', self.c0)
            pa.add_property('L', stride=16)
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

    def create_domain(self):
        from pysph.base.nnps import DomainManager

        return DomainManager(ymin=0, ymax=B, periodic_in_y=True)

    def create_inlet_outlet(self, particle_arrays):
        iom = self.iom
        io = iom.get_inlet_outlet(particle_arrays)
        return io

    def post_process(self, info_fname):
        from pysph.tools.interpolator import Interpolator
        from pysph.solver.utils import iter_output
        self.read_info(info_fname)
        if len(self.output_files) == 0:
            return

        files = self.output_files
        pressures = []
        vels = []
        times = []
        for sd, arrays in iter_output(files[0:]):
            fluid = arrays['fluid']
            times.append(sd['t'])
            interp = Interpolator([fluid], x=[1.9], y=[0.5], method='order1')
            p = interp.interpolate('p')
            u = interp.interpolate('u')
            v = interp.interpolate('v')
            vel = np.sqrt(u**2 + v**2)
            print(p)
            pressures.append(p)
            vels.append(vel)

        # import matplotlib
        # matplotlib.use('macosx')

        # from matplotlib import pyplot as plt
        # plt.plot(pressures)
        # plt.show()
        fname = os.path.join(self.output_dir, 'results.npz')
        np.savez(
            fname, p=pressures, t=times, v=vels
        )


    def customize_output(self):
        self._mayavi_config('''
        b = particle_arrays['fluid']
        b.scalar = 'u'
        ''')

if __name__ == '__main__':
    app = Pulse()
    app.run()
    app.post_process(app.info_filename)
