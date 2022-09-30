'''
Approximation in a finite domain
'''

from pysph.solver.application import Application
from pysph.solver.solver import Solver
from pysph.sph.equation import Group
import numpy as np
import os

L = 1.0
rho0 = 1.0

class Approx(Application):
    def add_user_options(self, group):
        group.add_argument(
            "--nx", action="store", type=int, dest="nx", default=50,
            help="Number of points along x direction. (default 50)"
        )
        group.add_argument(
            "--hdx", action="store", type=float, dest="hdx", default=1.2,
            help="Ratio h/dx."
        )
        group.add_argument(
            "--dim", action="store", type=int, dest="dim",
            default=2,
            help="Dimension of the domain"
        )
        group.add_argument(
            "--pack", action="store", type=int, dest="pack",
            default=0,
            help="is packing required"
        )
        group.add_argument(
            "--approx", action="store", type=str, dest="approx",
            default='first',
            help="derivative required"
        )


    def consume_user_options(self):
        nx = self.options.nx
        self.dim = self.options.dim
        self.dx = dx = L / nx
        self.volume = dx**self.dim
        self.hdx = self.options.hdx
        self.nu = 0.1

        self.approx = self.options.approx
        self.pack = self.options.pack
        self.c0 = 20.0

    def get_function(self, x, y):
        u = np.sin(4 * np.pi * (x + y))
        v = np.cos(4 * np.pi * (x + y))
        p = np.sin(4 * np.pi * x) + np.cos(4 * np.pi * y)
        return u, v, p

    def get_exact(self, x, y):
        if self.approx == 'first':
            return -4 * np.pi * np.cos(4 * np.pi * x)
        elif self.approx == 'second':
            return -32. * self.nu * np.pi**2 * np.sin(4 * np.pi * (x + y))

    def create_particles(self):
        # create the particles
        from pysph.base.utils import get_particle_array

        particles = []
        dx = self.dx
        xb, yb = np.mgrid[0:L+dx/2:dx, 0:L+dx/2:dx]
        u, v, p = self.get_function(xb, yb)
        fluid = get_particle_array(
            name='fluid', x=xb, y=yb, m=1.0, h=self.hdx*self.dx, rho=1.0, rhoc=1.0,
            u=u, v=v, p=p)

        particles.append(fluid)

        props = ['x', 'y', 'au', 'av', 'aw']
        for pa in particles:
            for prop in props:
                pa.add_property(prop)
            pa.add_property('m_mat', stride=9)
            pa.add_property('gradv', stride=9)
            pa.set_output_arrays(props)

        return particles

    def create_equations(self):
        from pysph.sph.wc.kernel_correction import GradientCorrectionPreStep, GradientCorrection
        from tsph_with_pst import (DivGrad, VelocityGradient, MomentumEquationSecondOrder)
        from pysph.sph.basic_equations import SummationDensity

        eqns = []
        g0 = []
        g0.append(SummationDensity(dest='fluid', sources=['fluid']))
        eqns.append(Group(equations=g0))

        g0 = []
        g0.append(GradientCorrectionPreStep(dest='fluid', sources=['fluid']))
        eqns.append(Group(equations=g0))

        g0 = []

        g0.append(GradientCorrection(dest='fluid', sources=['fluid']))
        if self.approx == 'first':
            g0.append(MomentumEquationSecondOrder(dest='fluid', sources=['fluid']))
            eqns.append(Group(equations=g0))
        elif self.approx == 'second':
            g0.append(VelocityGradient(dest='fluid', sources=['fluid']))
            eqns.append(Group(equations=g0))

            g0 = []
            g0.append(GradientCorrection(dest='fluid', sources=['fluid']))
            g0.append(DivGrad(dest='fluid', sources=['fluid'], nu=self.nu))
            eqns.append(Group(equations=g0))

        print(eqns)

        return eqns


    def create_solver(self):
        from pysph.solver.solver import Solver
        from pysph.sph.integrator import EulerIntegrator
        from pysph.base.kernels import QuinticSpline
        from config_approx import EulerStepDummy

        kernel = QuinticSpline(dim=self.dim)
        solver = Solver(dim=self.dim, kernel=kernel, dt=1.0, tf=1.0, integrator=EulerIntegrator(fluid=EulerStepDummy()))
        return solver


    def post_process(self, info_fname):
        from pysph.tools.interpolator import Interpolator
        self.read_info(info_fname)
        if len(self.output_files) == 0:
            return

        from pysph.solver.utils import iter_output
        from pysph.solver.utils import load


        data = load(self.output_files[-1])
        fluid = data['arrays']['fluid']

        x = fluid.x
        y = fluid.y

        exact = self.get_exact(x, y)
        comp = fluid.au
        print(comp)

        # import matplotlib.pyplot as plt
        # plt.plot(exact[1000:1500], label='exact')
        # plt.plot(comp[1000:1500], label='comp')
        # plt.grid()
        # plt.legend()
        # plt.show()


        filename = os.path.join(self.output_dir, 'results.npz')
        np.savez(filename, exact=exact, comp=comp, x=x, y=y)


if __name__ == "__main__":
    app = Approx()
    app.run()
    app.post_process(app.info_filename)