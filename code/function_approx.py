'''
Approximation for function after interplation using different methods
'''

from pysph.solver.application import Application
from pysph.solver.solver import Solver
import numpy as np
import os

from config_mms import boundary_normal
from config_mms import boundary_curve


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
            "--mms", action="store", type=str, dest="mms",
            default='mms1',
            help="mms type 'mms1', 'mms2'..."
        )
        group.add_argument(
            "--bctype", action="store", type=str, dest="bctype",
            default='adami',
            help="boundary type 'adami' "
        )
        group.add_argument(
            "--bc", action="store", type=str, dest="bc",
            default='u_slip',
            help="boundary type 'u_slip', 'u_no_slip' and 'p_solid'"
        )
        group.add_argument(
            "--dim", action="store", type=int, dest="dim",
            default=2,
            help="Dimension of the domain"
        )
        group.add_argument(
            "--nl", action="store", type=int, dest="nl",
            default=8,
            help="Number of layer of solid particles"
        )
        group.add_argument(
            "--domain", action="store", type=int, dest="domain",
            default=1,
            help="domain type"
        )
        group.add_argument(
            "--pack", action="store", type=int, dest="pack",
            default=0,
            help="is packing required"
        )


    def consume_user_options(self):
        nx = self.options.nx
        self.dim = self.options.dim
        self.dx = dx = L / nx
        self.volume = dx**self.dim
        self.hdx = self.options.hdx
        self.nl = self.options.nl

        self.mms = self.options.mms
        self.bctype = self.options.bctype
        self.bc = bc = self.options.bc
        self.bcs = {bc:['solid0'], 'mms':['solid1']}
        self.pack = self.options.pack
        self.c0 = 20.0

    def create_particles(self):
        # create the particles
        from config_mms import create_particles, boundary_curve, boundary_normal
        from config_solid_bc import set_bc_props, get_bc_require
        from pysph.base.utils import get_particle_array

        is_mirror, is_boundary, is_ghost, is_ghost_mirror, is_boundary_shift = get_bc_require(
            self.bctype)
        domain = self.options.domain
        particles = create_particles(self, rho0, L, is_mirror, is_boundary,
                                     is_ghost, is_ghost_mirror,
                                     is_boundary_shift, domain, self.pack)
        set_bc_props(self.bctype, particles)

        if self.bctype == 'colagrossi':
            from solid_bc.colagrossi import create_mirror_particles
            create_mirror_particles(None, self.solver, particles, self.domain)

        props = [
            'uf', 'vf', 'wf', 'ug', 'vg', 'wg', 'ug_star', 'vg_star', 'wg_star',
            'vdotn', { 'name': 'vgraddotn', 'stride': 3 }, { 'name':
            'gradp', 'stride': 3 }, { 'name': 'L', 'stride': 16 },
            {'name': 'gradv', 'stride': 9}, 'pgraddotn', 'arho', 'wij',
            {'name': 'normal', 'stride': 3}
        ]
        for pa in particles:
            for prop in props:
                if isinstance(prop, dict):
                    pa.add_property(**prop)
                else:
                    pa.add_property(prop)
            pa.set_output_arrays([
                'h', 'm', 'x', 'y', 'rho', 'u', 'v', 'w', 'p', 'ug', 'vg', 'wg', 'ug_star', 'vg_star',
                'wg_star', 'vdotn', 'vgraddotn', 'pgraddotn', 'normal'
            ])
            pa.add_constant('c0', 10.0)
        return particles

    def create_equations(self):
        from config_approx import create_equations
        return create_equations(self.bctype, self.bcs)

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
        from config_mms import get_props


        data = load(self.output_files[-1])
        fluid = data['arrays']['fluid']
        solid0 = None
        if self.bctype == 'colagrossi':
            solid0 = data['arrays']['mirror']
        elif self.bctype == 'marongiu':
            solid0 = data['arrays']['boundary_shift']
        else:
            solid0 = data['arrays']['solid0']


        dx = self.dx
        domain = self.options.domain
        xb = np.mgrid[0:L:self.dx]
        yb = boundary_curve(xb, L, domain)
        xn, yn = boundary_normal(xb, domain)

        cond = ((xb - 0.25) > 1e-14) & ((xb - 0.75) < 1e-14)# condition to test boundary
        interp = Interpolator([fluid, solid0], kernel=self.solver.kernel, x=xb, y=yb, method='order1')


        if self.bc == 'u_slip' :
            solid0.u[:] = solid0.ug_star[:]
            solid0.v[:] = solid0.vg_star[:]
            u0 = interp.interpolate('u')
            v0 = interp.interpolate('v')
            print(u0, v0, xn, yn)
            calc = u0*xn + v0*yn
            err = np.sqrt(sum((calc[cond])**2)/len(calc[cond]))
        elif self.bc == 'u_no_slip':
            solid0.u[:] = solid0.ug[:]
            solid0.v[:] = solid0.vg[:]
            u0 = interp.interpolate('u')
            v0 = interp.interpolate('v')

            calc = [u0, v0]
            err = np.sqrt(sum(calc[0][cond]**2 + calc[1][cond]**2)/len(calc[0][cond]))
        elif self.bc == 'p_solid':
            p0 = interp.interpolate('p', comp=1)
            p1 = interp.interpolate('p', comp=2)

            calc = p0*xn + p1*yn
            err = np.sqrt(sum((calc[cond])**2)/len(calc[cond]))

        print("L2 error is ", err)

        from config_mms import get_props
        import matplotlib
        # matplotlib.use('macosx')
        from matplotlib import pyplot as plt
        xcheck, ycheck = None, None
        yval = [1.0]
        if (domain == 1) or (domain == 2):
            ycheck = np.linspace(0.98, 1.02, 200)
            xcheck = 0.3 * np.ones_like(ycheck)
        elif (domain == 3):
            xn, yn = boundary_normal(np.array([0.5]), domain)
            yval = boundary_curve(np.array([0.5]), L, domain)

            _ycheck = np.linspace(0.98, 1.02, 200) - yval
            _xcheck = 0.5 * np.ones_like(_ycheck) - 0.5
            xcheck = (_xcheck * xn[0] + _ycheck * yn[0]) * xn[0] + 0.5
            ycheck = (_xcheck * xn[0] + _ycheck * yn[0]) * yn[0] + yval
        elif (domain == 4):
            xn, yn = boundary_normal(np.array([0.5]), domain)
            yval = boundary_curve(np.array([0.5]), L, domain)

            _ycheck = np.linspace(1.22, 1.28, 200) - yval
            _xcheck = 0.5 * np.ones_like(_ycheck) - 0.5
            xcheck = (_xcheck * xn[0] + _ycheck * yn[0]) * xn[0] + 0.5
            ycheck = (_xcheck * xn[0] + _ycheck * yn[0]) * yn[0] + yval

        elif (domain == 5):
            xn, yn = boundary_normal(np.array([0.5]), domain)
            yval = boundary_curve(np.array([0.5]), L, domain)

            _ycheck = np.linspace(1.22, 1.28, 200) - yval
            _xcheck = 0.5 * np.ones_like(_ycheck) - 0.5
            xcheck = (_xcheck * xn[0] + _ycheck * yn[0]) * xn[0] + 0.5
            ycheck = (_xcheck * xn[0] + _ycheck * yn[0]) * yn[0] + yval

        # print(xcheck, ycheck)

        uac, vac, wac, rhocac, pac = get_props(xcheck, ycheck, np.zeros_like(xcheck), 0.0, self.c0, self.mms)


        interp = Interpolator([fluid, solid0], kernel=self.solver.kernel, x=xcheck, y=ycheck, method='shepard')
        p = interp.interpolate('p')
        u = interp.interpolate('u')
        v = interp.interpolate('v')

        if self.bc.split('_')[0] == 'p':
            plt.plot(ycheck, p, label='P')
            plt.plot(ycheck, pac, '--k', label='Pac')
            y = [min(p), max(p)]
        else:
            plt.plot(ycheck, u, label='U')
            plt.plot(ycheck, uac, '--k', label='Uac')
            y = [min(u), max(u)]

        x0 = [yval[0]-self.dx/2, yval[0]-self.dx/2]
        plt.plot(x0, y, '-r')

        x1 = [yval[0]+(1.2-0.5)*self.dx, yval[0]+(1.2-0.5)*self.dx]
        plt.plot(x1, y, '--r')

        filename = os.path.join(self.output_dir, 'results.npz')
        np.savez(filename, x0=x0, x1=x1, y=y, ycheck=ycheck, u=u, p=p, uac=uac, pac=pac, err=err)

        # plt.grid()
        # plt.legend()
        # plt.show()


if __name__ == "__main__":
    app = Approx()
    app.run()
    app.post_process(app.info_filename)