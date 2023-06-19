r"""
Test :math:`k-\varepsilon` Model
Author: K T Prajwal Prathiksh
###
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from pprint import pprint

from pysph.base.nnps import DomainManager
from pysph.base.utils import get_particle_array
from pysph.base.kernels import WendlandQuinticC4
from pysph.sph.scheme import SchemeChooser
from pysph.solver.utils import iter_output

from turbulence import TurbulentFlowApp
from k_eps import KEpsilonScheme

# domain and constants
L = 1.0
U = 1.0
rho0 = 1.0

Cd = 0.09
sigma_k = 1.0
sigma_eps = 1.3
C1eps = 1.44
C2eps = 1.92


def get_sym_grad(scal_f):
    """
    Returns the symbolic gradient of a scalar function.
    """
    x, y = sp.symbols('x y')
    grad_f = sp.Matrix([sp.diff(scal_f, x), sp.diff(scal_f, y)])
    return grad_f

def get_sym_div(vec_f):
    """
    Returns the symbolic divergence of a vector function.
    """
    x, y = sp.symbols('x y')
    div_f = sp.diff(vec_f[0], x) + sp.diff(vec_f[1], y)
    return div_f

def get_test_solution(_x, _y, c0=10., test_case=0):
    cos, sin, pi = sp.cos, sp.sin, sp.pi
    diff, sqrt = sp.diff, sp.sqrt

    x, y, u, v = sp.symbols('x y u v')
    rho, p = sp.symbols('rho p')
    k, eps = sp.symbols('k epsilon')

    u = -cos(2*pi*x)*sin(2*pi*y)
    v = sin(2*pi*x)*cos(2*pi*y)
    vmag = sqrt(u**2 + v**2)

    p = -0.25*(cos(4*pi*x) + cos(4*pi*y))
    rho = (p*7/c0**2 + 1)**(1/7)

    # Initialize k and eps
    CHOICES = [0, 1, 2]
    if test_case not in CHOICES:
        raise ValueError("Invalid test case. Choose from %s" % CHOICES)
    if test_case == 0:
        k = 1.5*U*U
        eps = pow(0.09, 0.75)*pow(k, 1.5)/L
    elif test_case == 1:
        k = vmag**2
        eps = k
    elif test_case == 2:
        k = 0.5*vmag**2
        eps = 0.09*k**(3/2)/L
        
    # Calculate RHS of k and eps equations
    grad_u = sp.Matrix([[diff(u, x), diff(u, y)], [diff(v, x), diff(v, y)]])
    S = (grad_u + grad_u.T)/2
    frob_S = sqrt(S[0, 0]**2 + 2*S[0, 1]**2 + S[1, 1]**2)

    grad_k = get_sym_grad(k)
    grad_eps = get_sym_grad(eps)

    Pk = Cd * sqrt(2) * frob_S

    ## k equation
    k_div = get_sym_div((k*k/eps)*grad_k)
    k_rhs = (Cd/sigma_k)*k_div + Pk - eps

    ## eps equation
    eps_div = get_sym_div((k*k/eps)*grad_eps)
    eps_rhs = (Cd/sigma_eps)*eps_div + C1eps*eps*Pk/k - C2eps*eps*eps/k

    def numerical_subs(sym_var, _x, _y, debug=False):
        func = sp.lambdify((x, y), sym_var, 'numpy')
        if debug:
            print(func.__doc__)
            return
        return func(_x, _y)
    
    _u = numerical_subs(u, _x, _y)
    _v = numerical_subs(v, _x, _y)
    _p = numerical_subs(p, _x, _y)
    _rho = numerical_subs(rho, _x, _y)

    _k = numerical_subs(k, _x, _y)
    _eps = numerical_subs(eps, _x, _y)

    _k_rhs = numerical_subs(k_rhs, _x, _y)
    _eps_rhs = numerical_subs(eps_rhs, _x, _y)

    return _u, _v, _p, _rho, _k, _eps, _k_rhs, _eps_rhs


class TestKEpsModel(TurbulentFlowApp):
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
            "--hdx", action="store", type=float, dest="hdx", default=2.0,
            help="Ratio h/dx."
        )
        group.add_argument(
            "--c0-fac", action="store", type=float, dest="c0_fac",
            default=20.0, help="Speed of sound multiplier."
        )
        group.add_argument(
            "--re", action="store", type=float, dest="re", default=100,
            help="Reynolds number (defaults to 100)."
        )
        group.add_argument(
            "-ketc", "--k-eps-test-case", action="store", type=int,
            dest="k_eps_test_case", default=0, help="Run the k-eps test case."
        )

    def consume_user_options(self):
        self.perturb = self.options.perturb
        self.nx = self.options.nx
        self.hdx = self.options.hdx
        self.c0_fac = self.options.c0_fac
        self.re = self.options.re
        self.k_eps_test_case = self.options.k_eps_test_case

        self.dx = dx = L / self.nx
        self.volume = dx * dx
        self.h0 = self.hdx * dx
        self.nu = nu = U * L / self.re
        self.c0 = self.c0_fac * U

        dt_cfl = 0.25 * dx / (self.c0 + U)
        dt_viscous = 0.125 * dx**2 / nu
        dt_force = 0.25 * 1.0
        _d = dict(cfl=dt_cfl, viscous=dt_viscous, force=dt_force)
        pprint(_d)

        self.dt = dt = min(dt_cfl, dt_viscous, dt_force)
        self.tf = dt * 1.1

    def create_scheme(self):
        h0 = None
        hdx = None
        c0 = None
        k_eps = KEpsilonScheme(
            fluids=['fluid'], solids=[], dim=2, rho0=rho0, c0=c0, h0=h0,
            hdx=hdx, nu=None, gamma=7.0, kernel_corr=True
        )
        s = SchemeChooser(default='k_eps', k_eps=k_eps)
        return s

    def configure_scheme(self):
        scheme = self.scheme
        h0 = self.hdx * self.dx
        kernel = WendlandQuinticC4(dim=2)
        c0 = self.c0
        
        if self.options.scheme == 'k_eps':
            scheme.configure(
                hdx=self.hdx, nu=self.nu, h0=h0, c0=c0, periodic=True
            )

        scheme.configure_solver(kernel=kernel, dt=self.dt, tf=self.tf, pfreq=1)
    
    def create_equations(self):
        return self.scheme.get_equations()

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

        if self.perturb > 0:
            np.random.seed(1)
            factor = dx * self.options.perturb
            x += np.random.random(x.shape) * factor
            y += np.random.random(x.shape) * factor

        # Initialize
        m = self.volume * rho0
        h = self.hdx * dx
        re = self.re

        u, v, p, rho, k, eps, k_rhs, eps_rhs = get_test_solution(
            _x=x, _y=y, c0=self.c0, test_case=self.k_eps_test_case
        )

        # create the arrays
        fluid = get_particle_array(
            name='fluid', x=x, y=y, m=m, h=h, u=u, v=v, p=p, rho=rho, rhoc=rho,
            k=k, eps=eps, k_rhs=k_rhs, eps_rhs=eps_rhs
        )
        return fluid

    def create_particles(self):
        fluid = self.create_fluid()
        self.scheme.setup_properties([fluid], clean=False)

        print("Test K-Epsilon Model :: nfluid = %d, dt = %g" % (
            fluid.get_number_of_particles(), self.dt))
        
        nfp = fluid.get_number_of_particles()
        fluid.gid[:] = np.arange(nfp)

        fluid_oprops = fluid.output_property_arrays + [
            'ak', 'aeps', 'k_rhs', 'eps_rhs'
        ]
        fluid.set_output_arrays(fluid_oprops)
        
        return [fluid]

    def post_process(self, info_fname):
        self.read_info(info_fname)
        if len(self.output_files) == 0:
            return


        files = self.output_files
        t, k_rhs, eps_rhs = [], [], []
        k_rhs_ex, eps_rhs_ex = [], []
        x, y = [], []

        for sd, array in iter_output(files[0:], 'fluid'):
            t.append(sd['t'])
            k_rhs.append(array.get('ak'))
            eps_rhs.append(array.get('aeps'))

            k_rhs_ex.append(array.get('k_rhs'))
            eps_rhs_ex.append(array.get('eps_rhs'))

            x.append(array.get('x'))
            y.append(array.get('y'))

        fname = os.path.join(self.output_dir, "results.npz")
        np.savez(
            fname, t=t[-1],
            x=x[-1], y=y[-1],
            k_rhs=k_rhs[-1], eps_rhs=eps_rhs[-1],
            k_rhs_ex=k_rhs_ex[-1], eps_rhs_ex=eps_rhs_ex[-1]
        )

        def _common_plt_macro():
            plt.xlabel('x')
            plt.ylabel('y')
            plt.colorbar()
            plt.axis('equal')
            plt.xlim(0, L)
            plt.ylim(0, L)
        
        plt.figure(figsize=(12, 6))
        plt.clf()
        plt.subplot(1, 2, 1)
        plt.scatter(x[-1], y[-1], c=k_rhs[-1], s=10)
        plt.title(r'$k$ (RHS)')
        _common_plt_macro()

        plt.subplot(1, 2, 2)
        plt.scatter(x[-1], y[-1], c=k_rhs_ex[-1], s=10)
        plt.title(r'$k$ (RHS) Exact')
        _common_plt_macro()

        plt.suptitle(
            fr'$k$ at t = {t[-1]:.2f} (Re = {self.re}, c0 = {self.c0}) '
            f'(TC = {self.k_eps_test_case})'
        )
        fname = os.path.join(self.output_dir, "k_rhs.png")
        plt.savefig(fname, dpi=300)

        plt.clf()
        plt.subplot(1, 2, 1)
        plt.scatter(x[-1], y[-1], c=eps_rhs[-1], s=10)
        plt.title(r'$\epsilon$ (RHS)')
        _common_plt_macro()

        plt.subplot(1, 2, 2)
        plt.scatter(x[-1], y[-1], c=eps_rhs_ex[-1], s=10)
        plt.title(r'$\epsilon$ (RHS) Exact')
        _common_plt_macro()
        plt.suptitle(
            fr'$\epsilon$ at t = {t[-1]:.2f} (Re = {self.re}, c0 = {self.c0}) '
            f'(TC = {self.k_eps_test_case})'
        )
        fname = os.path.join(self.output_dir, "eps_rhs.png")
        plt.savefig(fname, dpi=300)
        

    def customize_output(self):
        self._mayavi_config('''
        b = particle_arrays['fluid']
        b.scalar = 'k'
        ''')


if __name__ == '__main__':
    app = TestKEpsModel()
    app.run()
    app.post_process(app.info_filename)