from collections import defaultdict
import numpy as np
from pysph.base.kernels import (
    Gaussian, CubicSpline, QuinticSpline,
    WendlandQuintic, WendlandQuinticC4, WendlandQuinticC6
)
from pysph.base.utils import get_particle_array
from pysph.base.nnps_base import DomainManager
from pysph.sph.equation import Equation, Group
from pysph.sph.basic_equations import SummationDensity
from pysph.tools.interpolator import Interpolator
from pysph.sph.scheme import Scheme
from compyle.api import declare
from scheme_equation import qs_dwdq, qs_dwdq2


def lambda2(x=0.0):
    if x < 0.0:
        return 0.0
    elif x < 0.5:
        return 1 - x*x
    elif x < 1.5:
        return (1 - x)*(2 - x)*0.5
    else:
        return 0.0


def lambda3(x=0.0):
    if x < 0.0:
        return 0.0
    elif x < 1.0:
        return (1 - x*x)*(2 - x)/2
    elif x <= 2.0:
        return (1 - x)*(2 - x)*(3 - x)/6
    else:
        return 0.0


def m4p(x=0.0):
    """From the paper by Chaniotis et al (JCP 2002).
    """
    if x < 0.0:
        return 0.0
    elif x < 1.0:
        return 1.0 - 0.5*x*x*(5.0 - 3.0*x)
    elif x < 2.0:
        return (1 - x)*(2 - x)*(2 - x)*0.5
    else:
        return 0.0


class Lambda2(Equation):
    def initialize(self, d_idx, d_prop):
        d_prop[d_idx] = 0.0

    def _get_helpers_(self):
        return [lambda2]

    def loop(self, s_idx, d_idx, s_temp_prop, d_prop, d_h, XIJ):
        xij = abs(XIJ[0]/d_h[d_idx])
        yij = abs(XIJ[1]/d_h[d_idx])
        d_prop[d_idx] += lambda2(xij)*lambda2(yij)*s_temp_prop[s_idx]


class Lambda3(Equation):
    def initialize(self, d_idx, d_prop):
        d_prop[d_idx] = 0.0

    def _get_helpers_(self):
        return [lambda3]

    def loop(self, s_idx, d_idx, s_temp_prop, d_prop, d_h, XIJ):
        xij = abs(XIJ[0]/d_h[d_idx])
        yij = abs(XIJ[1]/d_h[d_idx])
        d_prop[d_idx] += lambda3(xij)*lambda3(yij)*s_temp_prop[s_idx]


class M4(Equation):
    def __init__(self, dest, sources, hdx=1.2):
        self.hdx = hdx

        super(M4, self).__init__(dest, sources)

    def initialize(self, d_idx, d_prop):
        d_prop[d_idx] = 0.0

    def _get_helpers_(self):
        return [m4p]

    def loop(self, s_idx, d_idx, s_temp_prop, d_prop, d_h, XIJ):
        xij = abs(XIJ[0]/d_h[d_idx] * self.hdx)
        yij = abs(XIJ[1]/d_h[d_idx] * self.hdx)
        d_prop[d_idx] += m4p(xij)*m4p(yij)*s_temp_prop[s_idx]

class M41(Equation):
    '''An equation to be used for remeshing.
    '''
    def __init__(self, dest, sources, hdx=1.2):
        self.hdx = hdx

        super(M41, self).__init__(dest, sources)

    def initialize(self, d_idx, d_prop):
        d_prop[d_idx] = 0.0

    def _get_helpers_(self):
        return [m4p]

    def loop_all(self, d_idx, s_temp_prop, d_prop, d_h, d_x, d_y, s_x, s_y, s_m, s_rho, N_NBRS, NBRS):
        i, s_idx = declare('int', 2)

        V = 0.0

        for i in range(N_NBRS):
            s_idx = NBRS[i]
            xij = d_x[d_idx] - s_x[s_idx]
            yij = d_y[d_idx] - s_y[s_idx]
            vold = s_m[s_idx] / s_rho[s_idx]
            vnew = (d_h[d_idx]/self.hdx)**2
            xij = abs(xij/d_h[d_idx] * self.hdx)
            yij = abs(yij/d_h[d_idx] * self.hdx)
            V +=  m4p(xij)*m4p(yij)
            d_prop[d_idx] += m4p(xij)*m4p(yij)*s_temp_prop[s_idx]

        if abs(V) > 1e-14:
            # print(V)
            d_prop[d_idx] /= V


class SimpleSPH(Equation):
    def initialize(self, d_idx, d_prop):
        d_prop[d_idx] = 0.0

    def loop(self, s_idx, d_idx, s_temp_prop, d_prop, s_m, s_rho, WIJ):
        d_prop[d_idx] += s_temp_prop[s_idx]*s_m[s_idx]/s_rho[s_idx]*WIJ


simple_sph_eqs = [
    Group(
        equations=[SummationDensity(dest='src', sources=['src'])],
        update_nnps=True
    ),
    Group(
        equations=[SimpleSPH(dest='interpolate', sources=['src'])]
    )
]


def get_u(x, y):
    return np.cos(2*np.pi*x) * np.sin(2*np.pi*y)


def perturb_xy(x, y, delta):
    np.random.seed(327)
    x += delta*(np.random.random(x.shape) - 0.5)
    y += delta*(np.random.random(y.shape) - 0.5)


def get_x_y(n=20, perturb=0.2):
    dx = 1.0/(n)
    L = 1.0
    x, y = np.mgrid[dx/2:L:dx, dx/2:L:dx]
    if perturb > 0.0:
        delta = perturb*dx
        perturb_xy(x, y, delta)

    return x, y


def setup(hdx=1.0, n=50, perturb=0.2):
    dx = 1.0/n
    x, y = get_x_y(n=n, perturb=perturb)
    u = get_u(x, y)
    src = get_particle_array(name='src', x=x, y=y, u=u, m=1.0, h=dx*hdx)
    x_t, y_t = get_x_y(n=n, perturb=0.0)
    u_t = get_u(x_t, y_t)
    return src, x_t, y_t, u_t


def setup1(hdx=1.0, n=50, perturb=0.2):
    dx = 1.0/n
    x, y = get_x_y(n=n, perturb=perturb)
    x_t, y_t = get_x_y(n=n, perturb=0.0)
    u_t = get_u(x_t, y_t)
    src = get_particle_array(name='src', x=x, y=y, u=u_t, m=1.0, h=dx*hdx)
    return src, x_t, y_t, u_t


def check_kernel(kernel, hdx, perturb=0.2, equations=None, n=20):
    domain = DomainManager(
        xmin=0, xmax=1.0, ymin=0, ymax=1.0, periodic_in_x=True,
        periodic_in_y=True
    )
    src, x_t, y_t, u_t = setup(hdx=hdx, n=n, perturb=perturb)
    pas = [src]
    interp = Interpolator(
        pas, x=x_t, y=y_t, domain_manager=domain, kernel=kernel,
        equations=equations
    )
    u = interp.interpolate('u')
    return u, u_t


def find_error(kernel, hdx, perturb=0.2, equations=None):
    domain = DomainManager(
        xmin=0, xmax=1.0, ymin=0, ymax=1.0, periodic_in_x=True,
        periodic_in_y=True
    )

    interp = None
    l1, l2, linf, mom = [], [], [], []
    N = [10, 20, 40, 80, 160, 200, 300, 400, 800]
    for n in N:
        src, x_t, y_t, u_t = setup(hdx=hdx, n=n, perturb=perturb)
        pas = [src]
        if interp is None:
            interp = Interpolator(
                pas, x=x_t, y=y_t, domain_manager=domain, kernel=kernel,
                equations=equations
            )
        else:
            interp.update_particle_arrays(pas)
            interp.set_interpolation_points(x_t, y_t)
        u = interp.interpolate('u')
        diff = u - u_t
        n_p = x_t.size
        e1 = np.sum(np.abs(diff))/n_p
        e2 = np.sqrt(np.sum(diff**2)/n_p)
        einf = np.max(np.abs(diff))
        l1.append(e1)
        l2.append(e2)
        linf.append(einf)
        dx = 1.0/n
        old_mom = compute_moments(src.x, src.y, src.u, dx)
        new_mom = compute_moments(x_t, y_t, u, dx)
        mom_change = diff_mom(new_mom, old_mom)
        mom.append(mom_change)

    moment = defaultdict(list)
    for m in mom:
        for k in m:
            moment[k].append(m[k])
    moment = {k: np.asarray(moment[k]) for k in moment}

    N = np.asarray(N)
    l1, l2, linf = [np.asarray(_x) for _x in (l1, l2, linf)]

    return dict(sizes=N, l1=l1, l2=l2, linf=linf, moment=moment)


def diff_mom(m1, m2):
    return {k: abs(m1[k] - m2[k]) for k in m1}


def compute_moments(x, y, u, dx):
    m0 = np.sum(u)*dx*dx
    m1x = np.sum(u*x)*dx*dx
    m1y = np.sum(u*y)*dx*dx
    m2x = np.sum(u*x*x)*dx*dx
    m2y = np.sum(u*y*y)*dx*dx
    m2 = m2x + m2y
    m2xy = np.sum(u*x*y)*dx*dx
    return dict(m0=m0, m1x=m1x, m1y=m1y, m2x=m2x, m2y=m2y,
                m2xy=m2xy, m2=m2)


def setup_for_moment(hdx, n):
    np.random.seed(327)
    # points between (0.25, 0.25) to (0.75, 0.75)
    xr, yr = 0.5*np.random.random((2, n)) + 0.25
    u = np.random.random(n)
    x, y = np.mgrid[0:1:2j*n, 0:1:2j*n]
    dx = 0.5/n
    src = get_particle_array(name='src', x=xr, y=yr, u=u, m=1.0, h=dx*hdx)
    return src, x, y


def moment_errors(kernel, hdx, equations=None):
    interp = None
    mom = []
    N = [10, 20, 40, 80, 160, 200, 300, 400]
    for n in N:
        src, x, y = setup_for_moment(hdx=hdx, n=n)
        pas = [src]
        if interp is None:
            interp = Interpolator(
                pas, x=x, y=y, kernel=kernel,
                equations=equations
            )
        else:
            interp.update_particle_arrays(pas)
            interp.set_interpolation_points(x, y)
        u = interp.interpolate('u')
        dx = 1.0
        old_mom = compute_moments(src.x, src.y, src.u, dx)
        new_mom = compute_moments(x, y, u, dx)
        mom_change = diff_mom(new_mom, old_mom)
        mom.append(mom_change)

    moment = defaultdict(list)
    for m in mom:
        for k in m:
            moment[k].append(m[k])
    moment = {k: np.asarray(moment[k]) for k in moment}

    N = np.asarray(N)

    return dict(sizes=N, moment=moment)


def compare_kernels():
    l2_eqs = [Lambda2(dest='interpolate', sources=['src'])]
    l3_eqs = [Lambda3(dest='interpolate', sources=['src'])]
    m4_eqs = [M4(dest='interpolate', sources=['src'])]
    from tsph_with_pst import CopyPropsToGhost
    m4_eqs = [
        Group(equations=[
            SummationDensity(dest='src', sources=['src'])
        ]),
        Group(equations=[
            CopyPropsToGhost(dest='src', sources=None)
        ], real=False),
        Group(equations=[
            M4(dest='interpolate', sources=['src'], hdx=app.hdx)
        ])
    ]
    cases = {
        'Gaussian': dict(kernel=Gaussian(dim=2), hdx=1.2),
        'CubicSpline': dict(kernel=CubicSpline(dim=2), hdx=1.2),
        'QuinticSpline': dict(kernel=QuinticSpline(dim=2), hdx=1.0),
        'WendlandQuintic': dict(kernel=WendlandQuintic(dim=2), hdx=2.0),
        'WendlandQuinticC4': dict(kernel=WendlandQuinticC4(dim=2), hdx=2.0),
        'WendlandQuinticC6': dict(kernel=WendlandQuinticC6(dim=2), hdx=2.0),
        'Lambda2': dict(
            kernel=CubicSpline(dim=2), hdx=1.5, equations=l2_eqs
        ),
        'Lambda3': dict(
            kernel=CubicSpline(dim=2), hdx=1.5, equations=l3_eqs
        ),
        'M4': dict(
            kernel=CubicSpline(dim=2), hdx=1.5, equations=m4_eqs
        ),
    }

    for name, case in cases.items():
        result = find_error(**case)
        case['result'] = result

    return cases


def plot_results(cases, err='l1', ylabel=r'$L_1$ error'):
    from matplotlib import pyplot as plt

    for name, case in cases.items():
        r = case['result']
        n = r['sizes']
        plt.loglog(n, r[err], label=name)
    plt.loglog(n, 5.0/(n*n), 'k:', label=r'$O(h^2)$')
    plt.legend()
    plt.xlabel('N')
    plt.ylabel(ylabel)
    plt.grid()


class DoubleDerivativeApprox(Equation):
    '''
    [1]S. P. Korzilius, W. H. A. Schilders, and M. J. H. Anthonissen, “An
    Improved CSPM Approach for Accurate Second-Derivative Approximations with
    SPH,” JAMP, vol. 05, no. 01, pp. 168–184, 2017, doi:
    10.4236/jamp.2017.51017.  
    '''

    def __init__(self, dest, sources, nu, rho0, dim=2):
        r"""
        Parameters
        ----------
        nu : float
            kinematic viscosity
        """
        self.dim = dim
        self.nu = nu
        self.rho0 = rho0
        super(DoubleDerivativeApprox, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def _get_helpers_(self):
        return [qs_dwdq2, qs_dwdq]

    def loop(self, d_idx, d_au, d_av, d_aw, XIJ, HIJ, RIJ, s_idx, s_m, s_rho, d_u, d_v, d_w, s_u, s_v, s_w):
        i = declare('int')
        L = declare('matrix(9)')
        d2wij = declare('matrix(3)')

        qs_dwdq2(XIJ, RIJ, HIJ, d2wij)

        vj = s_m[s_idx] / s_rho[s_idx] * self.nu
        uji = (s_u[s_idx] - d_u[d_idx]) * vj
        vji = (s_v[s_idx] - d_v[d_idx]) * vj
        wji = (s_w[s_idx] - d_w[d_idx]) * vj

        d_au[d_idx] += uji * (d2wij[0] + d2wij[2])
        d_av[d_idx] += vji * (d2wij[0] + d2wij[2])
        d_aw[d_idx] += wji * (d2wij[0] + d2wij[2])

class RemeshScheme(Scheme):
    def __init__(self, fluids, solids, dim, rho0, c0, h0, hdx, nu=0.0, gamma=7.0, kernel_corr=False, damp_pre=False):
        """Parameters
        ----------

        fluids: list
            List of names of fluid particle arrays.
        solids: list
            List of names of solid particle arrays (or boundaries).
        dim: int
            Dimensionality of the problem.
        rho0: float
            Reference density.
        c0: float
            Reference speed of sound.
        gamma: float
            Gamma for the equation of state.
        h0: float
            Reference smoothing length.
        hdx: float
            Ratio of h/dx.
        nu: float
            Dynamic Viscosity

        """
        self.fluids = fluids
        self.solids = solids
        self.solver = None
        self.rho0 = rho0
        self.c0 = c0
        self.gamma = gamma
        self.dim = dim
        self.h0 = h0
        self.hdx = hdx
        self.nu = nu
        self.kernel_corr = kernel_corr
        self.damp_pre = damp_pre


    def add_user_options(self, group):
        from pysph.sph.scheme import add_bool_argument
        add_bool_argument(
            group, "kernel-corr", dest="kernel_corr",
            help="Use this if kernel correction is required",
            default=None
        )
        add_bool_argument(
            group, "damp-pre", dest="damp_pre",
            help="if True then apply pressure damping",
            default=None
        )

    def consume_user_options(self, options):
        vars = ["kernel_corr", "damp_pre"]

        data = dict((var, self._smart_getattr(options, var))
                    for var in vars)
        self.configure(**data)

    def configure_solver(self, kernel=None, integrator_cls=None,
                         extra_steppers=None, **kw):
        from pysph.base.kernels import CubicSpline
        from tsph_with_pst import RK2Stepper, RK2Integrator
        if kernel is None:
            kernel = CubicSpline(dim=self.dim)

        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        cls = integrator_cls if integrator_cls is not None else RK2Integrator
        step_cls = RK2Stepper
        for name in self.fluids + self.solids:
            if name not in steppers:
                steppers[name] = step_cls()

        integrator = cls(**steppers)

        from pysph.solver.solver import Solver
        self.solver = Solver(
            dim=self.dim, integrator=integrator, kernel=kernel, **kw
        )

    def get_equations(self):
        from pysph.sph.equation import Group
        from pysph.sph.basic_equations import \
            (SummationDensity)
        from tsph_with_pst import (
            LinearEOS, CopyPropsToGhost, MomentumEquationSecondOrder,
            ContinuityEquation, VelocityGradient, CopyGradVToGhost, DivGrad,
            DensityGradient, CopyGradRhoToGhost, DensityDamping
        )
        from pysph.sph.wc.kernel_correction import GradientCorrectionPreStep, GradientCorrection
        

        equations = []
        g1 = []
        all = self.fluids + self.solids

        g0 = []
        for name in all:
            g0.append(SummationDensity(dest=name, sources=all))
            g0.append(LinearEOS(dest=name, sources=None, rho0=self.rho0, gamma=self.gamma))
        equations.append(Group(equations=g0))

        g1 = []
        for name in all:
            g1.append(CopyPropsToGhost(dest=name, sources=None))
        equations.append(Group(equations=g1, real=False))

        g1 = []
        for name in all:
            g1.append(GradientCorrectionPreStep(dest=name, sources=all, dim=self.dim))
        equations.append(Group(equations=g1))

        if self.nu > 1e-14:
            g2=[]
            for name in all:
                g2.append(GradientCorrection(dest=name, sources=all))
            for name in self.fluids:
                g2.append(VelocityGradient(dest=name, sources=all))
            equations.append(Group(equations=g2, real=False))

            g1 = []
            for name in all:
                g1.append(CopyGradVToGhost(dest=name, sources=None))
            equations.append(Group(equations=g1, real=False))
        elif self.damp_pre:
            g2=[]
            for name in all:
                g2.append(GradientCorrection(dest=name, sources=all))
            for name in self.fluids:
                g2.append(DensityGradient(dest=name, sources=all))
            equations.append(Group(equations=g2))

            g1 = []
            for name in all:
                g1.append(CopyGradRhoToGhost(dest=name, sources=None))
            equations.append(Group(equations=g1, real=False))

        g3 = []
        for name in all:
            g3.append(GradientCorrection(dest=name, sources=all))
        for name in all:
            g3.append(MomentumEquationSecondOrder(dest=name, sources=all))
            g3.append(ContinuityEquation(dest=name, sources=all))
            # g3.append(DoubleDerivativeApprox(dest=name, sources=all, nu=self.nu, rho0=self.rho0))
            if self.damp_pre:
                g3.append(DensityDamping(dest=name, sources=all, gamma=0.1))
            if self.nu > 1e-14:
                g3.append(DivGrad(dest=name, sources=all, nu=self.nu, rho0=self.rho0))

        equations.append(Group(equations=g3))

        return equations

    def setup_properties(self, particles, clean=True):
        from pysph.base.utils import get_particle_array_wcsph
        dummy = get_particle_array_wcsph(name='junk')
        props = list(dummy.properties.keys()) + [
            'V0'
        ]
        props += ['vmax', {'name': 'dpos', 'stride': 3}, {'name': 'gradrc', 'stride': 3}, {'name': 'gradp', 'stride': 3}, 'ki', 'ki0', 'rhoc', 'rhoc0', 'ap', 'p0']
        props += ['xi', 'yi', 'zi', 'ui', 'vi', 'wi', 'rhoi']
        output_props = ['x', 'y', 'z', 'u', 'v', 'w', 'rho', 'm', 'h',
                        'pid', 'gid', 'tag', 'p', 'rhoc']
        if self.kernel_corr:
            delta_sph_props = [
                {'name': 'm_mat', 'stride': 9},
                {'name': 'gradv', 'stride': 9},
            ]
            props += delta_sph_props

        for pa in particles:
            self._ensure_properties(pa, props, clean)
            pa.set_output_arrays(output_props)

