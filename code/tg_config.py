# configurations for taylor green problem
###
from numpy import linspace, pi, sin, cos, exp, sqrt

from pysph.sph.equation import Group, Equation
from pysph.sph.scheme import TVFScheme, WCSPHScheme, SchemeChooser
from pysph.sph.wc.edac import EDACScheme
from pysph.sph.iisph import IISPHScheme

from pysph.sph.wc.kernel_correction import (
    GradientCorrectionPreStep, GradientCorrection,
    MixedKernelCorrectionPreStep, MixedGradientCorrection
)
from pysph.sph.wc.crksph import CRKSPHPreStep, CRKSPH, CRKSPHScheme
from pysph.sph.wc.gtvf import GTVFScheme
from pysph.sph.wc.pcisph import PCISPHScheme
from pysph.sph.isph.sisph import SISPHScheme
from pysph.sph.isph.isph import ISPHScheme

from pst import ShiftPositions
from delta_plus import DeltaPlusSPHScheme

from tsph_with_pst import TSPHScheme
from tsph_dsph import TSPHWithDSPHScheme
from tisph import SummationDensity, TISPHScheme
from ewcsph import EWCSPHScheme
from remesh import RemeshScheme

from monaghan2017 import Monaghan2017Scheme
from okra2022 import Okra2022Scheme

from compyle.api import declare


class EvaluateTorque(Equation):
    def initialize(self, d_idx, d_T, d_au, d_av, d_x, d_y, d_m, d_tag):
        au = d_au[d_idx]
        av = d_av[d_idx]
        x = d_x[d_idx]
        y = d_y[d_idx]
        m = d_m[d_idx]
        d_T[d_idx] = m * (av*x - au*y)


class CopyPropsToGhostEDAC(Equation):
    def initialize(self, d_idx, d_tag, d_gid, d_p, d_pavg, d_rho):
        idx = declare('int')
        if d_tag[d_idx] == 2:
            idx = d_gid[d_idx]
            d_p[d_idx] = d_p[idx]
            d_pavg[d_idx] = d_pavg[idx]
            d_rho[d_idx] = d_rho[idx]


class CopyPropsToGhostEDACSolid(Equation):
    def initialize(self, d_idx, d_tag, d_gid, d_p, d_rho, d_ug, d_vg):
        idx = declare('int')
        if d_tag[d_idx] == 2:
            idx = d_gid[d_idx]
            d_p[d_idx] = d_p[idx]
            d_rho[d_idx] = d_rho[idx]
            d_ug[d_idx] = d_ug[idx]
            d_vg[d_idx] = d_vg[idx]


def exact_solution(U, b, t, x, y):
    factor = U * exp(b*t)

    u = -cos(2*pi*x) * sin(2*pi*y)
    v = sin(2*pi*x) * cos(2*pi*y)
    p = -0.25 * (cos(4*pi*x) + cos(4*pi*y))

    return factor * u, factor * v, factor * factor * p


def configure_scheme(app, p0, gx=0.0):
    from pysph.base.kernels import QuinticSpline
    scheme = app.scheme
    h0 = app.hdx * app.dx
    pfreq = 100
    kernel = QuinticSpline(dim=2)
    if app.options.scheme == 'tvf':
        scheme.configure(pb=app.options.pb_factor * p0, nu=app.nu, h0=h0, gx=gx)
    elif app.options.scheme == 'tsph':
        scheme.configure(hdx=app.hdx, nu=app.nu, h0=h0, gx=gx, periodic=app.no_periodic)
    elif app.options.scheme == 'rsph':
        scheme.configure(hdx=app.hdx, nu=app.nu, h0=h0)
    elif app.options.scheme == 'ewcsph':
        scheme.configure(hdx=app.hdx, nu=app.nu, h0=h0, periodic=app.no_periodic)
    elif app.options.scheme == 'tdsph':
        scheme.configure(hdx=app.hdx, nu=app.nu, h0=h0)
    elif app.options.scheme == 'wcsph':
        if scheme.scheme.summation_density:
            from pysph.sph.integrator_step import WCSPHStep
            from patch import initialize, stage1, stage2
            WCSPHStep.initialize = initialize
            WCSPHStep.stage1 = stage1
            WCSPHStep.stage2 = stage2
        scheme.configure(hdx=app.hdx, nu=app.nu, h0=h0)
    elif app.options.scheme == 'edac':
        scheme.configure(h=h0, nu=app.nu, pb=app.options.pb_factor * p0, gx=gx)
    elif app.options.scheme.endswith('isph'):
        pfreq = 10
        scheme.configure(nu=app.nu)
    elif app.options.scheme == 'crksph':
        scheme.configure(h0=h0, nu=app.nu)
    elif app.options.scheme == 'gtvf':
        scheme.configure(pref=p0, nu=app.nu, h0=h0)
    elif app.options.scheme == 'delta_plus':
        correction = app.kernel_correction
        if correction == '':
            correction = 'gradient'
        scheme.configure(hdx=app.hdx, nu=app.nu, h0=h0, correction=correction)
    elif app.options.scheme == 'mon2017':
        scheme.configure(h0=h0)
    elif app.options.scheme == 'ok2022':
        scheme.configure(nu=app.nu, dx=app.dx, h0=h0)

    tf = app.options.final_time
    if tf is None:
        tf = app.tf
    times = linspace(0, tf, 50)
    scheme.configure_solver(kernel=kernel, tf=app.tf, dt=app.dt,
                            output_at_times=times, pfreq=100000000)


def create_scheme(rho0, c0, p0, solids=[]):
    h0 = None
    hdx = None
    wcsph = WCSPHScheme(
        ['fluid'], solids, dim=2, rho0=rho0, c0=c0, h0=h0,
        hdx=hdx, nu=None, gamma=7.0, alpha=0.0, beta=0.0
    )
    delta_plus = DeltaPlusSPHScheme(
        ['fluid'], solids, dim=2, rho0=rho0, c0=c0, h0=h0,
        hdx=hdx, nu=None, gamma=7.0, correction=None
    )
    tvf = TVFScheme(
        ['fluid'], solids, dim=2, rho0=rho0, c0=c0, nu=None,
        p0=p0, pb=None, h0=h0
    )
    edac = EDACScheme(
        ['fluid'], solids, dim=2, rho0=rho0, c0=c0, nu=None,
        pb=p0, h=h0
    )
    iisph = IISPHScheme(
        fluids=['fluid'], solids=solids, dim=2, nu=None,
        rho0=rho0, has_ghosts=True
    )
    crksph = CRKSPHScheme(
        fluids=['fluid'], dim=2, nu=None,
        rho0=rho0, h0=h0, c0=c0, p0=0.0
    )
    gtvf = GTVFScheme(
        fluids=['fluid'], solids=solids, dim=2, rho0=rho0, c0=c0,
        nu=None, h0=None, pref=None
    )
    pcisph = PCISPHScheme(
        fluids=['fluid'], dim=2, rho0=rho0, nu=None
    )
    sisph = SISPHScheme(
        fluids=['fluid'], solids=solids, dim=2, nu=None, rho0=rho0,
        c0=c0, alpha=0.0, has_ghosts=True, pref=p0,
        rho_cutoff=0.2, internal_flow=True, gtvf=True
    )
    isph = ISPHScheme(
        fluids=['fluid'], solids=solids, dim=2, nu=None, rho0=rho0, c0=c0,
        alpha=0.0
    )
    tsph = TSPHScheme(
        ['fluid'], solids, dim=2, rho0=rho0, c0=c0, h0=h0,
        hdx=hdx, nu=None, gamma=7.0, kernel_corr=True
    )
    ewcsph = EWCSPHScheme(
        ['fluid'], solids, dim=2, rho0=rho0, c0=c0, h0=h0,
        hdx=hdx, nu=None, gamma=7.0, kernel_corr=True
    )
    tdsph = TSPHWithDSPHScheme(
        ['fluid'], solids, dim=2, rho0=rho0, c0=c0, h0=h0,
        hdx=hdx, nu=None, gamma=7.0, kernel_corr=True
    )
    rsph = RemeshScheme(
        ['fluid'], solids, dim=2, rho0=rho0, c0=c0, h0=h0,
        hdx=hdx, nu=None, gamma=7.0, kernel_corr=True
    )
    tisph = TISPHScheme(
        fluids=['fluid'], solids=solids, dim=2, nu=None, rho0=rho0,
        c0=c0, alpha=0.0, has_ghosts=True, pref=p0,
        rho_cutoff=0.2, internal_flow=True, gtvf=True
    )
    mon2017 = Monaghan2017Scheme(
        fluids=['fluid'], solids=[], dim=2, rho0=rho0, c0=c0, h0=h0
    )
    ok2022 = Okra2022Scheme(
        fluids=['fluid'], solids=[], dim=2, rho0=rho0, p0=p0, c0=c0,
        nu=None, dx=None, h0=h0
    )
    s = SchemeChooser(
        default='tvf', wcsph=wcsph, tvf=tvf, edac=edac, iisph=iisph,
        crksph=crksph, gtvf=gtvf, pcisph=pcisph, sisph=sisph, isph=isph,
        delta_plus=delta_plus, tsph=tsph, tdsph=tdsph, tisph=tisph, ewcsph=ewcsph, rsph=rsph, mon2017=mon2017, ok2022=ok2022
    )
    return s


def create_equation(app, solids=[]):
    eqns = app.scheme.get_equations()
    # This tolerance needs to be fixed.
    tol = 0.5
    if app.kernel_correction == 'gradient':
        cls1 = GradientCorrectionPreStep
        cls2 = GradientCorrection
    elif app.kernel_correction == 'mixed':
        cls1 = MixedKernelCorrectionPreStep
        cls2 = MixedGradientCorrection
    elif app.kernel_correction == 'crksph':
        cls1 = CRKSPHPreStep
        cls2 = CRKSPH
    elif app.kernel_correction == 'kgf':
        from kgf_sph import KGFPreStep, KGFCorrection
        cls1 = KGFPreStep
        cls2 = KGFCorrection
    elif app.kernel_correction == 'order1':
        from kgf_sph import FirstOrderCorrection, FirstOrderPreStep
        cls1 = FirstOrderPreStep
        cls2 = FirstOrderCorrection

    all = ['fluid'] + solids
    if app.kernel_correction:
        g1 = Group(equations=[cls1('fluid', all, dim=2)], real=False)
        eq1 = cls1('fluid', all, dim=2)
        eq2 = cls2(dest='fluid', sources=all, dim=2, tol=tol)

        if app.options.scheme == 'wcsph':
            if app.scheme.scheme.summation_density:
                eqns[1].equations.insert(0, eq1)
                eqns[2].equations.insert(0, eq2)
            elif app.scheme.scheme.delta_sph:
                eqns[-1].equations.insert(0, eq2)
            else:
                eqns.insert(1, g1)
                eqns[2].equations.insert(0, eq2)
        elif app.options.scheme == 'delta_plus':
            if len(eqns[-1].equations) == 1:
                eqns[-2].equations.insert(0, eq2)
            else:
                eqns[-1].equations.insert(0, eq2)
        elif app.options.scheme == 'tvf':
            eqns[1].equations.append(g1.equations[0])
            eqns[2].equations.insert(0, eq2)
        elif app.options.scheme == 'gtvf':
            grp = eqns.groups[1]
            grp[-2].equations.append(g1.equations[0])
            grp[-1].equations.insert(0, eq2)
        elif app.options.scheme == 'edac':
            eqns.insert(1, g1)
            eqns[2].equations.insert(0, eq2)
        elif app.options.scheme == 'iisph':
            eqns.insert(1, g1)
            eq1 = cls2(dest='fluid', sources=all, dim=2, tol=tol)
            eqns[2].equations.insert(0, eq1)
            eq2 = cls2(dest='fluid', sources=all, dim=2, tol=tol)
            eqns[3].equations.insert(0, eq2)
            eq3 = cls2(dest='fluid', sources=all, dim=2, tol=tol)
            eqns[4].equations[0].equations.insert(0, eq3)
            eq4 = cls2(dest='fluid', sources=all, dim=2, tol=tol)
            eqns[4].equations[2].equations.insert(0, eq4)
            eq2 = cls2(dest='fluid', sources=all, dim=2, tol=tol)
            eqns[5].equations.insert(0, eq2)
        elif app.options.scheme == 'isph':
            stg1 = eqns.groups[0]
            stg1.insert(0, g1)
            stg1[1].equations.insert(0, eq2)
            g1 = Group(equations=[cls1('fluid', all, dim=2)], real=False)
            stg2 = eqns.groups[1]
            stg2.insert(0, g1)
            eq1 = cls2(dest='fluid', sources=all, dim=2, tol=tol)
            stg2[1].equations.insert(0, eq1)
            eq1 = cls2(dest='fluid', sources=all, dim=2, tol=tol)
            stg2[2].equations.insert(0, eq1)
            eq1 = cls2(dest='fluid', sources=all, dim=2, tol=tol)
            stg2[3].equations.insert(0, eq1)
        elif app.options.scheme == 'sisph':
            stg1 = eqns.groups[0]
            stg1.insert(1, g1)
            stg1[2].equations.insert(0, eq1)
            g1 = Group(equations=[cls1('fluid', all, dim=2)], real=False)
            stg2 = eqns.groups[1]
            stg2.insert(1, g1)
            eq1 = cls2(dest='fluid', sources=all, dim=2, tol=tol)
            stg2[2].equations.insert(0, eq1)
            eq1 = cls2(dest='fluid', sources=all, dim=2, tol=tol)
            print(stg2[3].equations[0])
            stg2[3].equations[1].equations.insert(0, eq1)
            eq1 = cls2(dest='fluid', sources=all, dim=2, tol=tol)
            stg2[5].equations.insert(0, eq1)


    if app.options.shift_freq > 0:
        if app.options.scheme == 'wcsph':
            if app.scheme.scheme.summation_density:
                from patch import XSPHCorrectionCopy
                eqns[-1].equations[-1] = XSPHCorrectionCopy(dest='fluid',
                                                            sources=None)

    if app.options.no_periodic:
        torque_eqn = Group(equations=[
            EvaluateTorque(dest='fluid', sources=None),
            ], real=False)
        eqns.insert(len(eqns), torque_eqn)

    from tsph_with_pst import CopyPropsToGhost, CopyPropsToGhostWithSolidTVF
    if app.options.scheme == 'edac':
        if len(solids) > 0:
            eq1 = Group(
                equations=[
                    CopyPropsToGhostEDAC(dest='fluid', sources=None),
                    CopyPropsToGhostEDACSolid(dest='channel', sources=None)
                    ],
                real=False)
            eqns.insert(1, eq1)
        else:
            eq1 = Group(
                equations=[
                    CopyPropsToGhostEDAC(dest='fluid', sources=None)
                    ],
                real=False)
            eqns.insert(1, eq1)
    elif app.options.scheme == 'tvf':
        if len(solids) > 0:
            eq1 = Group(
                equations=[
                    CopyPropsToGhost(dest='fluid', sources=None),
                    CopyPropsToGhostWithSolidTVF(dest='channel', sources=None)
                    ],
                real=False)
            eqns.insert(3, eq1)
        else:
            eq1 = Group(equations=[CopyPropsToGhost(dest='fluid', sources=None)], real=False)
            eqns.insert(2, eq1)

    if hasattr(app.options, 'prob'):
        if app.options.prob == 'sl':
            schemes = ['edac', 'tvf', 'ewcsph', 'delta_plus']
            if app.options.scheme in schemes:
                from tsph_with_pst import VelocityGradient
                eq1 = VelocityGradient(dest='fluid', sources=all)
                g2 = eqns[1].equations
                g2.insert(0, eq1)

    print(eqns)
    return eqns


def configure_particles(app, fluid):
    import numpy as np
    # volume is set as dx^2
    if app.options.scheme == 'sisph':
        nfp = fluid.get_number_of_particles()
        fluid.gid[:] = np.arange(nfp)
        fluid.add_output_arrays(['gid'])
    if app.options.scheme == 'tvf':
        fluid.V[:] = 1. / app.volume
    if app.options.scheme == 'iisph':
        # These are needed to update the ghost particle properties.
        nfp = fluid.get_number_of_particles()
        fluid.orig_idx[:] = np.arange(nfp)
        fluid.add_output_arrays(['orig_idx'])

    corr = app.kernel_correction
    if corr in ['mixed', 'crksph']:
        fluid.add_property('cwij')
    if corr == 'mixed' or corr == 'gradient':
        fluid.add_property('m_mat', stride=9)
        fluid.add_property('dw_gamma', stride=3)
    if corr == 'kgf' or corr == 'order1':
        fluid.add_property('L', stride=16)
    elif corr == 'crksph':
        # fluid.e = fluid.p/((app.scheme.scheme.gamma-1)*app.rho)
        fluid.add_property('ai')
        fluid.add_property('V')
        fluid.add_property('gradbi', stride=4)
        for prop in ['gradai', 'bi']:
            fluid.add_property(prop, stride=2)

    if app.options.shift_freq > 0:
        fluid.add_constant('vmax', [0.0])
        fluid.add_property('dpos', stride=3)
        fluid.add_property('gradv', stride=9)

    if app.options.scheme == 'isph':
        gid = np.arange(fluid.get_number_of_particles(real=False))
        fluid.add_property('gid')
        fluid.gid[:] = gid[:]
        fluid.add_property('dpos', stride=3)
        fluid.add_property('gradv', stride=9)

    fluid.add_property('T')
    fluid.add_output_arrays(['au', 'av', 'aw', 'T'])


def create_tools(app):
    tools = []
    options = app.options
    if options.remesh > 0:
        from remesh import M4, M41
        if options.remesh_eq == 'm4':
            from tsph_with_pst import CopyPropsToGhost
            equations = [
                Group(equations=[
                    SummationDensity(dest='fluid', sources=['fluid'])
                ]),
                Group(equations=[
                    CopyPropsToGhost(dest='fluid', sources=None)
                ], real=False),
                Group(equations=[
                    # M4(dest='interpolate', sources=['fluid'], hdx=app.hdx)
                    M41(dest='interpolate', sources=['fluid'], hdx=app.hdx)
                ])
            ]
        else:
            equations = None
        from pysph.solver.tools import SimpleRemesher
        if options.scheme == 'wcsph' or options.scheme == 'crksph':
            props = ['u', 'v', 'au', 'av', 'ax', 'ay', 'arho']
        elif options.scheme == 'rsph':
            props = ['u', 'v', 'rhoc', 'p']
        elif options.scheme == 'pcisph':
            props = ['u', 'v', 'p']
        elif options.scheme == 'tvf':
            props = ['u', 'v', 'uhat', 'vhat', 'au', 'av', 'auhat', 'avhat']
        elif options.scheme == 'edac':
            if 'uhat' in app.particles[0].properties:
                props = [
                    'u', 'v', 'uhat', 'vhat', 'p', 'au', 'av', 'auhat',
                    'avhat', 'ap'
                ]
            else:
                props = ['u', 'v', 'p', 'au', 'av', 'ax', 'ay', 'ap']
        elif options.scheme == 'iisph' or options.scheme == 'isph':
            # The accelerations are not really needed since the current
            # stepper is a single stage stepper.
            props = ['u', 'v', 'p']
        elif options.scheme == 'gtvf':
            props = [
                'uhat', 'vhat', 'what', 'rho0', 'rhodiv', 'p0',
                'auhat', 'avhat', 'awhat', 'arho', 'arho0'
            ]

        remesher = SimpleRemesher(
            app, 'fluid', props=props,
            freq=app.options.remesh, equations=equations
        )
        tools.append(remesher)

    if options.shift_freq > 0:
        shift = ShiftPositions(
            app, ['fluid'], freq=app.options.shift_freq,
            shift_kind=app.options.shift_kind,
            correct_velocity=app.options.correct_vel,
            parameter=app.options.shift_parameter,
            hdx=app.hdx
        )
        tools.append(shift)

    return tools


def ramp(t_star):
    """
    Ramp function for the external forcing term in the
    Taylor-Green vortex problem.

    Parameters
    ----------
    t_star : float
        Dimensionless time.

    Returns
    -------
    float
        Value of the ramp function at time t_star.
    """
    if t_star>=0.0 and t_star<0.1:
        return t_star*10.
    elif t_star>=0.1 and t_star<0.9:
        return 1.
    elif t_star>=0.9 and t_star<=1.:
        return (1.-t_star)*10.
    else:
        return 0.


def ext_force_colagrossi2021(x, y, t, L=1., U=1.):
    """
    External forcing term for the Taylor-Green vortex problem.

    References
    ----------
        .. [Colagrossi2021] A. Colagrossi, “Smoothed particle hydrodynamics 
        method from a large eddy simulation perspective . Generalization to a 
        quasi-Lagrangian model Smoothed particle hydrodynamics method from a 
        large eddy simulation perspective . Generalization to a 
        quasi-Lagrangian model,” vol. 015102, no. December 2020, 2021,
        doi: 10.1063/5.0034568.

    Parameters
    ----------
    x, y : array_like
        Coordinates.
    t : float
        Time.
    L : float, optional
        Length of the domain. Default is 1.
    U : float, optional
        Maximum velocity of the flow. Default is 1.
    
    Returns
    -------
    fx, fy : array_like
        External forcing term in the x- and y-directions.
    """
    A = 1.3*U*U/L
    x_star = x/L
    y_star = y/L
    t_star = t*U/L
    eightpi = 8.*pi
    
    if ramp(t_star) > 0.:
        fx = sin(eightpi*x_star)*cos(eightpi*y_star)
        fy = -cos(eightpi*x_star)*sin(eightpi*y_star)
        return A*ramp(t_star)*fx, A*ramp(t_star)*fy
    else:
        return 0.*x_star, 0.*y_star


def ext_force_antuono2020(x, y, z, t, nu, L=1., U=1.):
    """
    External forcing term for the Triperiodic Beltrami flow problem.

    References
    ----------
        .. [Antuono2020] M. Antuono, “Tri-periodic fully three-dimensional 
        analytic solutions for the Navier-Stokes equations,” J. Fluid Mech., 
        vol. 890, 2020, doi: 10.1017/jfm.2020.126.

        .. [Colagrossi2021] A. Colagrossi, “Smoothed particle hydrodynamics 
        method from a large eddy simulation perspective . Generalization to a 
        quasi-Lagrangian model Smoothed particle hydrodynamics method from a 
        large eddy simulation perspective . Generalization to a 
        quasi-Lagrangian model,” vol. 015102, no. December 2020, 2021,
        doi: 10.1063/5.0034568.

    Parameters
    ----------
    x, y, z : array_like
        Coordinates.
    t : float
        Time.
    nu : float
        Kinematic viscosity.
    L : float, optional
        Length of the domain. Default is 1.
    U : float, optional
        Maximum velocity of the flow. Default is 1.
    
    Returns
    -------
    fx, fy, fz : array_like
        External forcing term in the x-, y-, and z-directions.
    """
    piby6, fivepiby6 = pi/6, 5*pi/6
    k = 2*pi/L
    kx, ky, kz = k*x, k*y, k*z
    t_star = t*U/L
    A = sqrt(32./27.)
    A = A*(-3.*nu*k**2)

    if ramp(t_star) > 0.:
        fx = sin(kx - fivepiby6)*cos(ky - piby6)*sin(kz) -\
            cos(kz - fivepiby6)*sin(kx - piby6)*sin(ky)
        fy = sin(ky - fivepiby6)*cos(kz - piby6)*sin(kx) -\
            cos(kx - fivepiby6)*sin(ky - piby6)*sin(kz)
        fz = sin(kz - fivepiby6)*cos(kx - piby6)*sin(ky) -\
            cos(ky - fivepiby6)*sin(kz - piby6)*sin(kx)
        return A*ramp(t_star)*fx, A*ramp(t_star)*fy, A*ramp(t_star)*fz
    else:
        return 0.*x, 0.*y, 0.*z


def prestep(app, solver):
    """
    Pre-step function for the Taylor-Green vortex problem.
    """
    if app.options.ext_forcing:
        pa = app.particles[0]
        x, y = pa.x, pa.y
        L, U = app.L, app.U
        dt, t = solver.dt, solver.t
        fx, fy = ext_force_colagrossi2021(x, y, t, L, U)
        pa.u[:] += fx*dt
        pa.v[:] += fy*dt
        
    if app.options.scheme == 'wcsph':
        if app.scheme.scheme.delta_sph:
            p = app.particles[0].p
            app.particles[0].p -= min(p)