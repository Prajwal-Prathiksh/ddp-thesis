from numpy.lib.npyio import save
from code.mms_data.base import (
    momentum_eq, continuity, save_latex, save_formula, vec_grad, pressure_evol)
import sympy as sp
import os
import yaml

x, y, z, us, vs, ps, rhocs, t = sp.symbols('x, y, z, u, v, p, rhoc, t')
rhoc0, c0 = sp.symbols('rhoc0, c0')
rho0, rho = sp.symbols('rho0, rho')
PI = sp.pi
COS = sp.cos
SIN = sp.sin
EXP = sp.exp
SQRT = sp.sqrt
ATAN = sp.atan2

yaml_dict = {"tex":{}, "py":{}}


def compute_source(u, v, p, rhoc, mms, nu=0.0):
    su = momentum_eq([u, v, 0], rhoc, p, rhocs, nu=nu, comp=0)
    sv = momentum_eq([u, v, 0], rhoc, p, rhocs, nu=nu, comp=1)
    srho = continuity(rhoc, [u, v, 0], rhocs)
    spp = pressure_evol(p, [u, v, 0], rhocs)
    gradv = vec_grad([u, v, 0])

    save_latex(u, v, p, rhoc, su, sv, srho, spp, mms, yaml_dict)
    save_formula(u, v, p, rhoc, su, sv, srho, gradv, spp, mms, yaml_dict)


def mms_pres_d1():
    # works for mms do not change ####
    print('creating mms_pres_d1')
    u = SIN(2*PI*x) * COS(2*PI*y)*(y-1)
    v = -COS(2*PI*x) * SIN(2*PI*y)*(y-1)
    p = (COS(4*PI*x) + x**2)
    rhoc = p/c0**2 + rhoc0

    compute_source(u, v, p, rhoc, 'mms_pres_d1')


def mms_pres_d5():
    # works but has a limiting error
    print('creating mms_pres_d5')
    u = SIN(2*PI*x) * COS(2*PI*y) * (y-1)
    v = -COS(2*PI*x) * SIN(2*PI*y) * (y-1)
    p = ATAN((y-0.5)**2, (x-0.5)**2)
    rhoc = p/c0**2 + rhoc0

    compute_source(u, v, p, rhoc, 'mms_pres_d5')


def mms_noslip_d1():
    # works for MS DO NOT change
    print('creating mms_noslip_d1')
    u = EXP(-10*t) * (1-y)**2*SIN(2*PI*x) * COS(2*PI*y)
    v = -EXP(-10*t) * (1-y)**2*COS(2*PI*x) * SIN(2*PI*y)
    p = EXP(-10*t) * (COS(4*PI*x) + COS(4*PI*y))
    rhoc = p/c0**2 + rhoc0

    compute_source(u, v, p, rhoc, 'mms_noslip_d1', nu=0.25)


def mms_noslip_d5():
    # have slightly higher error
    print('creating mms_noslip_d5')
    r2 = (x-0.5)**2 + (y-0.5)**2
    u = EXP(-10*t) * (0.0625 - r2)*SIN(2*PI*r2)
    v = -EXP(-10*t) * (0.0625 - r2)*COS(2*PI*r2)
    p = EXP(-10*t) * (COS(4*PI*x) + COS(4*PI*y))
    rhoc = p/c0**2 + rhoc0

    compute_source(u, v, p, rhoc, 'mms_noslip_d5', nu=0.25)


def mms_noslip_d6():
    # have slightly higher error
    print('creating mms_noslip_d6')
    r2 = (x-0.5)**2 + (y-0.5)**2
    u = EXP(-10*t) * (0.25 - r2)*SIN(2*PI*r2)
    v = -EXP(-10*t) * (0.25 - r2)*COS(2*PI*r2)
    p = EXP(-10*t) * (COS(4*PI*x) + COS(4*PI*y))
    rhoc = p/c0**2 + rhoc0

    compute_source(u, v, p, rhoc, 'mms_noslip_d6', nu=0.25)


def mms_slip_d1():
    print('creating mms_slip_d1')
    u = SIN(2*PI*x) * COS(2*PI*y)*(y-1) + 1
    v = ((y-1)**2)*SIN(2*PI*y)
    p = (COS(4*PI*x) + COS(4*PI*y))
    rhoc = p/c0**2 + rhoc0

    compute_source(u, v, p, rhoc, 'mms_slip_d1')


def mms_slip_d5():
    print('creating mms_slip_d5')
    u = SIN(2*PI*x) * COS(2*PI*y)*(y-0.5)
    v = -SIN(2*PI*x) * COS(2*PI*y)*(x-0.5)
    p = (COS(4*PI*x) + COS(4*PI*y))
    rhoc = p/c0**2 + rhoc0

    compute_source(u, v, p, rhoc, 'mms_slip_d5')


def mms_io_vel():
    # note running with c0=40
    print('creating mms_io_vel')
    u = EXP(-10*t) * COS(2*PI*y) * y*(y-1) + 1
    v = -EXP(-10*t) * x**2*(x-1)**2 * SIN(2*PI*y)
    p =  EXP(-10*t) * (COS(4*PI*x) + COS(4*PI*y))
    rhoc = p/c0**2 + rhoc0

    compute_source(u, v, p, rhoc, 'mms_io_vel', nu=0.25)

def mms_out_vel_wave():
    # note running with c0=40
    print('creating mms_out_vel_wave')
    u =  (x-1)**2 * EXP(-200*(x-.9-40*t)**2) * COS(2*PI*y) * y*(y-1) + 1
    v =  0.0
    p =  (COS(4*PI*x) + COS(4*PI*y))
    rhoc = p/c0**2 + rhoc0

    compute_source(u, v, p, rhoc, 'mms_out_vel_wave', nu=0.0)


def mms_in_vel_wave():
    # note running with c0=40
    print('creating mms_in_vel_wave')
    u =  x**2 * EXP(-200*(x-0.1+40*t)**2) * COS(2*PI*y) * y*(y-1) + 1
    v = 0.0
    p =  (COS(4*PI*x) + COS(4*PI*y))
    rhoc = p/c0**2 + rhoc0

    compute_source(u, v, p, rhoc, 'mms_in_vel_wave', nu=0.0)


def mms_io_pres():
    # note running with c0=40
    print('creating mms_io_pres')
    u = EXP(-10*t) * COS(2*PI*y) * y*(y-1) + 1
    v = -EXP(-10*t) * (x-1)*x * SIN(2*PI*y)
    p =  EXP(-10*t) * y*(y-1) * COS(2*PI*y)
    rhoc = p/c0**2 + rhoc0

    compute_source(u, v, p, rhoc, 'mms_io_pres', nu=0.25)

def mms_inlet_pres():
    # note running with c0=40
    print('creating mms_inlet_pres')
    u =  COS(2*PI*y) * y*(y-1) + 1
    v =  0.0
    p =  x**2 * EXP(-200*(x-0.1+40*t)**2) * COS(2*PI*y)
    rhoc = p/c0**2 + rhoc0

    compute_source(u, v, p, rhoc, 'mms_inlet_pres', nu=0.0)


def mms_outlet_pres():
    # note running with c0=40
    print('creating mms_outlet_pres')
    u =  COS(2*PI*y) * y*(y-1) + 1
    v = 0.0
    p =  (x-1)**2 * EXP(-200*(x-0.9-40*t)**2) * COS(2*PI*y)
    rhoc = p/c0**2 + rhoc0

    compute_source(u, v, p, rhoc, 'mms_outlet_pres', nu=0.0)


mms_pres_d1()
mms_pres_d5()
mms_noslip_d1()
mms_noslip_d5()
mms_noslip_d6()
mms_slip_d1()
mms_slip_d5()
mms_io_vel()
mms_io_pres()
mms_in_vel_wave()
mms_out_vel_wave()
mms_inlet_pres()
mms_outlet_pres()

folder = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(folder, 'mms.yaml')
fp = open(filename, 'w')
yaml.dump(yaml_dict, fp)
fp.close()