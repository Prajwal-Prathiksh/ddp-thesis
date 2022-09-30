from math import exp, pi, sin, cos, sqrt
import numpy as np
from compyle.api import declare

'''
MMS for V shape boundary
'''

def get_props_p(x, y, z, t, c0):
    from numpy import sin, cos, exp, log, sqrt
    h = 0.25
    l = 1.0
    u, v, w, p = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
    cond0 = ( x < 0.25)
    u[cond0] = ((y[cond0] - l)**2*sin(2*pi*x[cond0])*cos(2*pi*y[cond0])) * np.ones_like(x[cond0])
    v[cond0] = ((y[cond0] - l)**2*sin(2*pi*y[cond0])*cos(2*pi*x[cond0])) * np.ones_like(x[cond0])
    w[cond0] = (0.0) * np.ones_like(x[cond0])
    p[cond0] = ((cos(4*pi*x[cond0]) + cos(4*pi*y[cond0]))*(y[cond0] - l)**2) * np.ones_like(x[cond0])

    cond0 = (x >= 0.25) & ( x < 0.5)
    u[cond0] = ((-(x[cond0] - 0.25) * h/0.25 + y[cond0] - l)**2*sin(2*pi*x[cond0])*cos(2*pi*y[cond0])) * np.ones_like(x[cond0])
    v[cond0] = ((-(x[cond0] - 0.25) * h/0.25 + y[cond0] - l)**2*sin(2*pi*y[cond0])*cos(2*pi*x[cond0])) * np.ones_like(x[cond0])
    w[cond0] = (0.0) * np.ones_like(x[cond0])
    p[cond0] = ((cos(4*pi*x[cond0]) + cos(4*pi*y[cond0]))*(-(x[cond0] - 0.25) * h/0.25 + y[cond0] - l)**2) * np.ones_like(x[cond0])

    cond0 = (x >= 0.5) & ( x < 0.75)
    u[cond0] = (((x[cond0] - 0.5) * h/0.25 + y[cond0] - l -h )**2*sin(2*pi*x[cond0])*cos(2*pi*y[cond0])) * np.ones_like(x[cond0])
    v[cond0] = (((x[cond0] - 0.5) * h/0.25 + y[cond0] - l -h )**2*sin(2*pi*y[cond0])*cos(2*pi*x[cond0])) * np.ones_like(x[cond0])
    w[cond0] = (0.0) * np.ones_like(x[cond0])
    p[cond0] = ((cos(4*pi*x[cond0]) + cos(4*pi*y[cond0]))*((x[cond0] - 0.5) * h/0.25 + y[cond0] - l -h )**2) * np.ones_like(x[cond0])

    cond0 = (x >= 0.75)
    u[cond0] = ((y[cond0] - l)**2*sin(2*pi*x[cond0])*cos(2*pi*y[cond0])) * np.ones_like(x[cond0])
    v[cond0] = ((y[cond0] - l)**2*sin(2*pi*y[cond0])*cos(2*pi*x[cond0])) * np.ones_like(x[cond0])
    w[cond0] = (0.0) * np.ones_like(x[cond0])
    p[cond0] = ((cos(4*pi*x[cond0]) + cos(4*pi*y[cond0]))*(y[cond0] - l)**2) * np.ones_like(x[cond0])

    rhoc = p/c0**2 + 1.0

    return u, v, w, rhoc, p


def get_props_u_slip(x, y, z, t, c0):
    from numpy import sin, cos, exp, log, sqrt
    h = 0.25
    l = 1.0
    u, v, w, p = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
    cond0 = ( x < 0.25)
    tx = np.ones_like(x[cond0]) * 1.0
    ty = np.ones_like(x[cond0]) * 0.0
    u[cond0] = (tx*((x[cond0]+1)**3 * (y[cond0]-l)**3)) * np.ones_like(x[cond0])
    v[cond0] = (ty*((x[cond0]+1)**3 * (y[cond0]-l)**3)) * np.ones_like(x[cond0])
    w[cond0] = (0.0) * np.ones_like(x[cond0])
    p[cond0] = ((cos(4*pi*x[cond0]) + cos(4*pi*y[cond0]))*(y[cond0] - l)**2) * np.ones_like(x[cond0])

    cond0 = (x >= 0.25) & ( x < 0.5)
    tx = np.ones_like(x[cond0]) * 1/sqrt(2)
    ty = np.ones_like(x[cond0]) * 1/sqrt(2)
    u[cond0] = (tx*((x[cond0]+1)**3 * (-(x[cond0] - 0.25) * h/0.25 + y[cond0] - l)**3)) * np.ones_like(x[cond0])
    v[cond0] = (ty*((x[cond0]+1)**3 * (-(x[cond0] - 0.25) * h/0.25 + y[cond0] - l)**3)) * np.ones_like(x[cond0])
    w[cond0] = (0.0) * np.ones_like(x[cond0])
    p[cond0] = ((cos(4*pi*x[cond0]) + cos(4*pi*y[cond0]))*(-(x[cond0] - 0.25) * h/0.25 + y[cond0] - l)**2) * np.ones_like(x[cond0])

    cond0 = (x >= 0.5) & ( x < 0.75)
    tx = np.ones_like(x[cond0]) * 1/sqrt(2)
    ty = np.ones_like(x[cond0]) * -1/sqrt(2)
    u[cond0] = (tx*((x[cond0]+1)**3 * ((x[cond0] - 0.5) * h/0.25 + y[cond0] - l -h )**3)) * np.ones_like(x[cond0])
    v[cond0] = (ty*((x[cond0]+1)**3 * ((x[cond0] - 0.5) * h/0.25 + y[cond0] - l -h )**3)) * np.ones_like(x[cond0])
    w[cond0] = (0.0) * np.ones_like(x[cond0])
    p[cond0] = ((cos(4*pi*x[cond0]) + cos(4*pi*y[cond0]))*((x[cond0] - 0.5) * h/0.25 + y[cond0] - l -h )**2) * np.ones_like(x[cond0])

    cond0 = (x >= 0.75)
    tx = np.ones_like(x[cond0]) * 1.0
    ty = np.ones_like(x[cond0]) * 0.0
    u[cond0] = (tx*((x[cond0]+1)**3 * y[cond0]**3)) * np.ones_like(x[cond0])
    v[cond0] = (ty*((x[cond0]+1)**3 * y[cond0]**3)) * np.ones_like(x[cond0])
    w[cond0] = (0.0) * np.ones_like(x[cond0])
    p[cond0] = ((cos(4*pi*x[cond0]) + cos(4*pi*y[cond0]))*(y[cond0] - l)**2) * np.ones_like(x[cond0])

    rhoc = p/c0**2 + 1.0

    return u, v, w, rhoc, p
