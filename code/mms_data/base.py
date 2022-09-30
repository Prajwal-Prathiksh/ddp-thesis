from numpy.lib.function_base import vectorize
import sympy as sp
import os


x, y, z, us, vs, ws, ps, t, h = sp.symbols('x, y, z, u, v, w, p, t, h')


def grad(pres):
    return [sp.diff(pres, x), sp.diff(pres, y), sp.diff(pres, z)]


def vec_grad(vel):
    return [
        sp.diff(vel[0], x),
        sp.diff(vel[0], y),
        sp.diff(vel[0], z),
        sp.diff(vel[1], x),
        sp.diff(vel[1], y),
        sp.diff(vel[1], z),
        sp.diff(vel[2], x),
        sp.diff(vel[2], y),
        sp.diff(vel[2], z)
    ]


def div(vel):
    return (sp.diff(vel[0], x) + sp.diff(vel[1], y) + sp.diff(vel[2], z))


def laplace(vel):
    gradv = vec_grad(vel)
    return [div(gradv[0:3]), div(gradv[3:6]), div(gradv[6:9])]

def laplace_scal(f):
    gradf = grad(f)
    return div(gradf)

def dot(x, y):
    return x[0]*y[0] + x[1]*y[1] + x[2]*y[2] 


def vec_mat_mul(A, x):
    x0 = A[0]*x[0] + A[1]*x[1] + A[2]*x[2]
    x1 = A[3]*x[0] + A[4]*x[1] + A[5]*x[2]
    x2 = A[6]*x[0] + A[7]*x[1] + A[8]*x[2]
    return [x0, x1, x2]


def continuity(rhoc, vel, rhocs, delta_coeff=0.0):
    srho = sp.diff(rhoc, t) + dot(grad(rhoc), [us, vs, ws]) + rhocs * div(vel) - delta_coeff*h * laplace_scal(rhoc)
    return srho


def pressure_evol(p, vel, rhocs, delta_coeff=0.0, c0=20):
    spp = sp.diff(p, t) + dot(grad(p), [us, vs, ws]) + rhocs * c0**2 * div(vel) - delta_coeff*h*c0/8 * laplace_scal(p)
    return spp


def momentum_eq(vel, rhoc, p, rhocs, nu=0.0, comp=0):
    adv = vec_mat_mul(vec_grad(vel), [us, vs, ws]) 
    s_u = sp.diff(vel[comp], t) + adv[comp] + grad(p)[comp]/rhocs - nu*laplace(vel)[comp]
    return s_u


def save_formula(u, v, p, rhoc, su, sv, srho, gradv, spp,key, yaml_dict, sw=0.0, w=0.0):
    yaml_dict['py'][key] = {}
    yaml_dict['py'][key]['u'] = str(u)
    yaml_dict['py'][key]['v'] = str(v)
    yaml_dict['py'][key]['w'] = str(w)
    yaml_dict['py'][key]['p'] = str(p)
    yaml_dict['py'][key]['rhoc'] = str(rhoc)
    yaml_dict['py'][key]['su'] = str(su)
    yaml_dict['py'][key]['sv'] = str(sv)
    yaml_dict['py'][key]['sw'] = str(sw)
    yaml_dict['py'][key]['srho'] = str(srho)
    yaml_dict['py'][key]['spp'] = str(spp)
    for i in range(9):
        yaml_dict['py'][key]['gradv%d'%i] = str(gradv[i])


def save_latex(u, v, p, rhoc, su, sv, srho, spp, key, yaml_dict, sw=0.0, w=0.0):
    yaml_dict['tex'][key] = {}
    yaml_dict['tex'][key]['u'] = str(sp.printing.latex(u))
    yaml_dict['tex'][key]['v'] = str(sp.printing.latex(v))
    yaml_dict['tex'][key]['w'] = str(sp.printing.latex(w))
    yaml_dict['tex'][key]['p'] = str(sp.printing.latex(p))
    yaml_dict['tex'][key]['rhoc'] = str(sp.printing.latex(rhoc))
    yaml_dict['tex'][key]['su'] = str(sp.printing.latex(su))
    yaml_dict['tex'][key]['sv'] = str(sp.printing.latex(sv))
    yaml_dict['tex'][key]['sw'] = str(sp.printing.latex(sw))
    yaml_dict['tex'][key]['srho'] = str(sp.printing.latex(srho))
    yaml_dict['tex'][key]['spp'] = str(sp.printing.latex(spp))