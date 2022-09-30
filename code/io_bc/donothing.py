'''C. E. Alvarado-Rodríguez, J. Klapp, L. D. G. Sigalotti, J. M.
Domínguez, and E. de la Cruz Sánchez, “Nonreflecting outlet boundary
conditions for incompressible flows using SPH,” Computers & Fluids, vol.
159, pp. 177–188, Dec. 2017, doi: 10.1016/j.compfluid.2017.09.020.
'''

from pysph.sph.equation import Equation
from compyle.api import declare
from pysph.sph.wc.edac import SourceNumberDensity
from solid_bc.takeda import SelfNumberDensity, EvaluateVelocity


def boundary_props():
    '''
    bid: Boundary particle closest to fluid
    cid: greater than zero if corner particle
    cbid: Boundary particle closed to fluid particle
    '''
    return ['disp', 'ioid', 'xn', 'yn', 'zn']


def get_io_names():
    return ['inlet', 'outlet']


def requires(bc):
    mirror_inlet = False
    mirror_outlet = False

    return mirror_inlet, mirror_outlet

def get_stepper(bc):
    from io_bc.common import InletStep, OutletStep
    return {'inlet':InletStep(), 'outlet':OutletStep()}

def io_bc(bcs, fluids, rho0, p0):
    print(bcs)
    import sys
    g0 = []
    g1 = []
    for bc in bcs:
        if bc == 'u_outlet':
            return []
        if bc == 'p_outlet':
            return []
        if bc == 'u_inlet':
            print("vel inlet doesn't exist")
            sys.exit(0)
        if bc == 'p_inlet':
            print("Pressure inlet bc doesn't exist")
            sys.exit(0)