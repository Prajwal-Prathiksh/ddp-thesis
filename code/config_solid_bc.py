from numpy.lib.arraysetops import isin
from pysph.sph.equation import Equation
from compyle.api import declare
from pysph.sph.wc.edac import SourceNumberDensity
from pysph.sph.equation import Group


def config_solid_bc(eqns, bctype, bcs, fluids, rho0, p0):
    print(bctype, bcs, fluids, rho0, p0)
    eqs = None
    if bctype == 'adami':
        from solid_bc.adami import solid_bc
        eqs = solid_bc(bcs, fluids, rho0, p0)
    elif bctype == 'takeda':
        from solid_bc.takeda import solid_bc
        eqs = solid_bc(bcs, fluids, rho0, p0)
    elif bctype == 'marongiu':
        from solid_bc.marongiu import solid_bc
        eqs = solid_bc(bcs, fluids, rho0, p0)
    elif bctype == 'hashemi':
        from solid_bc.hashemi import solid_bc
        eqs = solid_bc(bcs, fluids, rho0, p0)
    elif bctype == 'randles':
        from solid_bc.randles import solid_bc
        eqs = solid_bc(bcs, fluids, rho0, p0)
    elif bctype == 'colagrossi':
        from solid_bc.colagrossi import solid_bc
        eqs = solid_bc(bcs, fluids, rho0, p0)
    elif bctype == 'marrone':
        from solid_bc.marrone import solid_bc
        eqs = solid_bc(bcs, fluids, rho0, p0)
    elif bctype == 'esmaili':
        from solid_bc.esmaili import solid_bc
        eqs = solid_bc(bcs, fluids, rho0, p0)
    elif bctype == 'new':
        from solid_bc.new_bc import solid_bc
        eqs = solid_bc(bcs, fluids, rho0, p0)
    elif bctype == 'new_marrone':
        from solid_bc.new_marrone import solid_bc
        eqs = solid_bc(bcs, fluids, rho0, p0)
    elif bctype == 'new_fourtakas':
        from solid_bc.fourtakas_new import solid_bc, replace_momentum_eqns, replace_viscous_operator
        eqs = solid_bc(bcs, fluids, rho0, p0)
        if 'p_solid' in bcs:
            eqns = replace_momentum_eqns(eqns)
        elif 'u_no_slip' in bcs:
            eqns = replace_viscous_operator(eqns)

    elif bctype == 'mms':
        eqs = []
    else:
        import sys
        print('boundary not implemented', bctype)
        sys.exit(0)

    groups = get_iterative(bctype, bcs)
    iterative_added = False
    if eqs is not None:
        if len(eqs) > 0:
            # Add others equations as subsequent groups
            for i in range(len(eqs)):
                print(i)
                if (i in groups):
                    if iterative_added:
                        continue
                    iterative_added = True
                    iterate_eqs = []
                    for eq in groups:
                        iterate_eqs.append(Group(equations=eqs[eq]))
                    eqns.insert(
                        i + 1,
                        Group(equations=iterate_eqs,
                              iterate=True,
                              min_iterations=2,
                              max_iterations=5))
                else:
                    eqns.insert(i + 1, Group(equations=eqs[i]))
    else:
        import sys
        print('eqs cannot None solid bctype not found:', bctype)
        sys.exit(0)
    return eqns


def get_iterative(bctype, bcs):
    if bctype == "new":
        from solid_bc.new_bc import has_iterative
        return has_iterative(bcs)
    else:
        return []


def get_bc_require(bctype):
    if bctype == 'adami':
        from solid_bc.adami import requires
        return requires()
    elif bctype == 'takeda':
        from solid_bc.takeda import requires
        return requires()
    elif bctype == 'marongiu':
        from solid_bc.marongiu import requires
        return requires()
    elif bctype == 'hashemi':
        from solid_bc.hashemi import requires
        return requires()
    elif bctype == 'randles':
        from solid_bc.randles import requires
        return requires()
    elif bctype == 'colagrossi':
        from solid_bc.colagrossi import requires
        return requires()
    elif bctype == 'marrone':
        from solid_bc.marrone import requires
        return requires()
    elif bctype == 'esmaili':
        from solid_bc.esmaili import requires
        return requires()
    elif bctype == 'new':
        from solid_bc.new_bc import requires
        return requires()
    elif bctype == 'new_marrone':
        from solid_bc.new_marrone import requires
        return requires()
    elif bctype == 'new_fourtakas':
        from solid_bc.fourtakas_new import requires
        return requires()
    else:
        return False, False, True, False, False


def set_bc_props(bctype, particles):
    props = None
    if bctype == 'adami':
        from solid_bc.adami import boundary_props
        props = boundary_props()
    elif bctype == 'takeda':
        from solid_bc.takeda import boundary_props
        props = boundary_props()
    elif bctype == 'marongiu':
        from solid_bc.marongiu import boundary_props
        props = boundary_props()
    elif bctype == 'hashemi':
        from solid_bc.hashemi import boundary_props
        props = boundary_props()
    elif bctype == 'randles':
        from solid_bc.randles import boundary_props
        props = boundary_props()
    elif bctype == 'colagrossi':
        from solid_bc.colagrossi import boundary_props
        props = boundary_props()
    elif bctype == 'marrone':
        from solid_bc.marrone import boundary_props
        props = boundary_props()
    elif bctype == 'esmaili':
        from solid_bc.esmaili import boundary_props
        props = boundary_props()
    elif bctype == 'new':
        from solid_bc.new_bc import boundary_props
        props = boundary_props()
    elif bctype == 'new_marrone':
        from solid_bc.new_marrone import boundary_props
        props = boundary_props()
    elif bctype == 'new_fourtakas':
        from solid_bc.fourtakas_new import boundary_props
        props = boundary_props()

    if props is not None:
        for pa in particles:
            for prop in props:
                if isinstance(prop, dict):
                    pa.add_property(**prop)
                else:
                    pa.add_property(prop)


def set_solid_names(bctype, scheme):
    if bctype == 'adami':
        from solid_bc.adami import get_bc_names
        scheme.solids = get_bc_names()
    elif bctype == 'takeda':
        from solid_bc.takeda import get_bc_names
        scheme.solids = get_bc_names()
    elif bctype == 'marongiu':
        from solid_bc.marongiu import get_bc_names
        scheme.solids = get_bc_names()
    elif bctype == 'hashemi':
        from solid_bc.hashemi import get_bc_names
        scheme.solids = get_bc_names()
    elif bctype == 'randles':
        from solid_bc.randles import get_bc_names
        scheme.solids = get_bc_names()
    elif bctype == 'colagrossi':
        from solid_bc.colagrossi import get_bc_names
        scheme.solids = get_bc_names()
    elif bctype == 'marrone':
        from solid_bc.marrone import get_bc_names
        scheme.solids = get_bc_names()
    elif bctype == 'esmaili':
        from solid_bc.esmaili import get_bc_names
        scheme.solids = get_bc_names()
    elif bctype == 'new':
        from solid_bc.new_bc import get_bc_names
        scheme.solids = get_bc_names()
    elif bctype == 'new_marrone':
        from solid_bc.new_marrone import get_bc_names
        scheme.solids = get_bc_names()
    elif bctype == 'new_fourtakas':
        from solid_bc.fourtakas_new import get_bc_names
        scheme.solids = get_bc_names()
    else:
        scheme.solids = ['solid0', 'solid1']


def bc_pre_step(eval, solver, bctype, particles, domain):
    if bctype == 'colagrossi':
        from solid_bc.colagrossi import create_mirror_particles
        return create_mirror_particles(eval, solver, particles, domain)


def get_stepper(bctype):
    if bctype == 'marongiu':
        from solid_bc.marongiu import BoundaryRK2Stepper
        return {'boundary': BoundaryRK2Stepper()}
    else:
        return {}
