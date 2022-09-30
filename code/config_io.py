from sys import implementation
from pysph.base.kernels import QuinticSpline
from pysph.sph.equation import Equation
from compyle.api import declare
from pysph.sph.equation import Group


def config_io_bc(eqns, bctype, bcs, fluids, rho0, p0):
    eqs = None
    print(bctype)
    if bctype == 'donothing':
        from io_bc.donothing import io_bc
        eqs = io_bc(bcs, fluids, rho0, p0)
    elif bctype == 'mirror':
        from io_bc.mirror import io_bc
        eqs = io_bc(bcs, fluids, rho0, p0)
    elif bctype == 'hybrid':
        from io_bc.hybrid import io_bc
        eqs = io_bc(bcs, fluids, rho0, p0)
    elif bctype == 'mirror_new':
        from io_bc.mirror_new import io_bc
        eqs = io_bc(bcs, fluids, rho0, p0)
    elif bctype == 'mms':
        eqs = []
    else:
        import sys
        print('boundary not implemented', bctype)
        sys.exit(0)

    print(eqs)
    if eqs is not None:
        if len(eqs) > 0:
            # Add others equations as subsequent groups
            for i in range(len(eqs)):
                eqns.insert(i+1, Group(equations=eqs[i]))
    else:
        import sys
        print('eqs cannot None solid bctype not found:', bctype)
        sys.exit(0)
    return eqns


def get_inlet_outlet_manager(app, xi=0.0, xo=1.0, is_inlet=None):
    bc = app.bc
    bctype = app.bctype
    iolen = app.nl * app.dx
    print(iolen, app.nl, app.dx)
    iom = None
    from pysph.sph.bc.inlet_outlet_manager import (
        InletInfo, OutletInfo)
    isinlet, isoutlet = None, None
    if not is_inlet is None:
        isinlet = is_inlet
        isoutlet = True
    else:
        isinlet = bc.split('_')[-1] == 'inlet'
        isoutlet = not(isinlet)
    print(is_inlet, isinlet, isoutlet)
    if bctype == 'donothing' or bctype == 'mms':

        from pysph.sph.bc.donothing.inlet import Inlet
        from pysph.sph.bc.donothing.outlet import Outlet
        from pysph.sph.bc.donothing.simple_inlet_outlet import SimpleInletOutlet

        props_to_copy = ['x', 'y', 'z', 'u', 'v', 'w', 'm',
                            'h', 'rho', 'p', 'ioid', 'gid', 'rhoc']
        inlet_info = InletInfo(
            pa_name='inlet', normal=[-1.0, 0.0, 0.0],
            refpoint=[xi, 0.0, 0.0], has_ghost=False if is_inlet is None else isinlet,
            update_cls=Inlet
        )

        outlet_info = OutletInfo(
            pa_name='outlet', normal=[1.0, 0.0, 0.0],
            refpoint=[xo, 0.0, 0.0], update_cls=Outlet,
            props_to_copy=props_to_copy
        )

        inlet_info.length = iolen
        outlet_info.length = iolen
        inlet_info.dx = app.dx
        outlet_info.dx = app.dx

        iom = SimpleInletOutlet(
            fluid_arrays=['fluid'], inletinfo=[inlet_info],
            outletinfo=[outlet_info]
        )
        if is_inlet:
            iom.inlet_pairs = {'inlet':'mirror_inlet'}

    elif bctype == 'mirror':

        from pysph.sph.bc.mirror.inlet import Inlet
        from pysph.sph.bc.mirror.outlet import Outlet
        from pysph.sph.bc.mirror.simple_inlet_outlet import SimpleInletOutlet

        props_to_copy = ['x', 'y', 'z', 'u', 'v', 'w', 'm',
                            'h', 'rho', 'p', 'ioid', 'gid']
        inlet_info = InletInfo(
            pa_name='inlet', normal=[-1.0, 0.0, 0.0],
            refpoint=[xi, 0.0, 0.0], has_ghost=isinlet,
            update_cls=Inlet
        )

        outlet_info = OutletInfo(
            pa_name='outlet', normal=[1.0, 0.0, 0.0],
            refpoint=[xo, 0.0, 0.0], has_ghost=isoutlet, update_cls=Outlet,
            props_to_copy=props_to_copy
        )

        inlet_info.length = iolen
        outlet_info.length = iolen
        inlet_info.dx = app.dx
        outlet_info.dx = app.dx

        iom = SimpleInletOutlet(
            fluid_arrays=['fluid'], inletinfo=[inlet_info],
            outletinfo=[outlet_info]
        )

        if isinlet:
            iom.inlet_pairs = {'inlet':'mirror_inlet'}
        if isoutlet:
            iom.outlet_pairs = {'outlet':'mirror_outlet'}
        print(iom.outlet_pairs)

    if bctype == 'hybrid':

        from pysph.sph.bc.hybrid.inlet import Inlet
        from pysph.sph.bc.hybrid.outlet import Outlet
        from pysph.sph.bc.hybrid.simple_inlet_outlet import SimpleInletOutlet

        props_to_copy = [
            'x', 'y', 'z', 'u', 'v', 'w', 'm', 'h', 'rho', 'p', 'ioid', 'uta',
            'pta', 'rta', 'gid', 'rhoc'
        ]
        inlet_info = InletInfo(
            pa_name='inlet', normal=[-1.0, 0.0, 0.0],
            refpoint=[xi, 0.0, 0.0], has_ghost=False if is_inlet is None else isinlet,
            update_cls=Inlet
        )

        outlet_info = OutletInfo(
            pa_name='outlet', normal=[1.0, 0.0, 0.0],
            refpoint=[xo, 0.0, 0.0], update_cls=Outlet,
            props_to_copy=props_to_copy
        )

        inlet_info.length = iolen
        outlet_info.length = iolen
        inlet_info.dx = app.dx
        outlet_info.dx = app.dx

        iom = SimpleInletOutlet(
            fluid_arrays=['fluid'], inletinfo=[inlet_info],
            outletinfo=[outlet_info]
        )
        if is_inlet:
            iom.inlet_pairs = {'inlet':'mirror_inlet'}

    if bctype == 'mirror_new':

        from pysph.sph.bc.mirror.inlet import Inlet
        from pysph.sph.bc.mirror.outlet import Outlet
        from pysph.sph.bc.mirror.simple_inlet_outlet import SimpleInletOutlet

        props_to_copy = [
            'x', 'y', 'z', 'u', 'v', 'w', 'm', 'h', 'rho', 'p', 'ioid', 'uta',
            'pta', 'rta', 'gid', 'rhoc'
        ]
        inlet_info = InletInfo(
            pa_name='inlet', normal=[-1.0, 0.0, 0.0],
            refpoint=[xi, 0.0, 0.0], has_ghost=isinlet,
            update_cls=Inlet
        )

        outlet_info = OutletInfo(
            pa_name='outlet', normal=[1.0, 0.0, 0.0],
            refpoint=[xo, 0.0, 0.0], update_cls=Outlet,
            props_to_copy=props_to_copy, has_ghost=isoutlet
        )

        inlet_info.length = iolen
        outlet_info.length = iolen
        inlet_info.dx = app.dx
        outlet_info.dx = app.dx

        iom = SimpleInletOutlet(
            fluid_arrays=['fluid'], inletinfo=[inlet_info],
            outletinfo=[outlet_info]
        )
        if isinlet:
            iom.inlet_pairs = {'inlet':'mirror_inlet'}
        if isoutlet:
            iom.outlet_pairs = {'outlet':'mirror_outlet'}


    if app.options.intg == 'euler':
        print("euler")
        iom.active_stages.append(1)
    else:
        iom.active_stages.append(2)
    iom.dim = app.dim
    iom.kernel = QuinticSpline(dim=app.dim)

    return iom


def get_bc_require(bctype, bc):
    if bctype == 'donothing':
        from io_bc.donothing import requires
        return requires(bc)
    if bctype == 'mirror':
        from io_bc.mirror import requires
        return requires(bc)
    if bctype == 'hybrid':
        from io_bc.hybrid import requires
        return requires(bc)
    if bctype == 'mirror_new':
        from io_bc.mirror_new import requires
        return requires(bc)
    else:
        return False, False


def set_bc_props(bctype, particles):
    props = None
    if bctype == 'donothing':
        from io_bc.donothing import boundary_props
        props = boundary_props()
    if bctype == 'mirror':
        from io_bc.mirror import boundary_props
        props = boundary_props()
    if bctype == 'hybrid':
        from io_bc.hybrid import boundary_props
        props = boundary_props()
    if bctype == 'mirror_new':
        from io_bc.mirror_new import boundary_props
        props = boundary_props()

    if props is not None:
        for pa in particles:
            for prop in props:
                if isinstance(prop, dict):
                    pa.add_property(**prop)
                else:
                    pa.add_property(prop)


def set_fluid_names(bctype, scheme):
    if bctype == 'donothing':
        from io_bc.donothing import get_io_names
        scheme.ios = get_io_names()
    if bctype == 'mirror':
        from io_bc.mirror import get_io_names
        scheme.ios = get_io_names()
    if bctype == 'hybrid':
        from io_bc.hybrid import get_io_names
        scheme.ios = get_io_names()
    if bctype == 'mirror_new':
        from io_bc.mirror_new import get_io_names
        scheme.ios = get_io_names()
    else:
        scheme.ios = ['inlet', 'outlet']


def bc_pre_step(eval, solver, bctype, particles, domain):
    pass


def get_stepper(bctype, bc):
    if bctype == 'donothing':
        from io_bc.donothing import get_stepper
        return get_stepper(bc)
    if bctype == 'mirror':
        from io_bc.mirror import get_stepper
        return get_stepper(bc)
    if bctype == 'hybrid':
        from io_bc.hybrid import get_stepper
        return get_stepper(bc)
    if bctype == 'mirror_new':
        from io_bc.mirror_new import get_stepper
        return get_stepper(bc)
    else:
        from io_bc.donothing import get_stepper
        return get_stepper(bc)
