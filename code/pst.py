#NOQA
from math import sqrt
from compyle.api import declare
from pysph.sph.equation import Equation
from pysph.base.reduce_array import serial_reduce_array, parallel_reduce_array
from pysph.solver.tools import Tool


class EvaluateAverageDistance(Equation):
    def initialize(self, d_avgr, d_idx, d_nbr):
        d_avgr[d_idx] = 0.0
        d_nbr[d_idx] = 0.0

    def loop(self, d_idx, RIJ, d_avgr, d_nbr):
        d_avgr[d_idx] += RIJ
        d_nbr[d_idx] += 1

    def post_loop(self, d_idx, d_avgr, d_nbr):
        if d_nbr[d_idx] > 1e-14:
            d_avgr[d_idx] /= d_nbr[d_idx]


class SimpleShift(Equation):
    r"""**Simple shift**
    See the paper [XuStaLau2009], equation(35)
    """
    def __init__(self, dest, sources, const=0.01):
        self.beta = const
        super(SimpleShift, self).__init__(dest, sources)

    def initialize(self, d_idx, d_dpos):
        i = declare('int')
        for i in range(3):
            d_dpos[3 * d_idx + i] = 0.0

    def py_initialize(self, dst, t, dt):
        from numpy import sqrt
        vmag = sqrt(dst.u**2 + dst.v**2 + dst.w**2)
        dst.vmax[0] = serial_reduce_array(vmag, 'max')
        # dst.vmax[:] = parallel_reduce_array(dst.vmax, 'max')

    def loop(self, d_idx, XIJ, RIJ, d_vmax, d_dpos, dt, s_idx, d_avgr):

        eps = 1e-6
        r3ij = RIJ * RIJ * RIJ
        dxi = XIJ[0] / (r3ij + eps)
        dyi = XIJ[1] / (r3ij + eps)
        dzi = XIJ[2] / (r3ij + eps)

        ri = d_avgr[d_idx]
        fac = self.beta * ri*ri * d_vmax[0] * dt
        d_dpos[d_idx*3] += fac*dxi
        d_dpos[d_idx*3 + 1] += fac*dyi
        d_dpos[d_idx*3 + 2] = fac*dzi

    def post_loop(self, d_idx, d_dpos, d_x, d_y, d_z, d_h):
        val = sqrt(d_dpos[d_idx * 3]**2 + d_dpos[d_idx * 3 + 1]**2 +
                   d_dpos[d_idx * 3 + 2]**2)
        norm = val
        if val > 0.05 * d_h[d_idx]:
            norm = 0.05 * d_h[d_idx]

        fac = norm / val
        d_x[d_idx] += fac * d_dpos[d_idx * 3]
        d_y[d_idx] += fac * d_dpos[d_idx * 3 + 1]
        d_z[d_idx] += fac * d_dpos[d_idx * 3 + 2]


class ModifiedFickian(Equation):
    def __init__(self, dest, sources, fickian_const=1, tensile_const=0.24,
                 tensile_pow=4, hdx=1.0, tensile_correction=True):
        self.fickian_const = fickian_const
        self.tensile_const = tensile_const
        self.tensile_pow = tensile_pow
        self.hdx = hdx
        self.tensile_correction = tensile_correction
        super(ModifiedFickian, self).__init__(dest, sources)

    def initialize(self, d_idx, d_dpos):
        i = declare('int')
        for i in range(3):
            d_dpos[3 * d_idx + i] = 0.0

    def py_initialize(self, dst, t, dt):
        from numpy import sqrt
        vmag = sqrt(dst.u**2 + dst.v**2 + dst.w**2)
        dst.vmax[0] = serial_reduce_array(vmag, 'max')
        # dst.vmax[:] = parallel_reduce_array(dst.vmax, 'max')

    def loop(self, d_idx, d_h, s_idx, s_m, s_rho, dt, d_dpos, DWIJ, WIJ,
             SPH_KERNEL, d_vmax):
        dx = declare('matrix(3)')
        hi = d_h[d_idx]
        dx[0] = hi / self.hdx
        dx[1] = 0.0
        dx[2] = 0.0
        fij = 0.0
        wdx = SPH_KERNEL.kernel(dx, dx[0], d_h[d_idx])

        Vj = s_m[s_idx] / s_rho[s_idx]

        if self.tensile_correction:
            R = self.tensile_const
            n = self.tensile_pow
            fij = R * (WIJ / wdx)**n

        fac = -Vj * (1 + fij)/2 * self.fickian_const * hi * d_vmax[0] * dt
        d_dpos[3 * d_idx] += fac * DWIJ[0]
        d_dpos[3 * d_idx + 1] += fac * DWIJ[1]
        d_dpos[3 * d_idx + 2] += fac * DWIJ[2]

    def post_loop(self, d_idx, d_dpos, d_x, d_y, d_z):
        d_x[d_idx] += d_dpos[d_idx * 3]
        d_y[d_idx] += d_dpos[d_idx * 3 + 1]
        d_z[d_idx] += d_dpos[d_idx * 3 + 2]


class DeltaPlusSPHPST(Equation):
    def __init__(self, dest, sources, fickian_const=1, tensile_const=0.24,
                 tensile_pow=4, hdx=1.0, tensile_correction=True, umax=1.0):
        self.fickian_const = fickian_const
        self.tensile_const = tensile_const
        self.tensile_pow = tensile_pow
        self.hdx = hdx
        self.tensile_correction = tensile_correction
        self.umax = umax
        super(DeltaPlusSPHPST, self).__init__(dest, sources)

    def initialize(self, d_idx, d_dpos):
        i = declare('int')
        for i in range(3):
            d_dpos[3 * d_idx + i] = 0.0

    def loop(self, d_idx, d_h, s_idx, s_m, s_rho, dt, d_dpos, DWIJ, WIJ,
             SPH_KERNEL, d_vmax, d_c0):
        dx = declare('matrix(3)')
        hi = d_h[d_idx]
        dx[0] = hi / self.hdx
        dx[1] = 0.0
        dx[2] = 0.0
        fij = 0.0
        wdx = SPH_KERNEL.kernel(dx, dx[0], d_h[d_idx])

        Vj = s_m[s_idx] / s_rho[s_idx]

        if self.tensile_correction:
            R = self.tensile_const
            n = self.tensile_pow
            fij = R * (WIJ / wdx)**n

        Ma = self.umax/d_c0[0]
        fac = -Vj * (1 + fij)/2 * self.fickian_const * 4 * hi**2 * Ma
        d_dpos[3 * d_idx] += fac * DWIJ[0]
        d_dpos[3 * d_idx + 1] += fac * DWIJ[1]
        d_dpos[3 * d_idx + 2] += fac * DWIJ[2]

    def post_loop(self, d_idx, d_dpos, d_x, d_y, d_z):
        d_x[d_idx] += d_dpos[d_idx * 3]
        d_y[d_idx] += d_dpos[d_idx * 3 + 1]
        d_z[d_idx] += d_dpos[d_idx * 3 + 2]


class FickianShift(Equation):
    def __init__(self, dest, sources, fickian_const=1, tensile_const=0.2,
                 tensile_pow=4, hdx=1.0, tensile_correction=True):
        self.fickian_const = fickian_const
        self.tensile_const = tensile_const
        self.tensile_pow = tensile_pow
        self.hdx = hdx
        self.tensile_correction = tensile_correction
        super(FickianShift, self).__init__(dest, sources)

    def initialize(self, d_idx, d_dpos):
        i = declare('int')
        for i in range(3):
            d_dpos[3 * d_idx + i] = 0.0

    def loop(self, d_idx, d_h, s_idx, s_m, s_rho, dt, d_dpos, DWIJ, WIJ,
             SPH_KERNEL):
        dx = declare('matrix(3)')
        hi = d_h[d_idx]
        dx[0] = hi / self.hdx
        dx[1] = 0.0
        dx[2] = 0.0
        fij = 0.0
        wdx = SPH_KERNEL.kernel(dx, dx[0], d_h[d_idx])

        Vj = s_m[s_idx] / s_rho[s_idx]

        if self.tensile_correction:
            R = self.tensile_const
            n = self.tensile_pow
            fij = R * (WIJ / wdx)**n

        fij = 0.0
        fac = -Vj * (1 + fij) * self.fickian_const * hi**2 / 2
        d_dpos[3 * d_idx] += fac * DWIJ[0]
        d_dpos[3 * d_idx + 1] += fac * DWIJ[1]
        d_dpos[3 * d_idx + 2] += fac * DWIJ[2]

    def post_loop(self, d_idx, d_dpos, d_x, d_y, d_z, d_h):
        val = sqrt(d_dpos[d_idx * 3]**2 + d_dpos[d_idx * 3 + 1]**2 +
                   d_dpos[d_idx * 3 + 2]**2)
        norm = val
        if val > 0.2 * d_h[d_idx]:
            norm = 0.2 * d_h[d_idx]

        fac = 0.0
        if val > 1e-14:
            fac = norm / val
        d_x[d_idx] += fac * d_dpos[d_idx * 3]
        d_y[d_idx] += fac * d_dpos[d_idx * 3 + 1]
        d_z[d_idx] += fac * d_dpos[d_idx * 3 + 2]


class IterativePST(Equation):
    def initialize(self, d_idx, d_dpos):
        i = declare('int')
        for i in range(3):
            d_dpos[3 * d_idx + i] = 0.0

    def py_initialize(self, dst, t, dt):
        from numpy import sqrt
        vmag = sqrt(dst.u**2 + dst.v**2 + dst.w**2)
        dst.vmax[0] = serial_reduce_array(vmag, 'max')
        # dst.vmax[:] = parallel_reduce_array(dst.vmax, 'max')

    def loop(self, d_idx, d_h, s_idx, s_m, s_rho, dt, d_dpos, WIJ, DWIJ,
             SPH_KERNEL, d_vmax, XIJ, RIJ):
        Vj = s_m[s_idx] / s_rho[s_idx]

        fac = -Vj * d_h[d_idx] # * d_vmax[0] * dt
        d_dpos[3 * d_idx] += fac * XIJ[0] * WIJ / (RIJ + 1e-6)
        d_dpos[3 * d_idx + 1] += fac * XIJ[1] * WIJ / (RIJ + 1e-6)
        d_dpos[3 * d_idx + 2] += fac * XIJ[2] * WIJ / (RIJ + 1e-6)
        # d_dpos[3 * d_idx] += fac * DWIJ[0] #XIJ[0] * WIJ / (RIJ + 1e-6)
        # d_dpos[3 * d_idx + 1] += fac * DWIJ[1] #XIJ[1] * WIJ / (RIJ + 1e-6)
        # d_dpos[3 * d_idx + 2] += fac * DWIJ[2] #XIJ[2] * WIJ / (RIJ + 1e-6)

    def post_loop(self, d_idx, d_dpos, d_x, d_y, d_z):
        d_x[d_idx] += d_dpos[d_idx * 3]
        d_y[d_idx] += d_dpos[d_idx * 3 + 1]
        d_z[d_idx] += d_dpos[d_idx * 3 + 2]


class NumberDensityMoment(Equation):
    def __init__(self, dest, sources, debug=True, dim=2):
        self.debug = debug
        self.ki0 = 0.0
        self.ki = 0.0
        self.dim = dim

        super(NumberDensityMoment, self).__init__(dest, sources)

    def initialize(self, d_idx, d_ki0, d_ki, t, dt):
        if t < dt:
            d_ki0[d_idx] = 0.0
        d_ki[d_idx] = 0.0

    def loop(self, d_idx, d_ki0, d_ki, WIJ, d_h, t, dt):
        if t  < dt:
            d_ki0[d_idx] += d_h[d_idx]**self.dim * WIJ
        d_ki[d_idx] += d_h[d_idx]**self.dim * WIJ

    def reduce(self, dst, t, dt):
        if t < dt:
            dst.maxki0[0] = serial_reduce_array(dst.ki0, 'max')
            # dst.maxki0[:] = parallel_reduce_array(dst.maxki0, 'max')
            self.ki0 = dst.maxki0[0]
        dst.maxki[0] = serial_reduce_array(dst.ki, 'max')
        # dst.maxki[:] = parallel_reduce_array(dst.maxki, 'max')
        self.ki = dst.maxki[0]

    def converged(self):
        debug = self.debug
        diff = abs(self.ki - self.ki0)
        print(diff, self.ki, self.ki0)
        if diff - 0.001 < 1e-14:
            if debug:
                print("Converged:", diff)
            return 1.0
        else:
            if debug:
                print("Not converged:", diff)
            return -1.0


class CorrectVelocities(Equation):
    def initialize(self, d_idx, d_gradv):
        i = declare('int')
        for i in range(9):
            d_gradv[9*d_idx + i] = 0.0

    def loop(self, d_idx, s_idx, s_m, s_rho, d_gradv, DWIJ, VIJ):
        alp, bet = declare('int', 2)

        Vj = s_m[s_idx] / s_rho[s_idx]

        for alp in range(3):
            for bet in range(3):
                d_gradv[d_idx*9 + 3*bet + alp] += -Vj * VIJ[alp] * DWIJ[bet]

    def post_loop(self, d_idx, d_u, d_v, d_w, d_gradv, d_dpos, d_h):
        res = declare('matrix(3)')
        i, j = declare('int', 2)

        val = sqrt(d_dpos[d_idx * 3]**2 + d_dpos[d_idx * 3 + 1]**2 +
                   d_dpos[d_idx * 3 + 2]**2)
        norm = val
        if val > 0.05 * d_h[d_idx]:
            norm = 0.05 * d_h[d_idx]

        if val > 1e-14:
            fac = norm / val
        else:
            fac = norm

        for i in range(3):
            tmp = 0.0
            for j in range(3):
                tmp += d_gradv[d_idx*9 + 3*i + j] * fac * d_dpos[d_idx*3 + j]
            res[i] = tmp

        d_u[d_idx] += res[0]
        d_v[d_idx] += res[1]
        d_w[d_idx] += res[2]


class EvaluateVelocityGradient(Equation):
    def initialize(self, d_idx, d_gradv):
        i = declare('int')
        for i in range(9):
            d_gradv[9*d_idx + i] = 0.0

    def loop(self, d_idx, s_idx, s_m, s_rho, d_gradv, DWIJ, VIJ):
        alp, bet = declare('int', 2)

        Vj = s_m[s_idx] / s_rho[s_idx]

        for alp in range(3):
            for bet in range(3):
                d_gradv[d_idx*9 + 3*bet + alp] += -Vj * VIJ[alp] * DWIJ[bet]


class CorrectVelocitiesIPST(Equation):
    def initialize(self, d_idx, d_u, d_v, d_w, d_gradv, d_dpos, d_x, d_y, d_z,
                   d_x0, d_y0, d_z0):
        res, dx = declare('matrix(3)', 2)
        i, j = declare('int', 2)
        dx[0] = d_x[d_idx] - d_x0[d_idx]
        dx[1] = d_y[d_idx] - d_y0[d_idx]
        dx[2] = d_z[d_idx] - d_z0[d_idx]
        len = sqrt(dx[0]**2 + dx[1]**2 + dx[2]**2)
        fac = len
        # this is done to avoid periodic 
        # particle moving accross the domain
        if len > 0.5:
            fac = len - 1.0

        for i in range(3):
            tmp = 0.0
            for j in range(3):
                tmp += d_gradv[d_idx * 9 + 3 * i + j] * dx[j]
            res[i] = tmp

        if len > 1e-14:
            d_u[d_idx] += fac/len * res[0]
            d_v[d_idx] += fac/len * res[1]
            d_w[d_idx] += fac/len * res[2]


class ShiftPositions(Tool):
    def __init__(self, app, fluids, solids=[], freq=1, shift_kind='simple',
                 correct_velocity=False, parameter=None, hdx=1.0):
        """
        Parameters
        ----------

        app : pysph.solver.application.Application.
            The application instance.
        arr_name : array
            Name of the particle array whose position needs to be
            shifted.
        freq : int
            Frequency to apply particle position shift.
        shift_kind: str
            Kind to shift to apply available are "simple" and "fickian".
        correct_velocity: bool
            Correct velocities after shift in particle position.
        parameter: float
            Correct velocities after shift in particle position.
        """
        from pysph.solver.utils import get_array_by_name
        self.particles = app.particles
        self.dt = app.solver.dt
        self.dim = app.solver.dim
        self.kernel = app.solver.kernel
        self.fluids = fluids
        self.solids = solids
        all = self.fluids + self.solids
        self.array = [get_array_by_name(self.particles, name) for name in all]
        self.freq = freq
        self.kind = shift_kind
        self.correct_velocity = correct_velocity
        self.parameter = parameter
        self.hdx = hdx
        self.count = 1
        self._sph_eval = None
        options = ['simple', 'fickian', 'ipst', 'mod_fickian', 'delta_plus']
        assert self.kind in options, 'shift_kind should be one of %s' % options

    def _get_equations(self, kind='simple'):
        from pysph.sph.equation import Group
        eqns = []
        fluids = self.fluids
        all = self.fluids + self.solids
        if kind == 'simple':
            const = 0.01 if not self.parameter else self.parameter
            eqns.append(
                Group(equations=[
                    EvaluateAverageDistance(name, all) for name in fluids
                ]))
            eqns.append(
                Group(equations=[
                    SimpleShift(name, all, const=const) for name in fluids
                ],
                      update_nnps=True))
        elif kind == 'fickian':
            const = 4 if not self.parameter else self.parameter
            eqns.append(
                Group(equations=[
                    FickianShift(name, all, fickian_const=const, hdx=self.hdx, tensile_const=0.2)
                    for name in fluids
                ],
                      update_nnps=True))
        elif kind == 'mod_fickian':
            const = 4 if not self.parameter else self.parameter
            eqns.append(
                Group(equations=[
                    ModifiedFickian(name, all, fickian_const=const, hdx=self.hdx)
                    for name in fluids
                ],
                      update_nnps=True))
        elif kind == 'delta_plus':
            const = 1 if not self.parameter else self.parameter
            eqns.append(
                Group(equations=[
                    DeltaPlusSPHPST(name, all, fickian_const=const, hdx=self.hdx)
                    for name in fluids
                ],
                      update_nnps=True))
        elif kind == 'ipst':
            const = 1 if not self.parameter else self.parameter
            if self.correct_velocity:
                eqns.append(
                    Group(equations=[
                        EvaluateVelocityGradient(name, all)
                        for name in fluids
                    ]))
            eqns.append(
                Group(equations=[
                    IterativePST(name, all)
                    for name in fluids] +
                    [
                    NumberDensityMoment(name, all)
                    for name in fluids
                ],
                      update_nnps=False, iterate=True, min_iterations=1, max_iterations=10))

        if self.correct_velocity:
            if kind == 'ipst':
                eqns.append(
                    Group(equations=[
                        CorrectVelocitiesIPST(name, None)
                        for name in fluids
                    ], update_nnps=False))
            else:
                eqns[-1].equations.extend(
                    [CorrectVelocities(f, all) for f in fluids])

        print(eqns)
        return eqns

    def _get_sph_eval(self, kind):
        from pysph.tools.sph_evaluator import SPHEvaluator
        if self._sph_eval is None:

            for arr in self.particles:
                if kind == 'ipst':
                    if 'x0' not in arr.properties.keys():
                        arr.add_property('x0')
                        arr.add_property('y0')
                        arr.add_property('z0')
                    if 'maxki0' not in arr.constants.keys():
                        arr.add_constant('maxki0', [0.0])
                    if 'maxki' not in arr.constants.keys():
                        arr.add_constant('maxki', [0.0])
                    if 'ki0' not in arr.properties.keys():
                        arr.add_property('ki0')
                    if 'ki' not in arr.properties.keys():
                        arr.add_property('ki')
                    arr.x0[:] = arr.x[:]
                    arr.y0[:] = arr.y[:]
                    arr.z0[:] = arr.z[:]
                if 'vmax' not in arr.constants.keys():
                    arr.add_constant('vmax', [0.0])
                if 'dpos' not in arr.properties.keys():
                    arr.add_property('dpos', stride=3)
                if 'nbr' not in arr.properties.keys():
                    arr.add_property('nbr')
                if 'avgr' not in arr.properties.keys():
                    arr.add_property('avgr')
                if self.correct_velocity:
                    if 'gradv' not in arr.properties.keys():
                        arr.add_property('gradv', stride=9)

            eqns = self._get_equations(kind)
            sph_eval = SPHEvaluator(
                arrays=[arr], equations=eqns, dim=self.dim,
                kernel=self.kernel)
            return sph_eval
        else:
            return self._sph_eval

    def post_step(self, solver):
        if self.freq == 0:
            pass
        elif self.count % self.freq == 0:
            if self._sph_eval is None:
                self._sph_eval = self._get_sph_eval(self.kind)
            self._sph_eval.update()
            self._sph_eval.evaluate(dt=solver.dt, t=solver.t)
        self.count += 1
