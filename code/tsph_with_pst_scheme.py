import imp
from pysph.sph.scheme import Scheme
from pysph.sph.equation import Equation
from pysph.sph.integrator import Integrator, IntegratorStep

from pst import IterativePST, ModifiedFickian, DeltaPlusSPHPST
from pysph.sph.wc.linalg import gj_solve
from compyle.api import declare
from pysph.sph.integrator import EulerIntegrator
from tsph_with_pst import (
    TSPHScheme, RK2Integrator, RK2Stepper, VelocityGradient, VelocityGradientSoild,
    VelocityGradientDestSoild, VelocityGradientSolidSoild, LinearEOS,
    MomentumEquationSecondOrder, DivGrad, ContinuityEquation, ContinuityEquationSolid,
    SaveInitialdistances, DensityGradient, IterativePSTNew, UpdateDensity,
    UpdateVelocity
    )


class TSPHSchemeMain(TSPHScheme):
    def get_equations(self):
        from pysph.sph.equation import Group
        from pysph.sph.basic_equations import \
            (SummationDensity)
        # from pysph.sph.wc.basic import TaitEOS
        from pysph.sph.wc.transport_velocity import MomentumEquationArtificialViscosity

        equations = []
        g1 = []
        all = self.fluids + self.solids + self.ios

        g0 = []
        self.pressure_density_eq(g0)
        equations.append(Group(equations=g0, update_nnps=True))

        g0 = []
        for name in all:
            g0.append(SummationDensity(dest=name, sources=all))
        equations.append(Group(equations=g0, update_nnps=True))

        g0 = []
        from pysph.sph.wc.kernel_correction import GradientCorrectionPreStep, GradientCorrection
        for name in self.fluids+self.ios:
            g0.append(
                GradientCorrectionPreStep(dest=name,
                                            sources=all,
                                            dim=self.dim))
        for name in self.solids:
            g0.append(
            GradientCorrectionPreStep(dest=name,
                                    sources=all,
                                    dim=self.dim))
        equations.append(Group(equations=g0, update_nnps=True))

        g2 = []
        all = self.fluids + self.solids + self.ios
        from pysph.sph.wc.kernel_correction import GradientCorrectionPreStep, GradientCorrection
        for name in self.fluids+self.ios:
            g2.append(GradientCorrection(dest=name, sources=all))
        for name in self.fluids+self.ios:
            g2.append(VelocityGradient(dest=name, sources=self.fluids+self.ios))

        equations.append(Group(equations=g2, update_nnps=True))

        g3 = []
        if self.kernel_corr:
            for name in self.fluids:
                g3.append(GradientCorrection(dest=name, sources=all))
        self.get_continuity_eq(g3)
        self.get_viscous_eq(g3)
        self.get_pressure_grad(g3)

        equations.append(Group(equations=g3))

        return equations

    def get_continuity_eq(self, g3):
        all = self.fluids + self.ios
        for name in self.fluids:
            g3.append(ContinuityEquation(dest=name, sources=all))

    def post_step_ipst(self, pa_arr, domain, get_mms_eq, mms, bc, solids=None):
        if self.shifter is None:
            from pysph.tools.sph_evaluator import SPHEvaluator
            from pysph.sph.wc.kernel_correction import GradientCorrection, GradientCorrectionPreStep
            from pst import ModifiedFickian, IterativePST, NumberDensityMoment
            from pysph.sph.equation import Group
            from pysph.sph.basic_equations import \
                (SummationDensity)
            from config_mms import SetValuesonSolid

            equations = []
            if solids is None:
                _solids = self.solids
            else:
                _solids = solids
            all = self.fluids + _solids + self.ios

            g0 = []
            for name in all:
                g0.append(SummationDensity(dest=name, sources=all))
            equations.append(Group(equations=g0, update_nnps=True))


            g1 = []
            for name in self.fluids:
                g1.append(GradientCorrectionPreStep(dest=name, sources=all, dim=self.dim))
                g1.append(SaveInitialdistances(dest=name, sources=None))
            equations.append(Group(equations=g1, update_nnps=True))

            g2 = []
            for name in self.fluids:
                g2.append(GradientCorrection(dest=name, sources=all, dim=self.dim))
                g2.append(VelocityGradient(dest=name, sources=all, dim=self.dim))
                g2.append(DensityGradient(dest=name, sources=all, dim=self.dim))
            equations.append(Group(equations=g2))

            g3 = []
            for name in self.fluids:
                g3.append(IterativePSTNew(dest=name, sources=all))
                g3.append(NumberDensityMoment(dest=name, sources=all, debug=False))
            equations.append(Group(equations=g3, iterate=True, min_iterations=1, max_iterations=10, real=False))

            g4 = []
            for name in self.fluids:
                g4.append(UpdateVelocity(dest=name, sources=None))
                g4.append(UpdateDensity(dest=name, sources=None))
            equations.append(Group(equations=g4))


            print(equations)

            self.shifter = SPHEvaluator(
                arrays=pa_arr, equations=equations, dim=self.dim,
                kernel=self.solver.kernel, backend='cython'
            )

        if (self.pst_freq > 0 and self.solver.count % self.pst_freq == 0) & (self.solver.count > 1):
            self.shifter.update()
            self.shifter.evaluate(t=self.solver.t, dt=self.solver.dt)
