r"""
Tools required for turbulent flow simulations and analysis.
"""
# Library imports
import os
import numpy as np
from pysph.base.utils import get_particle_array
from pysph.solver.application import Application
from pysph.sph.equation import  Group
from pysph.sph.basic_equations import SummationDensity
from pysph.tools.interpolator import (
    SPHFirstOrderApproximationPreStep, SPHFirstOrderApproximation
)
from pysph.sph.wc.kernel_correction import (
    GradientCorrectionPreStep, GradientCorrection
)
import warnings

class TurbulentFlowApp(Application):
    """
    Base class for all turbulent flow applications.
    """
    def __init__(self, *args, **kw):
        super(TurbulentFlowApp, self).__init__(*args, **kw)

    # Post-processing tools
    def get_interpolation_equations(self, method):
        if method in ['sph', 'shepard', 'order1']:
            equations = None
        elif method == 'order1BL':
            equations = [
                Group(
                    equations=[SummationDensity(dest='fluid', sources=['fluid'])],
                    real=False
                ),
                Group(
                    equations=[
                        GradientCorrectionPreStep(
                            dest='fluid', sources=['fluid'], dim=self.dim
                        ),
                        GradientCorrection(
                            dest='fluid', sources=['fluid'], dim=self.dim,
                            tol=0.05
                        ),
                        SPHFirstOrderApproximationPreStep(
                            dest='interpolate', sources=['fluid'], dim=self.dim
                        ),
                        SPHFirstOrderApproximation(
                            dest='interpolate', sources=['fluid'], dim=self.dim)
                    ], real=True
                )
            ]
        else:
            raise ValueError("Unknown method: %s" % method)
        return equations
    
    def get_exact_energy_spectrum(self):
        warnings.warn("get_exact_energy_spectrum() is not implemented.")
        return None

    def dump_enery_spectrum(self, iter_idx=0):
        dim = self.dim
        if len(self.output_files) == 0:
            return

        from energy_spectrum import EnergySpectrum

        method = self.i_method
        if method not in ['sph', 'shepard', 'order1']:
            method = 'order1'

        self.espec_ob = EnergySpectrum.from_pysph_file(
            fname=self.output_files[iter_idx],
            dim=dim,
            L=self.L,
            i_nx=self.i_nx,
            kernel=self.i_kernel_cls,
            domain_manager=self.create_domain(),
            method=method,
            equations=self.get_interpolation_equations(method=self.i_method),
            U0=1.
        )
        self.espec_ob.compute()

        # Save npz file
        fname = os.path.join(self.output_dir, f"espec_result.npz")

        Ek_exact = self.get_exact_energy_spectrum()
        if Ek_exact is not None:
            l2_error = np.sqrt((self.espec_ob.Ek - Ek_exact)**2)
        else:
            l2_error = None

        np.savez(
            fname,
            k=self.espec_ob.k,
            t=self.espec_ob.t,
            Ek=self.espec_ob.Ek,
            EK_U=self.espec_ob.EK_U,
            EK_V=self.espec_ob.EK_V,
            EK_W=self.espec_ob.EK_W,
            Ek_exact=Ek_exact,
            l2_error=l2_error
        )
        print("Saved results to %s" % fname)

        # Save PySPH file
        from pysph.solver.utils import dump, load
        data = load(self.output_files[iter_idx])

        pa = data['arrays']['fluid']
        pa.add_property('EK_U', 'double', data=self.espec_ob.EK_U.flatten())
        pa.add_property(
            'EK_V',
            'double',
            data=self.espec_ob.EK_V.flatten() if dim > 1 else 0.)
        pa.add_property(
            'EK_W',
            'double',
            data=self.espec_ob.EK_W.flatten() if dim > 2 else 0.)

        pa.add_output_arrays(['EK_U', 'EK_V', 'EK_W'])

        counter = self.output_files[iter_idx].split("_")[-1].split('.')[0]
        fname = os.path.join(self.output_dir, f"espec_{counter}")
        if self.output_files[iter_idx].endswith(".npz"):
            fname += ".npz"
        else:
            fname += ".hdf5"
        dump(
            filename=fname,
            particles=[pa],
            solver_data=data['solver_data'],
            detailed_output=self.solver.detailed_output,
            only_real=self.solver.output_only_real,
            mpi_comm=None,
            compress=self.solver.compress_output
        )
        print("Saved %s" % fname)
