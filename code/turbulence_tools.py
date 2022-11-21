r"""
Tools required for turbulent flow simulations and analysis.
"""
# Library imports
import os
import logging
import numpy as np
from pysph.base.utils import get_particle_array
from pysph.solver.application import Application
from pysph.sph.equation import Group
from pysph.sph.basic_equations import SummationDensity
from pysph.base.kernels import (
    CubicSpline, WendlandQuinticC2_1D, WendlandQuintic, WendlandQuinticC4_1D,
    WendlandQuinticC4, WendlandQuinticC6_1D, WendlandQuinticC6,
    Gaussian, SuperGaussian, QuinticSpline
)
from pysph.tools.interpolator import (
    SPHFirstOrderApproximationPreStep, SPHFirstOrderApproximation
)
from pysph.sph.wc.kernel_correction import (
    GradientCorrectionPreStep, GradientCorrection
)

#TODO: Add support for openmp in interpolator and m_mat in interpolator cls
#TODO: Add more kernel corrections
#TODO Add second order interpolator?

# Local imports
from energy_spectrum import EnergySpectrum

logger = logging.getLogger(__name__)

# Kernel choices
KERNEL_CHOICES = [
    'CubicSpline', 'WendlandQuinticC2', 'WendlandQuinticC4',
    'WendlandQuinticC6', 'Gaussian', 'SuperGaussian', 'QuinticSpline'
]

# Interpolating method choices
INTERPOLATING_METHOD_CHOICES = ['sph', 'shepard', 'order1', 'order1BL']


def get_kernel_cls(name: str, dim: int):
    """
        Return the kernel class corresponding to the name initialized with the
        dimension.

        Parameters
        ----------
        name : str
            Name of the kernel class.
        dim : int
            Dimension of the kernel.

        Returns
        -------
        kernel_cls : class
            Kernel class (dim).
    """
    if dim not in [1, 2, 3]:
        raise ValueError("Dimension must be 1, 2 or 3.")
    mapper = {
        'CubicSpline': CubicSpline,
        'WendlandQuinticC2': WendlandQuinticC2_1D if dim == 1
        else WendlandQuintic,
        'WendlandQuinticC4': WendlandQuinticC4_1D if dim == 1
        else WendlandQuinticC4,
        'WendlandQuinticC6': WendlandQuinticC6_1D if dim == 1
        else WendlandQuinticC6,
        'Gaussian': Gaussian,
        'SuperGaussian': SuperGaussian,
        'QuinticSpline': QuinticSpline
    }
    if name not in mapper:
        raise ValueError("Kernel name not recognized")
    return mapper[name](dim=dim)


class TurbulentFlowApp(Application):
    """
    Base class for all turbulent flow applications.
    """

    def __init__(self, *args, **kw):
        super(TurbulentFlowApp, self).__init__(*args, **kw)
        self._add_turbulence_options()

    # Private methods
    def _add_turbulence_options(self):
        """
        Add options required for turbulent flow simulations.
        """
        parser = self.arg_parse
        turb_options = parser.add_argument_group(
            "Turbulence Options",
            "Command line arguments for simulating turbulent flow and its "
            "postprocessing"
        )
        turb_options.add_argument(
            "--i-nx", action="store", type=int, dest="i_nx", default=None,
            help="Number of interpolation points along x direction. If not "
            "specified, it is set to nx."
        )
        turb_options.add_argument(
            "--i-kernel", action="store", type=str, dest="i_kernel",
            default='WendlandQuinticC2', choices=KERNEL_CHOICES,
            help="Interpolation kernel."
        )
        turb_options.add_argument(
            "--i-method", action="store", type=str, dest="i_method",
            default='sph', choices=INTERPOLATING_METHOD_CHOICES,
            help="Interpolating method."
        )

        # Change order of groups so that user options are printed at the end
        turb_idx, user_idx = 0, 0
        for i, group in enumerate(parser._action_groups):
            if group.title == "Turbulence Options":
                turb_idx = i
            elif group.title == "User":
                user_idx = i
        if turb_idx < user_idx:
            return

        parser._action_groups[turb_idx], parser._action_groups[user_idx] = \
            parser._action_groups[user_idx], parser._action_groups[turb_idx]

    def _parse_command_line(self, *args, **kw):
        """
        Parse command line arguments specific to turbulent flow simulations.
        """
        super(TurbulentFlowApp, self)._parse_command_line(*args, **kw)
        nx = self.options.nx
        i_nx = self.options.i_nx
        self.options.i_nx = i_nx if i_nx is not None else nx

    def _log_interpolator_details(self, fname, dim, interp):
        msg = "Using interpolator:\n"
        msg += "-" * 70 + "\n"
        msg += "Reading data from: %s" % fname + "\n"
        msg += f"Kernel: {interp.kernel.__class__.__name__}(dim={dim})" + "\n"
        msg += f"Method: {interp.method}" + "\n"
        msg += f"Equations: \n"
        for eqn in interp.func_eval.equation_groups:
            msg += f"\t{eqn}" + "\n" 
        msg += "-" * 70
        logger.info(msg)

    # Public methods
    # Post-processing methods
    def get_interpolation_equations(self, method: str, dim: int):
        """
        Return the equations for interpolating the velocity field.

        Parameters
        ----------
        method : str
            Interpolating method.
            Can be: 'sph', 'shepard', 'order1', 'order1BL'
        dim : int
            Dimension of the problem.

        Returns
        -------
        equations : sequence
            Equations for interpolating the velocity field.
        """
        if method in ['sph', 'shepard', 'order1']:
            equations = None
        elif method == 'order1BL':
            equations = [
                Group(
                    equations=[
                        SummationDensity(dest='fluid', sources=['fluid'])
                    ],
                    real=False
                ),
                Group(
                    equations=[
                        GradientCorrectionPreStep(
                            dest='interpolate', sources=['fluid'],
                            dim=dim
                        ),

                    ], real=False
                ),
                Group(
                    equations=[
                        GradientCorrection(
                            dest='interpolate', sources=['fluid'], dim=dim, tol=0.1
                        ),
                        SPHFirstOrderApproximationPreStep(
                            dest='interpolate', sources=['fluid'], dim=dim
                        ),
                        SPHFirstOrderApproximation(
                            dest='interpolate', sources=['fluid'], dim=dim
                        )
                    ], real=True
                )
            ]
        else:
            raise ValueError("Unknown method: %s" % method)
        return equations

    def get_exact_energy_spectrum(self):
        """
        Get the exact energy spectrum of the flow.
        If not implemented, return None, and warns the user.
        """
        logger.warn("get_exact_energy_spectrum() is not implemented.")
        return None

    def dump_enery_spectrum(self, dim: int, L: float, iter_idx: int = 0):
        """
        Dump the energy spectrum to a *.npz file.

        Parameters
        ----------
        dim : int
            Dimension of the problem.

        L : float
            Length of the domain.

        iter_idx : int
            Iteration index.
        """
        if len(self.output_files) == 0:
            return

        method = self.options.i_method
        if method not in ['sph', 'shepard', 'order1']:
            method = 'order1'

        i_kernel_cls = get_kernel_cls(self.options.i_kernel, dim)

        eqs = self.get_interpolation_equations(
            method=self.options.i_method, dim=dim
        )
        self.espec_ob, interp = EnergySpectrum.from_pysph_file(
            fname=self.output_files[iter_idx],
            dim=dim,
            L=L,
            i_nx=self.options.i_nx,
            kernel=i_kernel_cls,
            domain_manager=self.create_domain(),
            method=method,
            equations=eqs,
            U0=1.,
            debug=True
        )
        self.espec_ob.compute()
        self._log_interpolator_details(self.output_files[iter_idx], dim, interp)

        # Save npz file
        fname = os.path.join(self.output_dir, f"espec_result_{iter_idx}.npz")

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
        logger.info("Energy spectrum results saved to: %s" % fname)

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
        logger.info("Energy spectrum PySPH-viz file saved to: %s" % fname)

    def plot_energy_spectrum_evolution(self, f_idx: list = [0, -1]):
        """
        Plot the evolution of energy spectrum for the given files indices.

        Parameters
        ----------
        f_idx : list
            List of file indices to plot. Default is [0, -1] which plots the
            first and last files.
        """
        from energy_spectrum import EnergySpectrum
        fnames = [
            os.path.join(self.output_dir, f'espec_result_{i}.npz')
            for i in f_idx
        ]
        EnergySpectrum.plot_from_npz_file(
            fnames=fnames, figname=os.path.join(
                self.output_dir, 'energy_spectrum_evolution.png'))
