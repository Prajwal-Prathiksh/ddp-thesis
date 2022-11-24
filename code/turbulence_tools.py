r"""
Tools required for turbulent flow simulations and analysis.
"""
# Library imports
import os
import logging
import numpy as np
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
    GradientCorrectionPreStep, GradientCorrection,
    MixedKernelCorrectionPreStep, MixedGradientCorrection
)
from pysph.solver.utils import dump, load

# Local imports
from energy_spectrum import EnergySpectrum

# TODO: Add support for openmp in interpolator and m_mat in interpolator cls
# TODO: Add more kernel corrections
# TODO Add second order interpolator?

logger = logging.getLogger(__name__)

# Kernel choices
KERNEL_CHOICES = [
    'CubicSpline', 'WendlandQuinticC2', 'WendlandQuinticC4',
    'WendlandQuinticC6', 'Gaussian', 'SuperGaussian', 'QuinticSpline'
]

# Interpolating method choices
INTERPOLATING_METHOD_CHOICES = [
    'sph', 'shepard', 'order1', 'order1BL', 'order1MC'
]


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
        """
        Initialize the application object, and add options required for
        turbulent flow simulations.
        """
        super().__init__(*args, **kw)
        self._add_turbulence_options()
        self.initial_vel_field_fname = None

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
            default='WendlandQuinticC4', choices=KERNEL_CHOICES,
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
        super()._parse_command_line(*args, **kw)
        nx = self.options.nx
        i_nx = self.options.i_nx
        self.options.i_nx = i_nx if i_nx is not None else nx

    def _log_interpolator_details(self, fname, dim, interp_ob):
        """
        Log details of the interpolator used.

        Parameters
        ----------
        fname : str
            Name of the file from which the PySPH particles are loaded.
        dim : int
            Dimension of the problem.
        interp_ob : object
            Interpolator object used.
        """
        msg = "Using interpolator:\n"
        msg += "-" * 70 + "\n"
        msg += f"Reading data from:\n\t{fname}\n"
        msg += f"Kernel:\n\t{interp_ob.kernel.__class__.__name__}(dim={dim})\n"
        msg += f"Method:\n\t{interp_ob.method}" + "\n"
        msg += "Equations:\n\t["
        for eqn in interp_ob.func_eval.equation_groups:
            msg += f"\n\t\t{eqn}"
        msg += "\n" + "-" * 70
        logger.info(msg)

    def _set_initial_vel_field_fname(self, fname: str):
        """
        Set the name of the file containing the initial velocity field.

        Parameters
        ----------
        fname : str
            Name of the file containing the initial velocity field.
        """
        self.initial_vel_field_fname = fname

    # Public methods
    def save_initial_vel_field(
        self, dim: int, u: np.ndarray, v: np.ndarray, w: np.ndarray,
        fname: str = None, **kwargs
    ):
        """
        Save the initial velocity field to a *.npz file in the output
        directory. The velocity field components should have appropriate
        dimensions, i.e. len(shape(u)) = len(shape(v)) = len(shape(w)) = dim.
        If a float is passed for (v or w) component  of velocity, it is
        converted to a numpy array of the shape of (u).

        Parameters
        ----------
        dim : int
            Dimension of the flow.
        u : np.ndarray
            Initial velocity field in x direction.
        v : np.ndarray
            Initial velocity field in y direction.
        w : np.ndarray
            Initial velocity field in z direction.
        fname : str, optional
            Name of the output file. If not specified, it is set to
            "initial_vel_field.npz".
        **kwargs : dict, optional
            Additional keyword arguments to be passed to numpy.savez.
        """
        if fname is None:
            fname = "initial_vel_field.npz"

        # Convert to numpy arrays if necessary
        if not isinstance(v, np.ndarray):
            v = np.full_like(u, v)
        if not isinstance(w, np.ndarray):
            w = np.full_like(u, w)

        assert len(u.shape) == dim, "u must have dimension dim"
        assert len(v.shape) == dim, "v must have dimension dim"
        assert len(w.shape) == dim, "w must have dimension dim"

        fname = os.path.join(self.output_dir, fname)
        np.savez(fname, dim=dim, u=u, v=v, w=w, **kwargs)
        self._set_initial_vel_field_fname(fname)

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
        consistent_method : str
            Consistent interpolating method to be used by the `Interpolator`
            class.
        """
        if method in ['sph', 'shepard', 'order1']:
            equations = None
            consistent_method = method
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
                            dest='interpolate', sources=['fluid'], dim=dim
                        ),

                    ], real=False
                ),
                Group(
                    equations=[
                        GradientCorrection(
                            dest='interpolate', sources=['fluid'], dim=dim,
                            tol=0.1
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
        elif method == 'order1MC':
            equations = [
                Group(
                    equations=[
                        SummationDensity(dest='fluid', sources=['fluid'])
                    ], real=False
                ),
                Group(
                    equations=[
                        MixedKernelCorrectionPreStep(
                            dest='interpolate', sources=['fluid'], dim=dim
                        )
                    ], real=False
                ),
                Group(
                    equations=[
                        MixedGradientCorrection(
                            dest='interpolate', sources=['fluid'], dim=dim,
                            tol=0.1
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
            raise ValueError(f"Unknown method: {method}")

        if equations:
            consistent_method = 'order1'
        return equations, consistent_method

    def get_exact_energy_spectrum(self):
        """
        Get the exact energy spectrum of the flow.
        If not implemented, return None, and warns the user.
        """
        logger.warning("get_exact_energy_spectrum() is not implemented.")
        return None

    def get_energy_spectrum_from_initial_vel_field(self, dim:int, U0: float):
        """
        Computes the energy spectrum from the initial velocity field saved
        using `save_initial_vel_field()`.

        Parameters
        ----------
        dim : int
            Dimension of the problem.
        U0 : float
            Initial velocity of the flow.
        
        Returns
        -------
        espec_initial_ob : `EnergySpectrum` object
            Energy spectrum object with computed energy spectrum.
        """
        fname = self.initial_vel_field_fname
        if fname is None:
            logger.warning(
                "Could not find initial velocity field file. "
                "Skipping computation of energy spectrum without "
                "interpolation! \nForgot to call "
                "`save_initial_vel_field()`?"
            )
            return None

        data = np.load(fname)
        u = data["u"]
        v = data["v"]
        w = data["w"]
        if dim == 1:
            v = w = None
        elif dim == 2:
            w = None

        espec_initial_ob = EnergySpectrum(
            dim=dim, u=u, v=v, w=w, t=0., U0=U0
        )
        espec_initial_ob.compute()

        return espec_initial_ob

    def get_energy_spectrum(self, fname: str, dim: int, L: float, U0:float=1):
        """
        Compute and get the energy spectrum from a given PySPH output file.

        Parameters
        ----------
        fname : str
            Name of the PySPH output file.
        dim : int
            Dimension of the problem.
        L : float
            Length of the domain.
        U0 : float, optional
            Reference velocity of the flow. Default is 1.
        
        Returns
        -------
        espec_ob : `EnergySpectrum` object
            Energy spectrum object with computed energy spectrum.
        """
        i_kernel_cls = get_kernel_cls(name=self.options.i_kernel, dim=dim)
        eqs, method = self.get_interpolation_equations(
            method=self.options.i_method, dim=dim
        )

        espec_ob, interp_ob = EnergySpectrum.from_pysph_file(
            fname=fname,
            dim=dim,
            L=L,
            i_nx=self.options.i_nx,
            kernel=i_kernel_cls,
            domain_manager=self.create_domain(),
            method=method,
            equations=eqs,
            U0=U0,
            debug=True
        )
        espec_ob.compute()

        self._log_interpolator_details(
            fname, dim, interp_ob
        )
        return espec_ob

    def save_energy_spectrum_as_pysph_view_file(
        self, fname:str, dim:int, espec_ob:object,
    ):
        """
        Save the energy spectrum as a PySPH viewable file.
        The file is saved as "espec_<counter>.npz/hdf5" in the output
        directory.
        The file can be viewed using the `pysph view` command.

        Parameters
        ----------
        fname : str
            Name of the file from which the energy spectrum was computed.
        dim : int
            Dimension of the problem.
        espec_ob : `EnergySpectrum` object
            The corresponding energy spectrum object of `fname` with the
            computed energy spectrum.
        """
        data = load(fname)

        # Add energy spectrum data to the particle array
        pa = data['arrays']['fluid']
        pa.add_property('EK_U', 'double', data=espec_ob.EK_U.flatten())
        pa.add_property(
            'EK_V', 'double', data=espec_ob.EK_V.flatten() if dim > 1 else 0.0
        )
        pa.add_property(
            'EK_W', 'double', data=espec_ob.EK_W.flatten() if dim > 2 else 0.0
        )
        pa.add_output_arrays(['EK_U', 'EK_V', 'EK_W'])

        # Save the data
        # Get the file index counter
        counter = fname.split("_")[-1].split('.')[0]
        fname = os.path.join(self.output_dir, f"espec_{counter}")
        if fname.endswith(".npz"):
            fname += ".npz"
        else:
            fname += ".hdf5"

        # Dump the file
        dump(
            filename=fname,
            particles=[pa],
            solver_data=data['solver_data'],
            detailed_output=self.solver.detailed_output,
            only_real=self.solver.output_only_real,
            mpi_comm=None,
            compress=self.solver.compress_output
        )
        msg = f"Energy spectrum PySPH viewable file saved to: {fname}"
        msg += f'. Can be viewed by running: \n\t$ pysph view "{fname}"'
        logger.info(msg)

    def energy_spectrum_post_processing(
        self, dim: int, L: float, U0:float=1.0, f_idx: int = 0,
        compute_without_interp: bool = False
    ):
        """
        Post-processing of the energy spectrum.

        Parameters
        ----------
        dim : int
            Dimension of the problem.
        L : float
            Length of the domain.
        U0 : float, optional
            Reference velocity of the flow. Default is 1.
        f_idx : int, optional
            Index of the output file to be used for computing the energy
            spectrum. Default is 0.
        compute_without_interp : bool
            If True, computes the energy spectrum with and without
            interpolating the velocity field. This requires the initial
            velocity field to be saved using `save_initial_vel_field()`.
        """
        if len(self.output_files) == 0:
            return
        
        # Get the energy spectrum
        espec_ob = self.get_energy_spectrum(
            fname=self.output_files[f_idx], dim=dim, L=L
        )

        Ek_no_interp, l2_error_no_interp = None, None
        if compute_without_interp:
            espec_initial_ob = self.get_energy_spectrum_from_initial_vel_field(
                dim=dim, U0=U0
            )
            if espec_initial_ob is not None:
                Ek_no_interp = espec_initial_ob.Ek
                l2_error_no_interp = np.sqrt((espec_ob.Ek - Ek_no_interp)**2)

        # Save npz file
        fname = os.path.join(self.output_dir, f"espec_result_{f_idx}.npz")

        Ek_exact = self.get_exact_energy_spectrum()
        if Ek_exact is not None:
            l2_error = np.sqrt((espec_ob.Ek - Ek_exact)**2)
        else:
            l2_error = None

        np.savez(
            fname,
            k=espec_ob.k,
            t=espec_ob.t,
            Ek=espec_ob.Ek,
            EK_U=espec_ob.EK_U,
            EK_V=espec_ob.EK_V,
            EK_W=espec_ob.EK_W,
            Ek_exact=Ek_exact,
            l2_error=l2_error,
            Ek_no_interp=Ek_no_interp,
            l2_error_no_interp=l2_error_no_interp
        )
        logger.info("Energy spectrum results saved to: %s", fname)

        # Save PySPH viewable file
        self.save_energy_spectrum_as_pysph_view_file(
            fname=self.output_files[f_idx], dim=dim, espec_ob=espec_ob
        )

    def plot_energy_spectrum_evolution(self, f_idx: list = None):
        """
        Plot the evolution of energy spectrum for the given files indices.

        Parameters
        ----------
        f_idx : list
            List of file indices to plot. Default is [0, -1] which plots the
            first and last files.
        """
        if f_idx is None:
            f_idx = [0, -1]
        fnames = [
            os.path.join(self.output_dir, f'espec_result_{i}.npz')
            for i in f_idx
        ]
        EnergySpectrum.plot_from_npz_file(
            fnames=fnames, figname=os.path.join(
                self.output_dir, 'energy_spectrum_evolution.png'))
