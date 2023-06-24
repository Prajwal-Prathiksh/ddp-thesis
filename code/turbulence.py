r"""
Tools required for turbulent flow simulations and analysis.
Author: K T Prajwal Prathiksh
###
"""
# Library imports
import glob
import os
import logging
import inspect
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from pysph.solver.application import Application
from pysph.sph.equation import Group
from pysph.sph.basic_equations import SummationDensity
from pysph.tools.interpolator import (
    SPHFirstOrderApproximationPreStep, SPHFirstOrderApproximation,
    Interpolator
)
from pysph.sph.wc.kernel_correction import (
    GradientCorrectionPreStep, GradientCorrection,
    MixedKernelCorrectionPreStep, MixedGradientCorrection
)
from pysph.solver.utils import dump, load

# Local imports
from turbulence_utils import *
from energy_spectrum import EnergySpectrum
from automate_utils import plot_vline

logger = logging.getLogger(__name__)


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
        self.initial_ek_fname = None

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
            default='WendlandQuinticC6', choices=KERNEL_CHOICES,
            help="Interpolation kernel."
        )
        turb_options.add_argument(
            "--i-radius-scale", action="store", type=float,
            dest="i_radius_scale", default=2.0,
            help="Interpolation kernel radius scale."
        )
        turb_options.add_argument(
            "--i-method", action="store", type=str, dest="i_method",
            default='sph', choices=INTERPOLATING_METHOD_CHOICES,
            help="Interpolating method."
        )
        turb_options.add_argument(
            "--ek-norm-order", action="store", type=str, dest="ek_norm_order",
            default='inf', choices=['-inf', '1', '2', 'inf'],
            help="Order of the norm used to compute the energy spectrum."
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
        self.options.i_nx = self.options.i_nx

        order = self.options.ek_norm_order
        if order == '-inf':
            order = -np.inf
        elif order == 'inf':
            order = np.inf
        else:
            order = int(order)
        self.options.ek_norm_order = order

    def _get_interp_vel_fname(self, f_idx: int):
        """
        Return the filename for the interpolated velocity field for a given
        output file index.

        Parameters
        ----------
        f_idx : int
            Output file index.

        Returns
        -------
        fname : str
            Filename for the interpolated velocity field.
        """
        idx = self.output_files[f_idx].split("_")[-1].split(".")[0]
        fname = os.path.join(self.output_dir, f"interp_vel_{idx}.npz")
        return fname

    def _get_ek_fname(self, f_idx: int):
        """
        Return the filename for the energy spectrum for a given output file
        index.

        Parameters
        ----------
        f_idx : int
            Output file index.

        Returns
        -------
        fname : str
            Filename for the energy spectrum.
        """
        idx = self.output_files[f_idx].split("_")[-1].split(".")[0]
        fname = os.path.join(self.output_dir, f"ek_{idx}.npz")
        return fname

    def _get_interpolation_equations(self, method: str, dim: int):
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

    def _get_interpolated_vel_field_for_one_file(
        self, interp_obj: object, f_idx: int, dim: int, L: float, 
        return_interp_obj: bool = False
    ):
        """
        Return the interpolated velocity field for a given file index of the
        output files.

        Parameters
        ----------
        interp_obj : object
            Interpolator object. If None, create a new interpolator object.
        f_idx : int
            Index of the output file.
        dim : int
            Dimension of the problem.
        L : float
            Length of the domain.
        return_interp_obj : bool, optional
            If True, return the interpolator object.

        Returns
        -------
        dx : float
            Grid spacing.
        ui, vi, wi : np.ndarray
            Interpolated velocity field components.
        interp_obj : Interpolator, optional
            Interpolator object.
        """
        data = load(self.output_files[f_idx])

        i_nx = self.options.i_nx
        radius_scale = self.options.i_radius_scale

        # Create meshgrid based on dimension
        _x = np.linspace(0, L, i_nx)
        dx = _x[1] - _x[0]
        if dim == 1:
            x = _x
            y = z = None
        elif dim == 2:
            x, y = np.meshgrid(_x, _x)
            z = None
        elif dim == 3:
            x, y, z = np.meshgrid(_x, _x, _x)

        # Setup default interpolator properties
        kernel = get_kernel_cls(self.options.i_kernel, dim=dim)

        # Check if radius_scale is instance of float
        if isinstance(radius_scale, float):
            kernel.radius_scale = radius_scale

        # Get interpolation equations and method
        eqs, method = self._get_interpolation_equations(
            method=self.options.i_method, dim=dim
        )

        # Reuse the interpolator object if one is provided
        if interp_obj is None:
            interp_obj = Interpolator(
                particle_arrays=list(data['arrays'].values()),
                kernel=kernel,
                x=x, y=y, z=z,
                domain_manager=self.create_domain(),
                method=method,
                equations=eqs,
            )
        else:
            interp_obj.update_particle_arrays(list(data['arrays'].values()))

        if dim == 1:
            _u = interp_obj.interpolate('u')
            ui = _u
            vi = wi = None
        elif dim == 2:
            _u = interp_obj.interpolate('u')
            _v = interp_obj.interpolate('v')
            ui = _u.reshape(i_nx, i_nx)
            vi = _v.reshape(i_nx, i_nx)
            wi = None
        elif dim == 3:
            _u = interp_obj.interpolate('u')
            _v = interp_obj.interpolate('v')
            _w = interp_obj.interpolate('w')
            ui = _u.reshape(i_nx, i_nx, i_nx)
            vi = _v.reshape(i_nx, i_nx, i_nx)
            wi = _w.reshape(i_nx, i_nx, i_nx)

        if return_interp_obj:
            return dx, ui, vi, wi, interp_obj
        return dx, ui, vi, wi

    def _get_ek_for_one_file(
        self, f_idx: int, dim: int, L: float, U0: float, func_config: str
    ):
        """
        Return the energy spectrum of the flow for a given file index of the
        output files.

        Parameters
        ----------
        f_idx : int
            Index of the output file.
        dim : int
            Dimension of the flow.
        L : float
            Length of the domain.
        U0 : float
            Reference velocity of the flow.
        func_config : str, optional
            Configuration of the function.

        Returns
        -------
        espec_ob : EnergySpectrum
            EnergySpectrum object.
        """
        fname = self._get_interp_vel_fname(f_idx)
        if fname not in self.interp_vel_files:
            msg = f"Interpolated velocity field file: {fname} not found."
            msg += "\nRunning self.compute_interpolated_vel_field() first."
            warnings.warn(msg)
            self.compute_interpolated_vel_field(
                f_idx_list=[f_idx], dim=dim, L=L
            )

        espec_ob = EnergySpectrum.from_interp_vel_ofile(fname=fname, U0=U0)
        espec_ob.compute(
            order=self.options.ek_norm_order, func_config=func_config
        )
        if self.get_exact_ek() is not None:
            espec_ob.compute_l2_error(ek_exact=self.get_exact_ek())
        return espec_ob

    def _get_ek_from_initial_vel_field(self, dim: int, U0: float):
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
        L = data["L"]
        dx = data["dx"]
        if dim == 1:
            v = w = None
        elif dim == 2:
            w = None

        espec_initial_ob = EnergySpectrum(
            dim=dim, L=L, dx=dx, u=u, v=v, w=w, t=0., U0=U0
        )
        espec_initial_ob.compute(order=self.options.ek_norm_order)
        espec_initial_ob.compute_l2_error(ek_exact=self.get_exact_ek())

        return espec_initial_ob

    def _log_interpolator_details(
        self, dim: int, interp_obj: object
    ):
        """
        Log details of the interpolator used.

        Parameters
        ----------
        dim : int
            Dimension of the problem.
        interp_obj : object
            Interpolator object used.
        """
        mg = "Using interpolator:\n"
        mg += "-" * 70 + "\n"
        mg += f"Kernel:\n\t{interp_obj.kernel.__class__.__name__}(dim={dim})\n"
        mg += f"Kernel radius scale:\n\t{interp_obj.kernel.radius_scale}\n"
        mg += f"Method:\n\t{interp_obj.method}" + "\n"
        mg += "Equations:\n\t["
        for eqn in interp_obj.func_eval.equation_groups:
            mg += f"\n\t\t{eqn}"
        mg += "\n" + "-" * 70
        logger.info(mg)

    def _set_initial_vel_field_fname(self):
        """
        Set the name of the file containing the initial velocity field,
        to "initial_vel_field.npz".
        """
        fname = "initial_vel_field.npz"
        self.initial_vel_field_fname = os.path.join(self.output_dir, fname)

    def _get_ek_plot_types_mapper(self, plot_type: str):
        """
        Return the plot type corresponding to the given plot type.

        Parameters
        ----------
        plot_type : str
            Plot type.
            Valid options: loglog, semilogx, semilogy, plot, stem

        Returns
        -------
        plot_type : dict
            Plot type dictionary.
            Includes: func, xlabel, ylabel
        """
        PLOT_TYPES_MAPPER = dict(
            loglog=dict(
                func=plt.loglog,
                xlabel=r"$k$",
                ylabel=r"$E(k)$"
            ),
            semilogx=dict(
                func=plt.semilogx,
                xlabel=r"$k$",
                ylabel=r"$E(k)$"
            ),
            semilogy=dict(
                func=plt.semilogy,
                xlabel=r"$k$",
                ylabel=r"$E(k)$"
            ),
            plot=dict(
                func=plt.plot,
                xlabel=r"$k$",
                ylabel=r"$E(k)$"
            ),
            stem=dict(
                func=plt.stem,
                xlabel=r"$k$",
                ylabel=r"$E(k)$"
            ),
        )
        if plot_type not in PLOT_TYPES_MAPPER:
            raise ValueError(
                f"Invalid plot_type: {plot_type}. Valid options are: "
                f"{list(PLOT_TYPES_MAPPER.keys())}"
            )
        return PLOT_TYPES_MAPPER[plot_type]

    def _save_energy_spectrum_as_pysph_view_file(
        self, fname: str, dim: int, espec_ob: object,
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
        pa.add_property('ek_u', 'double', data=espec_ob.ek_u.flatten())
        pa.add_property(
            'ek_v', 'double', data=espec_ob.ek_v.flatten() if dim > 1 else 0.0
        )
        pa.add_property(
            'ek_w', 'double', data=espec_ob.ek_w.flatten() if dim > 2 else 0.0
        )
        pa.add_output_arrays(['ek_u', 'ek_v', 'ek_w'])

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

    # Properties
    @property
    def interp_vel_files(self):
        """
        Return the list of files containing the interpolated velocity
        field data.
        """
        all_files = glob.glob(os.path.join(self.output_dir, "interp_vel_*"))
        files = sorted([f for f in all_files if f.endswith(".npz")])
        return files

    @property
    def ek_files(self):
        """
        Return the list of files containing the energy spectrum data.
        """
        all_files = glob.glob(os.path.join(self.output_dir, "ek_*"))
        files = sorted([f for f in all_files if f.endswith(".npz")])
        return files

    # Public methods
    def set_problem_parameters(self):
        # TODO Write function
        pass

    def save_initial_vel_field(
        self, dim: int, u: np.ndarray, v: np.ndarray, w: np.ndarray,
        L: float, dx: float, **kwargs
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
        L : float
            Length of the domain.
        dx : float
            Grid spacing.
        **kwargs : dict, optional
            Additional keyword arguments to be passed to numpy.savez.
        """
        # Convert to numpy arrays if necessary
        if not isinstance(v, np.ndarray):
            v = np.full_like(u, v)
        if not isinstance(w, np.ndarray):
            w = np.full_like(u, w)

        assert len(u.shape) == dim, "u must have dimension dim"
        assert len(v.shape) == dim, "v must have dimension dim"
        assert len(w.shape) == dim, "w must have dimension dim"

        self._set_initial_vel_field_fname()
        np.savez(
            self.initial_vel_field_fname,
            dim=dim, u=u, v=v, w=w, L=L, dx=dx, **kwargs
        )

    def get_length_of_ek(self, dim: int):
        """
        Calculate the length of the computed energy spectrum beforehand,
        by using the number of interpolation points and the number of
        dimensions.

        Parameters
        ----------
        dim : int
            Dimension of the flow.

        Returns
        -------
        N : int
            Length of the computed energy spectrum.
        """
        N = int(1 + np.ceil(np.sqrt(dim * self.options.i_nx**2) / 2))
        return N

    def get_exact_ek(self):
        """
        Get the exact energy spectrum of the flow.
        Should return a 1D numpy array.
        The length of the array should be the same as the length of the
        computed energy spectrum.

        `self.get_length_of_energy_spectrum()` can be used to get the
        length of the exact energy spectrum.

        Note: If not implemented, return None, and warns the user, since
        some post-processing functionalities will not be available.

        """
        func_name = inspect.stack()[0][3]
        msg = f"self.{func_name}() is not implemented. Some post-processing " \
            f"functionalities will not be available."
        logger.warning(msg)
        return None

    def get_expected_ek_slope(self):
        """
        Get the slope of the energy spectrum of the flow.
        Should return a float.

        Note: If not implemented, return None, and warns the user, since
        some post-processing functionalities will not be available.
        """
        func_name = inspect.stack()[0][3]
        msg = f"self.{func_name}() is not implemented. Some post-processing " \
            f"functionalities will not be available."
        logger.warning(msg)
        return None

    # Post-processing methods
    def get_f_idx_list(self, pct_list: list):
        """
        Get the list of indices of the output files, corresponding to the
        specified percentage of the simulation time.

        Parameters
        ----------
        pct_list : list
            List of percentages of the simulation time.
            Eg: [0, 25, 50, 75, 100]

        Returns
        -------
        f_idx_list : list
            List of indices of the output files.
        """
        pct_list = np.array(sorted(pct_list))
        
        # Check if there are values in pct_list that are not in [0, 100]
        if np.any(pct_list < 0) or np.any(pct_list > 100):
            raise ValueError(
                "Values in pct_list must be in the range [0, 100]."
            )

        n_ofiles = len(self.output_files)
        f_idx_list = np.round(pct_list * n_ofiles / 100).astype(int)

        # Replace values that are greater than n_ofiles with n_ofiles-1
        f_idx_list[f_idx_list >= n_ofiles] = n_ofiles - 1
        
        # Remove duplicates
        f_idx_list = np.unique(f_idx_list)
        f_idx_list = np.sort(f_idx_list)

        f_idx_list = f_idx_list.tolist()
        return f_idx_list

    def compute_interpolated_vel_field(
        self, f_idx_list: list, dim: int, L: float,
    ):
        """
        Interpolate the velocity field at the specified indices of the
        output files and save the interpolated velocity field to a *.npz
        file in the output directory.

        Parameters
        ----------
        f_idx_list : list
            List of indices of the output files.
        dim : int
            Dimension of the flow.
        L : float
            Length of the domain.
        """
        t0 = time.time()
        log_interpolator_details = True

        interp_obj = None
        for f_idx in f_idx_list:
            # Check if the interpolated velocity field already exists
            fname = self._get_interp_vel_fname(f_idx)
            if fname in self.interp_vel_files:
                msg = f"Interpolated velocity field already exists at: {fname}."
                print(msg)
                continue

            dx, ui, vi, wi, interp_obj =\
                self._get_interpolated_vel_field_for_one_file(
                    interp_obj=interp_obj, f_idx=f_idx, dim=dim, L=L, 
                    return_interp_obj=True
                )

            if log_interpolator_details:
                self._log_interpolator_details(dim=dim, interp_obj=interp_obj)
                log_interpolator_details = False

            # Save interpolated velocity field
            data = load(self.output_files[f_idx])
            t = data["solver_data"]["t"]
            fname = self._get_interp_vel_fname(f_idx)

            save_vars = dict(
                t=t, dim=dim, L=L, dx=dx, i_nx=self.options.i_nx,
                ui=ui, vi=vi, wi=wi
            )

            np.savez(fname, **save_vars)
            msg = f"Interpolated velocity field saved to: {fname}."
            logger.info(msg)

        t1 = time.time()
        msg = "Time taken to interpolate and save "
        msg += f"velocity field: {t1 - t0:.2f} s."
        print(msg)
        logger.info(msg)

    def compute_ek(
        self, f_idx_list: list, dim: int, L: float, U0: float,
        func_config: str = 'compyle', save_pysph_view_file: bool = False,
        **kwargs
    ):
        """
        Compute the energy spectrum of the flow at the specified indices of
        the output files and save the energy spectrum to a *.npz file in the
        output directory. It also computes the fit of the energy spectrum using
        linear regression.

        Parameters
        ----------
        f_idx_list : list
            List of indices of the output files.
        dim : int
            Dimension of the flow.
        L : float
            Length of the domain.
        U0 : float
            Reference velocity.
        func_config : str, optional
            Configuration of the function. Default is 'compyle'.
        save_pysph_view_file : bool, optional
            If True, saves the energy spectrum as a PySPH viewable file.
            Default is False.
        **kwargs : dict, optional
            Additional keyword arguments to be passed to the
            `espec.get_ek_fit()` method.
        """
        t0 = time.time()
        for f_idx in f_idx_list:
            # Check if the energy spectrum file already exists
            fname = self._get_ek_fname(f_idx)
            if fname in self.ek_files:
                msg = f"Energy spectrum already exists at: {fname}."
                print(msg)
                continue

            espec_ob = self._get_ek_for_one_file(
                f_idx=f_idx, dim=dim, L=L, U0=U0, func_config=func_config
            )
            k_fit, ek_fit, fit_params = espec_ob.get_ek_fit(**kwargs)

            exact_slope, slope_error = self.get_expected_ek_slope(), None
            if exact_slope is not None:
                slope_error = np.abs(exact_slope - fit_params['slope'])
                fit_params.update(
                    dict(exact_slope=exact_slope, slope_error=slope_error)
                )

            # Save energy spectrum
            data = load(self.output_files[f_idx])
            t = data["solver_data"]["t"]
            fname = self._get_ek_fname(f_idx)

            save_vars = dict(
                t=t, dim=dim, L=L, U0=U0,
                k=espec_ob.k, ek=espec_ob.ek,
                ek_u=espec_ob.ek_u, ek_v=espec_ob.ek_v, ek_w=espec_ob.ek_w,
                ek_exact=self.get_exact_ek(), l2_error=espec_ob.l2_error,
                k_fit=k_fit, ek_fit=ek_fit, fit_params=fit_params,
            )

            np.savez(fname, **save_vars)
            msg = f"Energy spectrum saved to: {fname}."
            logger.info(msg)

            if save_pysph_view_file:
                self._save_energy_spectrum_as_pysph_view_file(
                    fname=self.output_files[f_idx], dim=dim, espec_ob=espec_ob
                )

        t1 = time.time()
        msg = "Time taken to compute and save "
        msg += f"energy spectrum: {t1 - t0:.2f} s."
        print(msg)
        logger.info(msg)

    def compute_ek_from_initial_vel_field(
        self, dim: int, L: float, U0: float
    ):
        """
        Computes the energy spectrum from the initial velocity field saved
        using `save_initial_vel_field()` and saves the energy spectrum to a
        *.npz file in the output directory.

        Parameters
        ----------
        dim : int
            Dimension of the problem.
        L : float
            Length of the domain.
        U0 : float
            Reference velocity.
        """
        espec_ob = self._get_ek_from_initial_vel_field(dim=dim, U0=U0)
        if espec_ob is None:
            return

        # Save energy spectrum
        fname = "initial_ek.npz"
        fname = os.path.join(self.output_dir, fname)
        self.initial_ek_fname = fname
        save_vars = dict(
            t=0.0, dim=dim, L=L, U0=U0,
            k=espec_ob.k, ek=espec_ob.ek,
            ek_u=espec_ob.ek_u, ek_v=espec_ob.ek_v, ek_w=espec_ob.ek_w,
            ek_exact=self.get_exact_ek(), l2_error=espec_ob.l2_error,
        )
        np.savez(fname, **save_vars)
        msg = f"Energy spectrum of initial velocity field saved to: {fname}."
        logger.info(msg)

    # Plotting methods
    def plot_ek(
        self, f_idx: int, plot_type: str = "loglog", plot_fit: bool = True,
        exact: bool = False, wo_interp_initial: bool = False,
        ylims: tuple = (1e-10, 1), fname_suffix: str = "",
        title_suffix: str = "",
    ):
        """
        Plot the computed energy spectrum stored in the *.npz file.

        Parameters
        ----------
        f_idx : int
            Index of the output file from which the energy spectrum was
            computed.
        plot_type : str, optional
            Plot type.
            Default is loglog.
            Valid options: loglog, semilogx, semilogy, plot, stem
        plot_fit : bool, optional
            If True, plots the fit of the energy spectrum.
            Default is True.
        fit : bool, optional
        exact : bool, optional
            If True, plots the exact expected energy spectrum.
            Default is False.
        wo_interp_initial : bool, optional
            If True, plots the energy spectrum computed without interpolating
            the initial velocity field.
            Default is False.
        ylims : tuple, optional
            y-axis limits. If None, the limits are automatically determined.
            Default is (1e-10, 1).
        fname_suffix : str, optional
            Suffix to be added to the file name.
            Default is "".
        title_suffix : str, optional
            Suffix to be added to the title.
            Default is "".
        """
        fname = self._get_ek_fname(f_idx=f_idx)
        if fname not in self.ek_files:
            msg = f"Energy spectrum file: {fname} not found."
            msg += "\nRunn self.compute_ek() first."
            raise ValueError(msg)

        data = np.load(fname, allow_pickle=True)
        t = float(data['t'])
        k, ek = data['k'], data['ek']

        k_fit, ek_fit, fit_params = None, None, None
        if plot_fit and np.size(data['k_fit']) > 1:
            k_fit, ek_fit = data['k_fit'], data['ek_fit']
            fit_params = data['fit_params']
            fit_params = dict(enumerate(fit_params.flatten()))[0]

        ek_exact = data['ek_exact']
        if not exact or np.size(ek_exact) == 1:
            ek_exact = None

        ek_initial_data = None
        if wo_interp_initial and self.initial_ek_fname is not None:
            ek_initial_data = np.load(self.initial_ek_fname, allow_pickle=True)

        plotter = self._get_ek_plot_types_mapper(plot_type=plot_type)
        plot_func = plotter['func']

        plt.figure()
        plot_func(k, ek, label="Computed")

        if k_fit is not None:
            slope, r_value = fit_params['slope'], fit_params['r_value']
            r_squared = r_value**2
            label = r"Fit: $(E(k) \propto k^{" + f"{slope:.2f}" + r"}) (R^2 "
            label += f"={r_squared:.2f}" + r")$"
            plot_func(k_fit, ek_fit, 'r-.', label=label)

        if ek_exact is not None:
            slope = fit_params['exact_slope']
            label = r"Exact: $(E(k) \propto k^{" + f"{slope:.2f}" + r"})$"
            plot_func(k, ek_exact, 'k-', label=label)

        if ek_initial_data is not None:
            plot_func(
                ek_initial_data['k'], ek_initial_data['ek'], 'g--',
                label=r"Without interpolation $(t=0)$"
            )

        plt.xlabel(plotter['xlabel'])
        plt.ylabel(plotter['ylabel'])
        plt.legend(fontsize=8)

        plt.title(f"Energy spectrum at t = {t:.2f} {title_suffix}")

        # Limit y-axis
        ymin, ymax = plt.ylim()
        ymin = max(ymin, ylims[0])
        ymax = min(ymax, ylims[1])
        plt.ylim(ymin, ymax)
        plt.minorticks_on()

        # Plot a vertical line at the middle of the k range.
        plot_vline(k, 2)
        plot_vline(k, 8)

        f_idx_str = fname.split("_")[-1].split(".")[0]
        fname = f"ek_{f_idx_str}_{plot_type}{fname_suffix}.png"
        fname = os.path.join(self.output_dir, fname)
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Energy spectrum plot saved to: {fname}.")

    def plot_ek_evolution(
        self, f_idx: list = None, plot_fit: bool = True,
        ylims: tuple = (1e-10, 1), fname_suffix: str = "",
        title_suffix: str = "",
    ):
        """
        Plot the evolution of energy spectrum for the given computed files
        indices.

        Parameters
        ----------
        f_idx : list
            List of output file indices to plot.
            Default is None which plots the first and last files' energy
            spectrum.
            If 'all', plots the evolution of energy spectrum for all the
            computed files.
        plot_fit : bool, optional
            If True, plots the fit of the energy spectrum for the last file.
            Default is True.
        ylims : tuple, optional
            y-axis limits. If None, the limits are automatically determined.
            Default is (1e-10, 1).
        fname_suffix : str, optional
            Suffix to be added to the file name.
            Default is "".
        title_suffix : str, optional
            Suffix to be added to the title.
            Default is "".
        """
        import matplotlib as mpl

        if len(self.ek_files) < 2:
            msg = "At least two energy spectrum files are required to plot"
            msg += " the evolution."
            raise ValueError(msg)

        if f_idx is None:
            files = [self.ek_files[0], self.ek_files[-1]]
        elif f_idx == 'all':
            files = self.ek_files
        else:
            files = [self._get_ek_fname(f_idx=i) for i in f_idx]

        n_files = len(files)
        # Get first and last time values.
        t0 = float(np.load(files[0])['t'])
        tf = float(np.load(files[-1])['t'])

        # Set the color map.
        c = np.arange(1, n_files + 1)
        c_ticks = np.linspace(t0, tf, n_files)
        norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap="viridis")
        cmap.set_array([])

        fig, ax = plt.subplots()
        for idx, fname in enumerate(files):
            data = np.load(fname, allow_pickle=True)
            k, ek = data['k'], data['ek']
            ax.loglog(k, ek, c=cmap.to_rgba(idx + 1))

            if plot_fit and idx == n_files - 1:
                if np.size(data['k_fit']) <= 1:
                    continue
                k_fit, ek_fit = data['k_fit'], data['ek_fit']
                fit_params = data['fit_params']
                fit_params = dict(enumerate(fit_params.flatten()))[0]

                slope, r_value = fit_params['slope'], fit_params['r_value']
                r_squared = r_value**2
                label = f"Fit (t={data['t']:.2f}): "
                label += r"$(E(k) \propto k^{" + f"{slope:.2f}" + r"})"\
                    "(R^2 "
                label += f"={r_squared:.2f}" + r")$"
                ax.loglog(k_fit, ek_fit, 'r-.', label=label)

        ax.legend(fontsize=8)

        # Limit y-axis
        ymin, ymax = ax.get_ylim()
        ymin = max(ymin, ylims[0])
        ymax = min(ymax, ylims[1])
        ax.set_ylim(ymin, ymax)

        ax.set_xlabel(r'$k$')
        ax.set_ylabel(r'$E(k)$')
        ax.grid(True, which='both', ls='--')
        cbar = fig.colorbar(cmap, ticks=c, ax=ax)
        cbar.ax.set_yticklabels([f"{t:.2f}" for t in c_ticks])
        cbar.set_label(r'$t$', rotation=90, labelpad=5)
        ax.set_title(f"Energy spectrum evolution {title_suffix}")

        # Save the figure
        fname = os.path.join(
            self.output_dir,
            f'energy_spectrum_evolution{fname_suffix}.png'
        )
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"Energy spectrum evolution plot saved to: {fname}.")
