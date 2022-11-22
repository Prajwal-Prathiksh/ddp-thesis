#NOQA
r"""
An automator script to reproduce the results of the thesis
###
"""
# Library imports.
import os
import itertools as IT
import numpy as np
import matplotlib.pyplot as plt

# Automan imports
from automan.api import Automator, Simulation 
from automan.api import CondaClusterManager
from automan.api import PySPHProblem
from automan.api import mdict, dprod, opts2path
from automan.utils import filter_cases, compare_runs

# Local imports.
from code.automate_utils import styles

#TODO: Add error checking for SimPlotter
#TODO: Add uniform_naming for Simplotter methods
#TODO: Run for multiple nx, perturb, i_nx
#TODO: Add a method to dump cases generated in a json? and keep appending it when new cases are generated
#TODO: Add a GUI for filtering and plotting comparisons plots
#TODO: Refacto code to make the above possible

BACKEND = " --openmp"
N_CORES, N_THREADS = 1, 2

class SineVelProfilePlotters(Simulation):
    """
    Helper methods for making comparisons plots for the sinusoidal velocity
    profile problem.
    """
    def Ek_loglog(self, **kw):
        """
        Plot the energy spectrum in loglog scale.
        """
        data = np.load(self.input_path('espec_result_0.npz'))
        plt.loglog(data['k'], data['Ek'], **kw)

    def Ek_plot(self, **kw):
        """
        Plot the energy spectrum.
        """
        data = np.load(self.input_path('espec_result_0.npz'))
        plt.plot(data['k'], data['Ek'], **kw)
    
    def Ek_loglog_exact(self, **kw):
        """
        Plot the exact energy spectrum in loglog scale.
        """
        data = np.load(self.input_path('espec_result_0.npz'))
        plt.loglog(data['k'], data['Ek_exact'], **kw)    
    
    def l2_error(self, **kw):
        """
        Plot the l2 error (wrt exact solution) in loglog scale.
        """
        data = np.load(self.input_path('espec_result_0.npz'))
        plt.loglog(data['k'], data['l2_error'], **kw)
    
    def Ek_loglog_no_interp(self, **kw):
        """
        Plot the energy spectrum calculated without interpolation in loglog scale.
        """
        data = np.load(self.input_path('espec_result_0.npz'))
        plt.loglog(data['k'], data['Ek_no_interp'], **kw)
    
    def l2_error_no_interp(self, **kw):
        """
        Plot the l2 error (wrt solution calculated without interpolation) in loglog scale.
        """
        data = np.load(self.input_path('espec_result_0.npz'))
        plt.loglog(data['k'], data['l2_error_no_interp'], **kw)
    
    
class SineVelProfile(PySPHProblem):
    """
    Automator to run the sinusoidal velocity profile problem.
    """
    def get_name(self):
        """
        Problem name.
        """
        return 'sine_vel_profiles'

    def plot_energy_spectrum(
        self, cases, labels, plt_type="loglog", styles=styles,
        title_suffix=""
    ):
        """
        Plot the energy spectrum.

        Parameters
        ----------
        cases : sequence
            Sequence of 'Simulation' objects.
        labels : sequence
            Sequence of labels for the cases.
        plt_type : str
            Type of plot. Can be: 'loglog', 'plot', 'l2_error',
            'l2_error_no_interp'
        styles: callable: returns an iterator/iterable of style keyword
            arguments.
            Defaults to the ``styles`` function defined in this module.
        title_suffix : str
            Suffix to be added to the title. Default is an empty string.
        """
        plt.figure()
        if plt_type == "loglog":
            compare_runs(
                sims=cases,
                method=SineVelProfilePlotters.Ek_loglog,
                exact=None,
                labels=labels,
                styles=styles,
            )
        elif plt_type == "plot":
            compare_runs(
                sims=cases,
                method=SineVelProfilePlotters.Ek_plot,
                exact=None,
                labels=labels,
                styles=styles,
            )
        elif plt_type == "l2_error":
            compare_runs(
                sims=cases,
                method=SineVelProfilePlotters.l2_error,
                exact=None,
                labels=labels,
                styles=styles,
            )
        elif plt_type == "l2_error_no_interp":
            compare_runs(
                sims=cases,
                method=SineVelProfilePlotters.l2_error_no_interp,
                exact=None,
                labels=labels,
                styles=styles,
            )
        else:
            raise ValueError("Invalid plt_type: {}".format(plt_type))

        title = "Energy spectrum"
        if plt_type == "loglog":
            xlabel = r"$log(k)$"
            ylabel = r"$log(E_k)$"
            title += " (loglog)"
        elif plt_type == "plot":
            xlabel = r"$k$"
            ylabel = r"$E_k$"
            title += " (loglog)"
        elif plt_type == "l2_error":
            xlabel = r"$log(k)$"
            ylabel = r"$log(L_2 error)$"
            title += r" ($L_2$ error)"
        elif plt_type == "l2_error_no_interp":
            xlabel = r"$log(k)$"
            ylabel = r"$log(L_2 error)$"
            title += r" ($L_2$ error, no interpolation)"
        
        title += " ({})".format(", ".join(labels)) + title_suffix
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()

        title = title.replace("$", "")
        plt.savefig(
            self.output_path(title + ".png"),
            dpi=350,
            bbox_inches="tight",
        )
        plt.close()     
        
    def setup(self):
        """
        Setup the problem.
        """
        base_cmd = "python code/sine_velocity_profile.py " + BACKEND
        base_cmd += " --max-steps 0"
        
        # Create parameteric cases
        def get_complete_parametric_opts():
            # Domain parameters
            perturb_opts = mdict(perturb=[0, 1e-2, 1e-1, 1])
            dim_nx_opts = mdict(dim=[1], nx=[401, 801, 1601])
            dim_nx_opts += mdict(dim=[2], nx=[51, 101, 201])
            dim_nx_opts += mdict(dim=[3], nx=[31, 51, 101])

            all_options = dprod(perturb_opts, dim_nx_opts)

            # Interpolator parameters
            from code.sine_velocity_profile import (
                KERNEL_CHOICES, INTERPOLATING_METHOD_CHOICES
            )
            i_kernel_opts = mdict(i_kernel=KERNEL_CHOICES)
            i_method_opts = mdict(i_method=INTERPOLATING_METHOD_CHOICES)
            interpolator_opts = dprod(i_kernel_opts, i_method_opts)

            all_options = dprod(all_options, interpolator_opts)
            return all_options

        def get_example_opts():
            perturb_opts = mdict(perturb=[0])
            dim_nx_opts = mdict(dim=[1], nx=[5001, 10001, 20001])
            dim_nx_opts += mdict(dim=[2], nx=[251, 501])
            # dim_nx_opts += mdict(dim=[3], nx=[101])

            all_options = dprod(perturb_opts, dim_nx_opts)

            # from code.sine_velocity_profile import KERNEL_CHOICES, INTERPOLATING_METHOD_CHOICES
            KERNEL_CHOICES = [
                'WendlandQuinticC4'
            ]
            INTERPOLATING_METHOD_CHOICES = ['order1']#, 'order1BL', 'order1MC']
            
            i_kernel_opts = mdict(i_kernel=KERNEL_CHOICES)
            i_method_opts = mdict(i_method=INTERPOLATING_METHOD_CHOICES)
            interpolator_opts = dprod(i_kernel_opts, i_method_opts)

            all_options = dprod(all_options, interpolator_opts)
            return all_options

        all_options = get_example_opts()

        # Setup cases
        self.cases = [
            Simulation(
                root=self.input_path(opts2path(kw)),
                base_command=base_cmd,
                job_info=dict(n_core=N_CORES, n_thread=N_THREADS),
                **kw
            )
            for kw in all_options
        ]
    
    def run(self):
        """
        Run the problem.
        """
        self.make_output_dir()
        tmp = dict(
            dim=[1,2,3],
            nx=[20001, 501, 101]
        )
        for dim, nx in zip(tmp['dim'], tmp['nx']):
            fcases = filter_cases(self.cases, dim=dim, nx=nx)
            title_suffix = " (dim={}, nx={})".format(dim, nx)
            labels = ['i_method', 'perturb']
            self.plot_energy_spectrum(
                fcases, labels, plt_type="l2_error", title_suffix=title_suffix
            )
            self.plot_energy_spectrum(
                fcases, labels, plt_type="l2_error_no_interp",
                title_suffix=f"{title_suffix} (no interpolation)"
            )


        for dim in tmp['dim']:
            fcases = filter_cases(self.cases, dim=dim, perturb=0)
            title_suffix = " (dim={})".format(dim)
            labels = ['i_method', 'nx']
            self.plot_energy_spectrum(
                fcases, labels, plt_type="l2_error", title_suffix=title_suffix
            )
            self.plot_energy_spectrum(
                fcases, labels, plt_type="l2_error_no_interp",
                title_suffix=f"{title_suffix} (no interpolation)"
            )



if __name__ == "__main__":
    PROBLEMS = [
        SineVelProfile,
    ]
    automator = Automator(
        simulation_dir='outputs',
        output_dir=os.path.join('manuscript', 'code_figures'),
        all_problems=PROBLEMS,
        cluster_manager_factory=CondaClusterManager
    )
    automator.run()

    
