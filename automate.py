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
from automan.utils import filter_cases, filter_by_name

# Local imports.
from code.automate_utils import styles, custom_compare_runs

BACKEND = " --openmp"
N_CORES, N_THREADS = 1, 2

class SineVelProfilePlotters(Simulation):
    """
    Helper methods for making comparisons plots for the sinusoidal velocity
    profile problem.
    """
    def ek_loglog(self, **kw):
        """
        Plot the energy spectrum in loglog scale.
        """
        data = np.load(self.input_path('espec_result_00000.npz'))
        plt.loglog(data['k'], data['ek'], **kw)

    def ek_plot(self, **kw):
        """
        Plot the energy spectrum.
        """
        data = np.load(self.input_path('espec_result_00000.npz'))
        plt.plot(data['k'], data['ek'], **kw)
    
    def ek_loglog_no_interp(self, **kw):
        """
        Plot the energy spectrum calculated without interpolation in loglog scale.
        """
        data = np.load(self.input_path('espec_result_00000.npz'))
        kw.pop('label', None)
        plt.loglog(
            data['k'], data['ek_no_interp'],
            label=r'$E_k$ (no interpolation)', **kw
        )
        # Plot a vertical line at the middle of the k range.
        k_mid = data['k'][len(data['k'])//2]
        plt.axvline(k_mid, color='k', linestyle='--')
        plt.annotate(r'$k/2$', xy=(k_mid*1.1, 1e-2))
        k_quarter = data['k'][len(data['k'])//8]
        plt.axvline(k_quarter, color='k', linestyle='--')
        plt.annotate(r'$k/8$', xy=(k_quarter*1.1, 1e-2))

    
    def ek_loglog_exact(self, **kw):
        """
        Plot the exact energy spectrum in loglog scale.
        """
        data = np.load(self.input_path('espec_result_00000.npz'))
        plt.loglog(data['k'], data['ek_exact'], **kw)    
    
    def ek_plot_exact(self, **kw):
        """
        Plot the exact energy spectrum.
        """
        data = np.load(self.input_path('espec_result_00000.npz'))
        plt.plot(data['k'], data['ek_exact'], **kw)
    
    def l2_error(self, **kw):
        """
        Plot the L_2 error (wrt exact solution) in loglog scale.
        """
        data = np.load(self.input_path('espec_result_00000.npz'))
        plt.loglog(data['k'], data['l2_error'], **kw)

    def l2_error_no_interp(self, **kw):
        """
        Plot the L_2 error of the solution calculated without interpolation vs
        the exact solution in loglog scale.
        """
        data = np.load(self.input_path('espec_result_00000.npz'))
        l2_error_expected = data['ek_exact'] - data['ek_no_interp']
        l2_error_expected = np.sqrt(l2_error_expected**2)

        kw.pop('label', None)
        plt.loglog(
            data['k'], l2_error_expected,
            label=r'$L_2$ error (no interpolation)', **kw
        )
        k_mid = data['k'][len(data['k'])//2]
        plt.axvline(k_mid, color='k', linestyle='--')
        plt.annotate(r'$k/2$', xy=(k_mid*1.1, 1e-2))
    
    
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
        self, cases, labels, plt_type="loglog", title_suffix="", plot_legend:bool=True, plot_grid:bool=False, axis_scale:str=None,
        styles=styles
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
        title_suffix : str
            Suffix to be added to the title. Default is an empty string.
        plot_legend : bool
            Whether to plot the legend or not. Default is True.
        plot_grid : bool
            Whether to plot the grid or not. Default is False.
        axis_scale : str
            Passed to plt.axis(). Default is None.
        styles: callable: returns an iterator/iterable of style keyword
            arguments.
            Defaults to the ``styles`` function defined in this module.
        """
        title_beginning = "Energy spectrum"
        plotter_map = dict(
            loglog=dict(
                method=SineVelProfilePlotters.ek_loglog,
                exact=SineVelProfilePlotters.ek_loglog_no_interp,
                xlabel=r'$log(k)$',
                ylabel=r'$log(E_k)$',
                title_middle="(loglog)"
            ),
            plot=dict(
                method=SineVelProfilePlotters.ek_plot,
                exact=SineVelProfilePlotters.ek_plot_exact,
                xlabel=r'$k$',
                ylabel=r'$E_k$',
                title_middle="(plot)"
            ),
            l2_error=dict(
                method=SineVelProfilePlotters.l2_error,
                exact=SineVelProfilePlotters.l2_error_no_interp,
                xlabel=r'$log(k)$',
                ylabel=r'$log(L_2)$',
                title_middle=r"($L_2$ error wrt exact solution)"
            ),
            l2_error_no_interp=dict(
                method=SineVelProfilePlotters.l2_error_no_interp,
                exact=None,
                xlabel=r'$log(k)$',
                ylabel=r'$log(L_2)$',
                title_middle=r"($L_2$ error wrt no interpolation solution)"
            )
        )

        # Check if the plotter type is valid.
        if plt_type not in plotter_map.keys():
            raise ValueError(
                f"Invalid plotter type: {plt_type}. Valid types are: {plotter_map.keys()}"
            )

        # Get the plotter method and exact plotter method from the map.
        method = plotter_map[plt_type]['method']
        exact = plotter_map[plt_type]['exact']

        plt.figure()
        custom_compare_runs(
            sims=cases,
            method=method,
            exact=exact,
            labels=labels,
            styles=styles,
            exact_sim_idx=-1
        )

        def _get_title(beginning, middle, suffix):
            # Trim leading and trailing spaces for all the parts.
            beginning = beginning.strip()
            middle = middle.strip()
            suffix = suffix.strip()

            # Join the parts with a space in between.
            title = " ".join([beginning, middle, suffix])

            # Use title case for the title.
            title = title.title()

            return title
        
        plt.xlabel(plotter_map[plt_type]['xlabel'])
        plt.ylabel(plotter_map[plt_type]['ylabel'])
        title = _get_title(
            title_beginning, plotter_map[plt_type]['title_middle'],
            title_suffix
        )
        plt.title(title)

        # Limit y-axis to 1e-10 to y-axis max.
        if "log" in plotter_map[plt_type]['xlabel']:
            _, ymax = plt.ylim()
            plt.ylim(1e-20, ymax)
            plt.minorticks_on()

        if plot_legend:
            plt.legend()
        if plot_grid:
            plt.grid()
        if axis_scale:
            plt.axis(axis_scale)

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
            # perturb_opts = mdict(perturb=[0, 0.01])
            perturb_opts = mdict(perturb=[0.01], hdx=[1.2, 3])
            # dim_nx_opts = mdict(dim=[1], nx=[5001, 10001, 20001])
            # dim_nx_opts += mdict(dim=[2], nx=[251, 501])
            dim_nx_opts = mdict(dim=[1], nx=[20001])
            dim_nx_opts += mdict(dim=[2], nx=[501])
            dim_nx_opts += mdict(dim=[3], nx=[51])

            all_options = dprod(perturb_opts, dim_nx_opts)

            # KERNEL_CHOICES = [
            #     'CubicSpline', 'WendlandQuinticC2', 'WendlandQuinticC4',
            #     'WendlandQuinticC6', 'Gaussian', 'SuperGaussian', 'QuinticSpline'
            # ]
            KERNEL_CHOICES = [
                'WendlandQuinticC4'
            ]
            # INTERPOLATING_METHOD_CHOICES = ['sph', 'shepard', 'order1', 'order1BL', 'order1MC']
            INTERPOLATING_METHOD_CHOICES = ['sph', 'shepard', 'order1',]
            # INTERPOLATING_METHOD_CHOICES = ['order1', 'order1BL', 'order1MC']
            # INTERPOLATING_METHOD_CHOICES = ['shepard']
            
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
            nx=[20001, 501, 101, 51]
        )
        for dim, nx in zip(tmp['dim'], tmp['nx']):
            perturb=1e-2
            fcases = filter_cases(self.cases, dim=dim, nx=nx)#, perturb=perturb)
            if len(fcases) == 0:
                continue
            title_suffix = f"(dim={dim}, nx={nx})"#, perturb={perturb})"
            # labels = ['i_method', 'perturb']
            labels = ['i_method', 'hdx']
            # self.plot_energy_spectrum(
            #     fcases, labels, plt_type="l2_error", title_suffix=title_suffix,
            #     plot_grid=True
            # )
            self.plot_energy_spectrum(
                fcases, labels, plt_type="loglog", title_suffix=title_suffix,
                plot_grid=True
            )


class TempTGV(PySPHProblem):
    def get_name(self):
        """
        Problem name.
        """
        return "temp_tgv"
    
    def setup(self):
        """
        Setup the problem.
        """
        base_cmd = "python code/taylor_green.py" + BACKEND
        opts = mdict(scheme=[
            'edac', 'tsph --method sd --scm wcsph --pst-freq 10',
            'mon2017', 'ok2022'
        ])
        opts = dprod(
            opts, 
            mdict(
                re=[500, 1000, 5000, 10_000, 20_000, 50_000],
                tf=[1.], perturb=[0.2], nx=[50]
            )
        )
        
        def get_path(opt):
            temp = 'tsph' if 'tsph' in opt['scheme'] else opt['scheme']
            return f'scheme_{temp}_re_{opt["re"]}'
    
        # Setup cases
        self.cases = [
            Simulation(
                root=self.input_path(get_path(kw)),
                base_command=base_cmd,
                job_info=dict(n_core=N_CORES, n_thread=N_THREADS),
                **kw
            )
            for kw in opts
        ]
    
    def run(self):
        """
        Run the problem.
        """
        self.make_output_dir()


if __name__ == "__main__":
    PROBLEMS = [
        SineVelProfile,
        TempTGV,
    ]
    automator = Automator(
        simulation_dir='outputs',
        output_dir=os.path.join('manuscript', 'code_figures'),
        all_problems=PROBLEMS,
        cluster_manager_factory=CondaClusterManager
    )
    automator.run()

    
