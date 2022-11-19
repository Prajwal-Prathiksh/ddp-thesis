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

BACKEND = " --openmp"

#TODO: Add error checking for SimPlotter
#TODO: Add uniform_naming for Simplotter methods
#TODO: Run for multiple nx, perturb, i_nx
#TODO: Add a GUI for filtering and plotting comparisons plots
#TODO: Refacto code to make the above possible

def styles(sims):
    ls = [dict(linestyle=x[0], color=x[1]) for x in
          IT.product(["-", "--", "-.", ":"], 'kbgrycm')]
    return IT.cycle(ls)

class SimPlotter(Simulation):
    def scalar_Ek_loglog(self, **kw):
        # Load npz file.
        data = np.load(self.input_path('espec_result_0.npz'))
        plt.loglog(data['k'], data['Ek'], **kw)
    
    def scalar_Ek_loglog_exact(self, **kw):
        # Load npz file.
        data = np.load(self.input_path('espec_result_0.npz'))
        plt.loglog(data['k'], data['Ek_exact'], **kw)
    
    def scalar_Ek_plot(self, **kw):
        # Load npz file.
        data = np.load(self.input_path('espec_result_0.npz'))
        # Get label from **kw.
        plt.plot(data['k'], data['Ek'], **kw)
    
    def l2_error(self, **kw):
        # Load npz file.
        data = np.load(self.input_path('espec_result_0.npz'))
        plt.loglog(data['k'], data['l2_error'], **kw)

class SineVelProfile(PySPHProblem):
    def get_name(self):
        """
            Problem name.
        """
        return 'sine_vel_profiles'

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
            dim_nx_opts = mdict(dim=[1], nx=[31, 1601])
            dim_nx_opts += mdict(dim=[2], nx=[21, 101, 201])
            dim_nx_opts += mdict(dim=[3], nx=[21, 51])

            all_options = dprod(perturb_opts, dim_nx_opts)

            # KERNEL_CHOICES = [
            #     'WendlandQuinticC4'
            # ]
            from code.sine_velocity_profile import KERNEL_CHOICES

            INTERPOLATING_METHOD_CHOICES = ['shepard']
            
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
                job_info=dict(n_core=1, n_thread=1),
                **kw
            )
            for kw in all_options
        ]
    
    def run(self):
        """
            Run the problem.
        """
        self.make_output_dir()

        def temp_plot(dim, nx, labels, plot_type, title):
            plt.figure()
            filtered_cases = filter_cases(
                self.cases, dim=dim, nx=nx, perturb=0,
            )
            if plot_type == 'loglog':
                compare_runs(
                    sims=filtered_cases, method=SimPlotter.scalar_Ek_loglog,
                    exact=SimPlotter.scalar_Ek_loglog_exact, labels=labels, styles=styles
                )
            elif plot_type == 'plot':
                compare_runs(
                    sims=filtered_cases, method=SimPlotter.scalar_Ek_plot,
                    exact=None, labels=labels, styles=styles
                )
            elif plot_type == 'l2_error':
                compare_runs(
                    sims=filtered_cases, method=SimPlotter.l2_error,
                    exact=None, labels=labels, styles=styles
                )
            plt.title(title)
            plt.xlabel('k')
            plt.ylabel('E(k)')
            plt.legend()
            plt.savefig(self.output_path(title + '.png'))
            plt.close()
        
        temp_plot(
            dim=1, nx=1601,
            labels=['i_kernel'],
            plot_type='l2_error',
            title='1D_i_kernel_l2_error'
        )
        temp_plot(
            dim=2, nx=201,
            labels=['i_kernel'],
            plot_type='l2_error',
            title='2D_i_kernel_l2_error'
        )
        temp_plot(
            dim=3, nx=51,
            labels=['i_kernel'],
            plot_type='l2_error',
            title='3D_i_kernel_l2_error'
        )

        temp_plot(
            dim=1, nx=1601,
            labels=['i_method'],
            plot_type='loglog',
            title='1D_i_method_loglog'
        )
        temp_plot(
            dim=1, nx=1601,
            labels=['i_method'],
            plot_type='l2_error',
            title='1D_i_method_l2_error'
        )
        temp_plot(
            dim=1, nx=31,
            labels=['i_method'],
            plot_type='plot',
            title='1D_i_method_plot'
        )
        temp_plot(
            dim=2, nx=101,
            labels=['i_method'],
            plot_type='loglog',
            title='2D_i_method_loglog'
        )
        temp_plot(
            dim=2, nx=101,
            labels=['i_method'],
            plot_type='l2_error',
            title='2D_i_method_l2_error'
        )
        temp_plot(
            dim=2, nx=21,
            labels=['i_method'],
            plot_type='plot',
            title='2D_i_method_plot'
        )
        temp_plot(
            dim=3, nx=51,
            labels=['i_method'],
            plot_type='loglog',
            title='3D_i_method_loglog'
        )
        temp_plot(
            dim=3, nx=51,
            labels=['i_method'],
            plot_type='l2_error',
            title='3D_i_method_l2_error'
        )
        temp_plot(
            dim=3, nx=21,
            labels=['i_method'],
            plot_type='plot',
            title='3D_i_method_plot'
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

    
