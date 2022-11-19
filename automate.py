# Library imports.
import os
import numpy as np
import matplotlib.pyplot as plt

# Automan imports
from automan.api import Automator, Simulation 
from automan.api import CondaClusterManager
from automan.api import PySPHProblem
from automan.api import mdict, dprod, opts2path
from automan.utils import filter_by_name, filter_cases, compare_runs

BACKEND = " --openmp"

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
        base_cmd += " --max-steps 0 --no-plots"
        
        # Create parameteric cases
        def get_complete_parametric_opts():
            # Domain parameters
            perturb_opts = mdict(perturb=[1e-2, 1e-1, 1])
            dim_nx_opts = mdict(dim=[1], nx=[401, 801, 1601])
            dim_nx_opts += mdict(dim=[2], nx=[51, 101, 201])
            dim_nx_opts += mdict(dim=[3], nx=[31, 51, 101])

            # Add dim_nx_opts to dprod to account for zero perturbation cases
            all_options = dprod(perturb_opts, dim_nx_opts) + dim_nx_opts

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
            perturb_opts = mdict(perturb=[1e-1])
            dim_nx_opts = mdict(dim=[1], nx=[801, 1601])
            dim_nx_opts += mdict(dim=[2], nx=[51, 101])

            all_options = dprod(perturb_opts, dim_nx_opts) + dim_nx_opts

            KERNEL_CHOICES = [
                'WendlandQuinticC2', 'WendlandQuinticC4', 'Gaussian'
            ]
            INTERPOLATING_METHOD_CHOICES = ['sph', 'shepard', 'order1']
            i_kernel_opts = mdict(i_kernel=KERNEL_CHOICES)
            i_method_opts = mdict(i_method=INTERPOLATING_METHOD_CHOICES)
            interpolator_opts = dprod(i_kernel_opts, i_method_opts)

            all_options = dprod(all_options, interpolator_opts)
            return all_options

        all_options = get_complete_parametric_opts()
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


if __name__ == "__main__":
    PROBLEMS = [
        SineVelProfile,
    ]
    automator = Automator(
        simulation_dir='outputs',
        output_dir=os.path.join('manuscript', 'figures'),
        all_problems=PROBLEMS,
        cluster_manager_factory=CondaClusterManager
    )
    automator.run()

    
