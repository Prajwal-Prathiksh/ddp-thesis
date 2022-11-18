# Library imports.
import os
import numpy as np
import matplotlib.pyplot as plt

# Automan imports
from automan.api import Automator, Simulation 
from automan.api import CondaClusterManager
from automan.api import PySPHProblem as Problem
from automan.api import mdict, dprod, opts2path

BACKEND = " --openmp"

class GenerateSineVelData(Problem):
    def get_name(self):
        """
            Problem name.
        """
        return 'sine_vel_profiles'

    def setup(self):
        """
            Setup the problem.
        """
        base_cmd = "python code/sine_velocity_profile.py " + BACKEND + " "
        
        # Create parameteric cases
        perturb_opts = mdict(perturb=[1])
        dim_nx_opts = mdict(dim=[1], nx=[51])
        dim_nx_opts += mdict(dim=[2], nx=[51])

        # Add dim_nx_opts to dprod to account for the zero perturbation cases
        all_options = dprod(perturb_opts, dim_nx_opts) #+ dim_nx_opts
        print(all_options)

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
        GenerateSineVelData
    ]
    automator = Automator(
        simulation_dir='outputs',
        output_dir=os.path.join('manuscript', 'figures'),
        all_problems=PROBLEMS,
        cluster_manager_factory=CondaClusterManager
    )
    automator.run()

    
