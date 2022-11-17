# Library imports.
import os
from itertools import product
import numpy as np
import matplotlib.pyplot as plt

# Automan imports
from automan.api import Automator, Simulation 
from automan.api import PySPHProblem as Problem
from automan.automation import filter_cases, filter_by_name

class EnergySpectrum2D(Problem):
    def get_name(self):
        """
            Problem name.
        """
        return 'energy_spectrum_2d'

    def setup(self):
        """
            Setup the problem.
        """

        base_cmd = "python code/energy_spectrum.py"

    
