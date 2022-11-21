r"""
Utilary functions and classes for the automate.py script.
###
"""
# Library imports.
import os
import itertools as IT
import numpy as np
import matplotlib.pyplot as plt
from functools import wraps

# Automan imports
from automan.api import Automator, Simulation
from automan.api import CondaClusterManager
from automan.api import PySPHProblem
from automan.api import mdict, dprod, opts2path
from automan.utils import filter_cases, compare_runs


def styles(sims):
    """
    Return iterable of styles for the given simulations.

    Parameters
    ----------
    sims : list of Simulation
        The simulations for which the styles are to be returned.

    Returns
    -------
    styles : iterable
        The styles for the given simulations.
    """
    ls = [dict(linestyle=x[0], color=x[1]) for x in
          IT.product(["-", "--", "-.", ":"], 'kbgrycm')]
    return IT.cycle(ls)
