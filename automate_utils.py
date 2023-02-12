r"""
Utilary functions and classes for the automate.py script.
Author: K T Prajwal Prathiksh
###
"""
# Library imports.
import os
import collections
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

def custom_compare_runs(
    sims:list, method:callable, labels:list, exact:callable=None,
    styles=styles, exact_sim_idx:int=0
):
    """
    Given a sequence of Simulation instances, a method name, the labels to
    compare and an optional method name for an exact solution, this calls the
    methods with the appropriate parameters for each simulation.

    Parameters
    ----------
    sims: sequence
        Sequence of `Simulation` objects.
    method: callable
        The plotter method to call on each simulation.
    labels: sequence
        Sequence of parameters to use as labels for the plot.
    exact: callable
        The exact solution method to call on the simulation.
    styles: callable: returns an iterator/iterable of style keyword arguments.
        Defaults to the ``styles`` function defined in this module.
    exact_sim_idx: int
        The index of the simulation to use for the exact solution.
        Default is 0.
    """
    ls = styles(sims)
    if isinstance(ls, collections.abc.Iterable):
        ls = iter(ls)
    if exact is not None:
        exact(sims[exact_sim_idx], **next(ls))
    for s in sims:
        method(s, label=s.get_labels(labels), **next(ls))         
