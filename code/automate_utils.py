r"""
Utilary functions and classes for the automate.py script.
Author: K T Prajwal Prathiksh
###
"""
# Library imports
import os
import glob
import json
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

LINESTYLE = [
    (0, (1, 1)),
    (0, (5, 5)),
    (0, (3, 5, 1, 5)),
    (0, (3, 5, 1, 5, 1, 5)),
    (0, (1, 1)),
    (0, (5, 1)),
    (0, (3, 1, 1, 1)),
    (0, (3, 1, 1, 1, 1, 1))
]

MARKER = [
    '*', 'o', '<', '+', '^', 'x', '>', '.'
]

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
    ls = [
        dict(linestyle=x[0], color=x[1])
        for x in IT.product(
            ["-", "-.", "--", ":"], 'bgrycm'
        )
    ]
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
        exact(sims[exact_sim_idx], linestyle='-', color='k')
    for s in sims:
        method(s, label=s.get_labels(labels), **next(ls))         

def plot_vline(k, n=2):
    k_by_n = k[len(k)//n]
    plt.axvline(k_by_n, color='y', linestyle='--')
    txt = f'k/{n}'
    plt.annotate(r'$' + txt + r'$', xy=(k_by_n*1.1, 1e-2), color='y')

def get_all_unique_values(list_of_dicts, key, option=None):
    """
    Get all unique values of a key in a list of dictionaries.

    Parameters
    ----------
    list_of_dicts : list of dict
        The list of dictionaries.
    key : str
        The key for which the unique values are to be returned.
    option : dict, optional
        The option to filter the list of dictionaries. The default is None.

    Returns
    -------
    res : list
        The list of unique values of the key in the list of dictionaries.
    
    Example
    -------
    >>> get_all_unique_values([{'a': 1, 'b': 2}, {'a': 2, 'b': 2}], 'a')
    [1, 2]
    >>> get_all_unique_values([{'a': 1, 'b': 1}, {'a': 2, 'b': 2}], 'a', {'b': 2})
    [2]
    """
    if option is None:
        res = list(set([d[key] for d in list_of_dicts]))
        res.sort()
    else:
        temp_list = []
        for k, v in option.items():
            valids = [d for d in list_of_dicts if d[k] == v]
            for d in valids:
                if d not in temp_list:
                    temp_list.append(d)
        res = list(set([d[key] for d in temp_list]))
        res.sort()
    return res

def get_cpu_time(case):
    """
    Get the CPU time from the info file.

    Parameters
    ----------
    case : Case
        The case for which the CPU time is to be returned.
    
    Returns
    -------
    cpu_time : float
        The CPU time of the case.
    """
    info = glob.glob(case.input_path('*.info'))[0]
    with open(info) as fp:
        data = json.load(fp)
    return round(data['cpu_time'], 2)

def get_label_from_scheme(scheme):
    """
    Get the label for a given scheme.

    Parameters
    ----------
    scheme : str
        The scheme for which the label is to be returned.
    
    Returns
    -------
    label : str
        The label for the given scheme.
    """
    label = None
    if scheme == "tvf":
        label = 'TVF'
    elif scheme == 'ewcsph':
        label = 'EWCSPH'
    elif scheme == 'delta_plus':
        label = '$\delta^{+}SPH$'
    elif scheme == 'tsph':
        label = 'L-IPST-C'
    elif scheme == 'edac':
        label = 'EDAC'
    elif scheme == 'deltales':
        label = r'$\delta$-LES-SPH $(\rho_c)$'
    elif scheme == 'deltales_sd':
        label = r'$\delta$-LES-SPH $(\rho)$'
    elif scheme == 'k_eps':
        label = '$k-\epsilon$'
    elif scheme == 'ok2022':
        label = 'SPH-LES'
    elif scheme == 'mon2017':
        label = r'SPH-$\epsilon$'

    if label is None:
        raise NotImplementedError(f'No label for: {scheme}')
    else:
        return fr'{label}'
