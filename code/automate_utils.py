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
    
def get_colagrossi2021_fig1_ke(re):
    """
    Returns the data extracted from Figure 1 of Colagrossi et al. (2021) for
    the kinetic energy of the flow.
    """
    if re == 1e6:
        x = [0, 0.1275794476976334, 0.26829240537367094, 0.48780470523266317, 0.6735457664228459, 0.8255158036551533, 0.9043151458381081, 0.9437148169295854, 0.9943715675773322, 1.1744839816339134, 1.3264538041552867, 2.063790045915218, 2.7842401315634095, 3.4709196226755936, 4.168855549211246, 4.697936355957521, 5.401501144337708, 5.964352975041857]
        y = [0, 0.0009063363330951701, 0.017220517101224297, 0.07462232912850364, 0.16042296002871512, 0.24864047245082715, 0.3012084522857153, 0.3229606971514757, 0.3386706921134246, 0.3374622283029443, 0.3368579963977042, 0.3338368483962684, 0.33081570039483266, 0.32900299315434717, 0.3277945523933969, 0.3277945523933969, 0.3253776362972013, 0.32658610010768163]
    elif re == 1e5:
        x = [0, 0.14446517438750506, 0.3527201799792935, 0.5497183207257461, 0.7523451086058007, 0.893058066281838, 0.9606001141975891, 0.9887427057327965, 1.0675420479157514, 1.2138834380144565, 1.5009382152110675, 1.951219679774388, 2.6041277175068283, 3.842401745055959, 5.181989273900583, 5.91369665381598]
        y = [0, 0.0009063363330951701, 0.032326257108403034, 0.09939575656999483, 0.19969788404737993, 0.29335347209188833, 0.32658610010768163, 0.33444108030150854, 0.3350452891572187, 0.33081570039483266, 0.32839878429863706, 0.3223564882957655, 0.31631419229289404, 0.30845921209906707, 0.30543806409763136, 0.30362536838191084]
    elif re == 1e4:
        x = [0, 0.15009360681017303, 0.3414633151339578, 0.5778609122609538, 0.8255158036551533, 0.8986864987045062, 0.6735457664228459, 0.9549716817749213, 0.9943715675773322, 1.0731704803384192, 1.365853689957698, 2.1031895022957614, 2.8517826089010283, 3.926829519661582, 5.013133724689339, 5.958724972041058]
        y = [0, 0.0021147770940455057, 0.030513595966977725, 0.10362536838191092, 0.22145014043790534, 0.2758308044637489, 0.1477341245929669, 0.292145008281408, 0.29879151313998964, 0.29637462009332405, 0.27280965646231314, 0.22507553186934628, 0.18821751242211202, 0.16706947641206177, 0.15679756859727417, 0.151963736404883]
    else:
        x, y = None, None
        return x, y
    return x, y