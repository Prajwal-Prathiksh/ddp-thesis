r"""
Utilary functions and classes for the automate.py script.
###
"""
# Library imports
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
    exact_sim_idx:int=0, styles=styles,
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
    exact_sim_idx: int
        The index of the simulation to use for the exact solution.
        Default is 0.
    styles: callable: returns an iterator/iterable of style keyword arguments.
        Defaults to the ``styles`` function defined in this module.
    """
    ls = styles(sims)
    if isinstance(ls, collections.abc.Iterable):
        ls = iter(ls)
    if exact is not None:
        exact(sims[exact_sim_idx], **next(ls))
    for s in sims:
        method(s, label=s.get_labels(labels), **next(ls))         

class SimulationPlotter(object):
    """
    A class to simplify plotting of simulation results for multiple
    simulations, particularly comparison plots.
    This class is built using the  `compare_runs` function to simplify plotting of simulation results.

    Each `Problem` class can be associated with a corresponding `SimulationPlotter` class. The `SimulationPlotter` class would contain all
    of the plotting methods for the `Problem` class, and a corresponding `plotter_map` dictionary that maps the method names to the
    corresponding plotting methods. The dictionary could also have optional keyword arguments that are passed to the plotter.

    Example
    -------
    >>> class PowersPlotters(SimulationPlotter):
    ...     def powers_loglog(self, **kw):
    ...         '''
    ...         Plot the powers in log-log scale.
    ...         '''
    ...         data = np.load(self.input_path('results.npz'))
    ...         plt.loglog(data['x'], data['y'], **kw)
    ...     def powers_semilogy(self, **kw):
    ...         '''
    ...         Plot the powers in semi-log scale in the y-axis.
    ...         '''
    ...         data = np.load(self.input_path('results.npz'))
    ...         plt.semilogy(data['x'], data['y'], **kw)
    ...     def powers_stem(self, **kw):
    ...         '''
    ...         Plot the powers as a stem plot.
    ...         '''
    ...         data = np.load(self.input_path('results.npz'))
    ...         plt.stem(data['x'], data['y'], **kw)
    ...     def plotter_map(self):
    ...         p_map = dict(
    ...             powers_loglog=dict(
        #TODO
    )
    >>> class Powers(Problem):
    ...     def get_name(self):
    ...         return "powers"
    ...     def setup(self):
    ...         base_cmd = 'python powers.py --output-dir $output_dir'
    ...         self.cases = [
    ...             Simulation(
    ...                 root=self.input_path(str(i)),
    ...                 base_command=base_cmd,
    ...                 power=float(i)
    ...             )
    ...             for i in range(1, 10)
    ...         ]
    ...     def run(self):
    ...         self.make_output_dir()
    

    """
    def __init__(self, out_dir):
        self.out_dir = out_dir

    # Private methods
    def _get_plot_title(
        self, title_prefix:str, title_main:str, title_suffix:str
    ):
        """
        Get the plot title, after adding the prefix and suffix.

        Parameters
        ----------
        title_prefix: str
            The prefix to add to the title.
        title_main: str
            The main title.
        title_suffix: str
            The suffix to add to the title.

        Returns
        -------
        title: str
            The title.
        """
        # Trim leading and trailing spaces for all the parts.
        title_prefix = title_prefix.strip()
        title_main = title_main.strip()
        title_suffix = title_suffix.strip()

        # Join the parts with a space in between.
        title = " ".join([title_prefix, title_main, title_suffix])

        # Use title case for the title.
        title = title.title()

        # Format the title as raw string.
        title = r"{}".format(title)

        return title

    def _get_plot_properties(self, plot_type):
        default_plotter_props = dict(
            exact=None,
            exact_sim_idx=0,
            xlabel=r'$x$',
            ylabel=r'$y$',
            styles=styles,
            plot_legend=True,
            plot_grid=False,
            axis_scale=None
        )
        check_keys = ['method', 'title_main']
        plotter_map = self.get_plotter_map()
        if not all(
            k in plotter_map[plot_type].keys()
            for k in check_keys
        ):
            raise ValueError(
                f"Plotter type - {plot_type} is missing the following "
                f"keys: {check_keys}"
            )

        plot_props = plotter_map[plot_type].copy()

        # If plot_props does have any of the default keys, then use the
        # default values.
        for k, v in default_plotter_props.items():
            if k not in plot_props.keys():
                plot_props[k] = v
        return plot_props

    def _comparison_plotter(
        self, sims, labels, method, exact, exact_sim_idx, title, xlabel, ylabel, styles, plot_legend:bool=True, plot_grid:bool=False, axis_scale:str=None,
        
    ):        
        plt.clf()
        plt.figure()
        custom_compare_runs(
            sims=sims,
            method=method,
            exact=exact,
            labels=labels,
            styles=styles,
            exact_sim_idx=exact_sim_idx
        )

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)

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

    # Public methods
    def output_path(self, *args):
        """Given any arguments relative to the output_dir return the
        absolute path.
        """
        return os.path.join(self.out_dir, *args)
        
    def get_plotter_map(self):
        return None
    
    def get_title_prefix(self):
        return ""
    
    def plot_comparison(
        self, plot_type, sims, labels, title_suffix=""
    ):
        plotter_map = self.get_plotter_map()
        if plotter_map is None:
            raise ValueError(
                "`get_plotter_map` method should return a valid dictionary."
            )

        # Check if the plot type is valid.
        if plot_type not in plotter_map.keys():
            raise ValueError(
                f"Invalid plotter type: {plot_type}. Valid types are: {plotter_map.keys()}"
            )

        # Get the plot properties.
        plot_props = self._get_plot_properties(plot_type)
        
        title=self._get_plot_title(
            title_prefix=self.get_title_prefix(),
            title_main=plot_props['title_main'],
            title_suffix=title_suffix
        )
        self._comparison_plotter(
            sims=sims,
            labels=labels,
            method=plot_props['method'],
            exact=plot_props['exact'],
            exact_sim_idx=plot_props['exact_sim_idx'],
            title=title,
            xlabel=plot_props['xlabel'],
            ylabel=plot_props['ylabel'],
            styles=plot_props['styles'],
            plot_legend=plot_props['plot_legend'],
            plot_grid=plot_props['plot_grid'],
            axis_scale=plot_props['axis_scale']
        )