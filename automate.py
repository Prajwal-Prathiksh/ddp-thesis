#NOQA
r"""
An automator script to reproduce the results of the thesis
Author: K T Prajwal Prathiksh
###
"""
# Library imports
import json
import os
import time
from itertools import cycle, product
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint

# Automan imports
from automan.api import Automator, Simulation 
from automan.api import CondaClusterManager
from automan.api import PySPHProblem
from automan.api import mdict, dprod, opts2path
from automan.utils import filter_cases, filter_by_name

# Local imports.
from code.automate_utils import (
    styles, custom_compare_runs, plot_vline, get_all_unique_values,
    get_cpu_time, get_label_from_scheme, LINESTYLE, MARKER
)

BACKEND = " --openmp "
N_CORES, N_THREADS = 4, 8

class SineVelProfilePlotters(Simulation):
    """
    Helper methods for making comparisons plots for the sinusoidal velocity
    profile problem.
    """
    def ek_loglog(self, **kw):
        """
        Plot the energy spectrum in loglog scale.
        """
        data = np.load(self.input_path('ek_00000.npz'))
        plt.loglog(data['k'], data['ek'], **kw)

    def ek_plot(self, **kw):
        """
        Plot the energy spectrum.
        """
        data = np.load(self.input_path('ek_00000.npz'))
        plt.plot(data['k'], data['ek'], **kw)
    
    def ek_loglog_no_interp(self, **kw):
        """
        Plot the energy spectrum calculated without interpolation in loglog scale.
        """
        data = np.load(self.input_path('ek_00000.npz'))
        data_wo_interp = np.load(self.input_path('initial_ek.npz'))
        kw.pop('label', None)
        plt.loglog(
            data['k'], data_wo_interp['ek'],
            label=r'$E_k$ (no interpolation)', **kw
        )

        # Plot a vertical line at the middle of the k range.
        plot_vline(data['k'], 2)
        plot_vline(data['k'], 8)
   
    def ek_loglog_exact(self, **kw):
        """
        Plot the exact energy spectrum in loglog scale.
        """
        data = np.load(self.input_path('ek_00000.npz'))
        plt.loglog(data['k'], data['ek_exact'], **kw)    
    
    def ek_plot_exact(self, **kw):
        """
        Plot the exact energy spectrum.
        """
        data = np.load(self.input_path('ek_00000.npz'))
        plt.plot(data['k'], data['ek_exact'], **kw)
    
    def l2_error(self, **kw):
        """
        Plot the L_2 error (wrt exact solution) in loglog scale.
        """
        data = np.load(self.input_path('ek_00000.npz'))
        plt.loglog(data['k'], data['l2_error'], **kw)

    def l2_error_no_interp(self, **kw):
        """
        Plot the L_2 error of the solution calculated without interpolation vs
        the exact solution in loglog scale.
        """
        data = np.load(self.input_path('ek_00000.npz'))
        data_wo_interp = np.load(self.input_path('initial_ek.npz'))
        l2_error_expected = data['ek_exact'] - data_wo_interp['ek']
        l2_error_expected = np.sqrt(l2_error_expected**2)

        kw.pop('label', None)
        plt.loglog(
            data['k'], l2_error_expected,
            label=r'$L_2$ error (no interpolation)', **kw
        )

        # Plot a vertical line at the middle of the k range.
        plot_vline(data['k'], 2)
        plot_vline(data['k'], 8)    
    
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
            ymin, ymax = plt.ylim()
            ymin = max(ymin, 1e-7)
            ymax = min(ymax, 1e-1)
            plt.ylim(ymin, ymax)
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
        def get_opts():
            perturb_opts = mdict(
                perturb=[0.01], hdx=[1.2, 3],
                i_radius_scale=[1.2, 3]
            )
            dim_nx_opts = mdict(dim=[1], nx=[701, 1001], n_freq=[350])
            dim_nx_opts += mdict(dim=[2], nx=[701, 801], n_freq=[350])
            dim_nx_opts += mdict(dim=[3], nx=[71, 101], n_freq=[35])

            all_options = dprod(perturb_opts, dim_nx_opts)
            KERNEL_CHOICES = ['WendlandQuinticC4']
            INTERPOLATING_METHOD_CHOICES = ['sph', 'shepard', 'order1',]
            
            i_kernel_opts = mdict(i_kernel=KERNEL_CHOICES)
            i_method_opts = mdict(i_method=INTERPOLATING_METHOD_CHOICES)
            interpolator_opts = dprod(i_kernel_opts, i_method_opts)

            all_options = dprod(all_options, interpolator_opts)
            return all_options

        all_options = get_opts()

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
            nx = [701, 701, 71],
            n_freq=[350, 350, 35],
        )
        for dim, nx, n_freq in zip(tmp['dim'], tmp['nx'], tmp['n_freq']):
            def _temp_plotter(fcases, title_suffix, labels):
                if len(fcases) == 0:
                    return
                print(f"title_suffix = {title_suffix}")
                print(f"labels = {labels}")
                self.plot_energy_spectrum(
                    fcases, labels, plt_type="l2_error",
                    title_suffix=title_suffix, plot_grid=True
                )
                self.plot_energy_spectrum(
                    fcases, labels, plt_type="loglog", 
                    title_suffix=title_suffix, plot_grid=True
                )
            # Set 1
            fcases = filter_cases(
                self.cases, dim=dim, nx=nx, n_freq=n_freq, i_radius_scale=3
            )
            title_suffix = f"dim={dim}, nx={nx}, n_freq={n_freq}, " \
                            f"i_radius_scale=3"
            labels = ['i_method', 'hdx']
            _temp_plotter(fcases, title_suffix, labels)

            # Set 2
            fcases = filter_cases(
                self.cases, dim=dim, n_freq=n_freq, hdx=1.2,
                i_radius_scale=3
            )
            title_suffix = f"dim={dim}, n_freq={n_freq}, hdx=1.2, " \
                            f"i_radius_scale=3"
            labels = ['i_method', 'nx']
            _temp_plotter(fcases, title_suffix, labels)
            
            # Set 3
            fcases = filter_cases(
                self.cases, dim=dim, nx=nx, n_freq=n_freq, hdx=1.2
            )
            title_suffix = f"dim={dim}, nx={nx}, n_freq={n_freq}, " \
                            f"hdx=1.2"
            labels = ['i_method', 'i_radius_scale']
            _temp_plotter(fcases, title_suffix, labels)

class TGV2DSchemeComparison(PySPHProblem):
    def get_name(self):
        """
        Problem name.
        """
        return "tgv_2d_scheme_comparison"
    
    def setup(self):
        """
        Setup the problem.
        """
        base_cmd = "python code/taylor_green.py" + BACKEND
        # scheme_opts = mdict(scheme=['edac', 'mon2017', 'ok2022'])
        scheme_opts = mdict(
            scheme=['tsph'], method=['sd'], scm=['wcsph'], pst_freq=[10]
        )
        # scheme_opts += mdict(scheme=['deltales'], les_no_pst=[None])
        # scheme_opts += mdict(
        #     scheme=['deltales'], les_no_pst=[None], les_no_tc=[None]
        # )
        scheme_opts += mdict(scheme=['deltales'], pst_freq=[10, 50])
        scheme_opts += mdict(scheme=['deltales_sd'], pst_freq=[10, 50])

        integrator_opts = mdict(
            integrator=['pec'], integrator_dt_mul_fac=[1]
        )
        # integrator_opts += mdict(
            # integrator=['rk2'], integrator_dt_mul_fac=[2]
        # )
        # integrator_opts += mdict(
        #     integrator=['rk3'], integrator_dt_mul_fac=[3]
        # )
        # integrator_opts += mdict(
        #     integrator=['rk4'], integrator_dt_mul_fac=[4]
        # )
        res_opts = mdict(
            re=[1000, 10_000, 50_000, 100_000], 
            tf=[4], n_o_files=[50], nx=[50],
            c0_fac=[20], hdx=[2]
        )

        self.sim_opts = sim_opts = dprod(
            scheme_opts,
            dprod(integrator_opts, res_opts)
        )

        self.case_info = {}
        for i in range(len(sim_opts)):
            sim_name = opts2path(
                sim_opts[i],
                keys=[
                    'scheme', 'integrator', 'integrator_dt_mul_fac', 're',
                    'c0_fac', 'nx', 'les_no_pst', 'les_no_tc', 'pst_freq'
                ],
                kmap=dict(
                    integrator_dt_mul_fac='dtmul', c0_fac='c0',
                    les_no_pst='no_pst', les_no_tc='no_tic',
                    pst_freq='pst'
                )
            ).replace('None_', '')
            self.case_info[sim_name] = sim_opts[i]
        
        # Setup cases
        self.cases = [
            Simulation(
                root=self.input_path(name),
                base_command=base_cmd,
                job_info=dict(n_core=4, n_thread=8),
                **kw
            ) for name, kw in self.case_info.items()
        ]
        print(len(self.cases), 'cases created')

        for case in self.cases:
            self.case_info[case.name]['case'] = case
    
    def run(self):
        """
        Run the problem.
        """
        self.make_output_dir()

        # Unpack simulation opts
        sim_opts = self.sim_opts
        res = get_all_unique_values(sim_opts, 're')
        integrators = get_all_unique_values(sim_opts, 'integrator')
        integrator_dt_mul_facs = get_all_unique_values(
            sim_opts, 'integrator_dt_mul_fac'
        )
        print(res, integrators, integrator_dt_mul_facs)

        # Make plots
        KIND_CHOICES = ['ke', 'decay', 'linf', 'l1', 'p_l1', 'lm', 'am']

        # min_dtmf = 1
        # if min_dtmf:
        #     integrator_dt_mul_facs.sort()
        # else:
        #     integrator_dt_mul_facs.sort(reverse=True)
        # for re in res:
        #     for k in KIND_CHOICES:
        #         filtered_cases = []
        #         for intg in integrators:
        #             for dtmf in integrator_dt_mul_facs:
        #                 fcases = filter_cases(
        #                     self.cases, re=re, integrator=intg,
        #                     integrator_dt_mul_fac=dtmf
        #                 )
        #                 if len(fcases) >= 1:
        #                     filtered_cases.extend(fcases)
        #                     break
        #         if len(filtered_cases) == 0:
        #             continue
        #         fname = f"{k}_re_{re}.png"
        #         self.plot_sim_prop_history(
        #             cases=filtered_cases, kind=k, fname=fname
        #         )


        # for re in res:
        #     for intg in integrators:
        #             for k in KIND_CHOICES:
        #                 fcases = filter_cases(
        #                     self.cases, re=re, integrator=intg,
        #                 )
        #                 if len(fcases) == 0:
        #                     continue
        #                 fname = f"{k}_re_{re}_{intg}.png"
        #                 self.plot_sim_prop_history(
        #                     cases=fcases, kind=k, fname=fname
        #                 )

        for re in res:
            for k in KIND_CHOICES:
                fcases = filter_cases(self.cases, re=re)
                if len(fcases)  == 0:
                    continue
                fname = f"{k}_re_{re}.png"
                self.plot_sim_prop_history(
                    cases=fcases, kind=k, fname=fname
                )
    
    def get_sim_prop_history(self, case, kind):
        """
        Get the simulation property history.
        """
        KIND_CHOICES = ['ke', 'decay', 'linf', 'l1', 'p_l1', 'lm', 'am']
        if kind not in KIND_CHOICES:
            raise ValueError(f"kind must be one of {KIND_CHOICES}")
        t = case.data['t']
        prop = case.data[kind]
        return t, prop
    
    def get_expected_sim_prop_history(self, cases, kind):
        """
        Get the expected simulation property history.
        """
        KIND_CHOICES = dict(ke='ke_ex', decay='decay_ex')
        if kind not in KIND_CHOICES.keys():
            raise ValueError(f"kind must be one of {KIND_CHOICES.keys()}")
        
        t, prop = [], []
        for case in cases:
            _t, _prop = case.data['t'], case.data[KIND_CHOICES[kind]]
            t.append(_t)
            prop.append(_prop)
        
        # Get index of max time
        idx = np.argmax([_t[-1] for _t in t])

        # Return the expected property history
        return t[idx], prop[idx]
    
    def plot_sim_prop_history(self, cases, kind, fname, title_suffix=''):
        """
        Plot the simulation property history.
        """
        plot_exact = False
        if kind in ['ke', 'decay']:       
            # Get the expected simulation property history
            t_exp, prop_exp = self.get_expected_sim_prop_history(cases, kind)
            plot_exact = True

        # Set plotter method
        if kind in ['ke', 'decay']:
            plt_method = plt.semilogy
        else:
            plt_method = plt.plot
        
        plt.figure()
        # Plot the simulation property history
        if plot_exact:
            plt_method(t_exp, prop_exp, 'k--', label='exact')

        for case in cases:
            t, prop = self.get_sim_prop_history(case, kind)
            plt_method(t, prop, label=case.name)
        
        plt.xlabel('t')
        plt.ylabel(kind)
        plt.title(f"{kind} history {title_suffix}")
        plt.legend()
        plt.grid()
        plt.savefig(self.output_path(fname), dpi=300, bbox_inches='tight')
        plt.close()
        
class TGV2DExtForceSchemeComparison(PySPHProblem):
    def get_name(self):
        """
        Problem name.
        """
        return "tgv_2d_ext_force_scheme_comparison"
    
    def setup(self):
        """
        Setup the problem.
        """
        base_cmd = "python code/taylor_green.py --ext-forcing " + BACKEND
    
        # scheme_opts = mdict(scheme=['edac'])
        # scheme_opts += mdict(
        #     scheme=['tsph'], method=['no_sd'], scm=['edac'], pst_freq=[10],
        # )
        # scheme_opts += mdict(scheme=['mon2017'])
        scheme_opts = mdict(
            scheme=['tsph'], method=['sd'], scm=['wcsph'], pst_freq=[10],
            n_o_files=[50]
        )
        integrator_opts = mdict(
            integrator=['pec'], integrator_dt_mul_fac=[1],
        )
        integrator_opts += mdict(
            integrator=['rk3'], integrator_dt_mul_fac=[3],
        )
        integrator_opts += mdict(
            integrator=['rk4'], integrator_dt_mul_fac=[4],
        )
        res_opts = mdict(
            re=[10_000, 100_000, 1_000_000],
            nx=[200], tf=[6.], c0_fac=[80]
        )

        self.sim_opts = sim_opts = dprod(
            scheme_opts,
            dprod(integrator_opts, res_opts)
        )

        self.case_info = {}
        for i in range(len(sim_opts)):
            sim_name = opts2path(
                sim_opts[i],
                keys=[
                    'scheme', 'integrator', 'integrator_dt_mul_fac', 're',
                    'c0_fac', 'nx', 'tf'
                ],
                kmap=dict(integrator_dt_mul_fac='dtmul', c0_fac='c0')
            )
            self.case_info[sim_name] = sim_opts[i]

        # Setup cases
        self.cases = [
            Simulation(
                root=self.input_path(name),
                base_command=base_cmd,
                job_info=dict(n_core=4, n_thread=8),
                **kw
            ) for name, kw in self.case_info.items()
        ]
        print(len(self.cases), 'cases created')

        for case in self.cases:
            self.case_info[case.name]['case'] = case
    
    def run(self):
        """
        Run the problem.
        """
        self.make_output_dir()

class TB3DExtForceSchemeComparison(PySPHProblem):
    def get_name(self):
        """
        Problem name.
        """
        return "tb_3d_ext_force_scheme_comparison"
    
    def setup(self):
        """
        Setup the problem.
        """
        base_cmd = "python code/triperiodic_beltrami.py " + BACKEND

        scheme_opts = mdict(
            scheme=['tsph'], method=['sd'], scm=['wcsph'], pst_freq=[10]
        )
        scheme_opts += mdict(
            scheme=['tsph'], method=['no_sd'], scm=['edac'], pst_freq=[10],
            c0_fac=[40]
        )

        res_opts = mdict(
            nx=[40], tf=[3.],
            re=[1_000_000]
        )
        opts = dprod(scheme_opts, res_opts)

        # Setup cases
        self.cases = [
            Simulation(
                root=self.input_path(opts2path(kw)),
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

class RunTimeDomainManager(PySPHProblem):
    def get_name(self):
        """
        Problem name.
        """
        return "runtime_domain_manager"
    
    def setup(self):
        """
        Setup the problem.
        """
        base_cmd = "python code/taylor_green.py --no-post-process " +\
            " --scheme tsph --method sd --scm wcsph --pst-freq 10 " +\
            " --nx 800 --max-steps 10 --re 500 --pfreq 100000 --tf 10000 " +\
            " --disable-output "
        
        opts = [
            # Single Core - Multi-Thread
            dict(
                n_core=-1, n_thread=i,
                backend=" --openmp " if i > 1 else " "
            )
            for i in [1, 2, 4, 8, 16, 32]
        ]
        # opts += [
        #     dict(
        #         n_core=-1, n_thread=2*i, backend=" --openmp "
        #     )
        #     for i in range(2, 6)
        # ]
        
        # Setup cases
        self.cases = [
            Simulation(
                root=self.input_path(
                    f"n_core_{kw['n_core']}_n_thread_{kw['n_thread']}_p"
                ),
                base_command=base_cmd + kw['backend'],
                job_info=dict(n_core=kw['n_core'], n_thread=kw['n_thread']),
            )
            for kw in opts
        ]
        self.cases += [
            Simulation(
                root=self.input_path(
                    f"n_core_{kw['n_core']}_n_thread_{kw['n_thread']}_no_p"
                ),
                base_command=base_cmd + kw['backend'] + " --no-periodic ",
                job_info=dict(n_core=kw['n_core'], n_thread=kw['n_thread']),
            )
            for kw in opts
        ]
        
    def create_rt_table(self):
        """
        Create the runtime table.
        """
        n_cores, n_threads, runtimes = [], [], []
        dm_runtimes, eval_runtimes = [], []
        periodic = []

        for case in self.cases:
            n_cores.append(case.job_info['n_core'])
            n_threads.append(case.job_info['n_thread'])
            periodic.append(
                'yes' if 'no-periodic' not in case.base_command else 'no'
            )
            
            # Read simulation *.info file
            fname = os.path.join(case.root, 'taylor_green.info')
            with open(fname, 'r') as f:
                data = json.load(f)
            runtimes.append(data['cpu_time'])

            # Read profile_info.csv file
            fname = os.path.join(case.root, 'profile_info.csv')
            df = pd.read_csv(fname)
            cond = df.function.str.contains('update_domain')
            if cond.sum() != 1:
                raise ValueError("Number of update_domain calls != 1")
            dm_runtimes.append(float(df[cond].time))

            cond = df.function.str.contains('eval_0')
            if cond.sum() != 1:
                raise ValueError("Number of eval_0 calls != 1")
            eval_runtimes.append(float(df[cond].time))

        # Create table
        df = pd.DataFrame(
            data=dict(
                n_cores=n_cores,
                n_threads=n_threads,
                runtime=runtimes,
                domain_manager=dm_runtimes,
                eval_0=eval_runtimes,
                periodic=periodic,
            )
        )
        fname = self.output_path('runtime_table.csv')
        df.to_csv(fname, index=False)
        print(f"Saved runtime table to {fname}")
    
    def run(self):
        """
        Run the problem.
        """
        self.make_output_dir()
        self.create_rt_table()
        
class TGV2DIntegratorComparison(PySPHProblem):
    def get_name(self):
        return 'tgv_2d_integrator_comparison'

    def _get_file(self):
        return 'code/taylor_green.py --no-plot --openmp --tf 0.1 --n-o-files 8 '

    def setup(self):
        scheme_opts = mdict(
            scheme=['tsph'], method=['sd'], scm=['wcsph'], pst_freq=[10]
        )
        scheme_opts += mdict(scheme=['deltales'], pst_freq=[10])
        scheme_opts += mdict(scheme=['deltales_sd'], pst_freq=[10])

        integrator_opts = mdict(
            integrator=['pec'], integrator_dt_mul_fac=[1], pst_freq=[10],
        )
        # integrator_opts += mdict(
        #     integrator=['rk2'], integrator_dt_mul_fac=[2], pst_freq=[10],
        #     # NOTE: dt_mul cannot be 1 for rk2. This has been hardcoded for
        #     # now. Refer below for more details.
        # )
        # integrator_opts += mdict(
        #     integrator=['rk2'], integrator_dt_mul_fac=[1], pst_freq=[10],
        #     adaptive_timestep=[None]
        #     # NOTE: dt_mul = 1 and integ = 'rk2' for appropriate plotting and
        #     # labelling. This has been hardcoded for now.
        # )
        # integrator_opts += mdict(
        #     integrator=['rk3'], integrator_dt_mul_fac=[3, 6], pst_freq=[10],
        # )
        # integrator_opts += mdict(
        #     integrator=['rk4'], integrator_dt_mul_fac=[2, 4, 8], pst_freq=[10],
        # )
        res_opts = mdict(nx=[25, 50, 100], c0_fac=[20], re=[100, 1000, 5000])
        
        self.sim_opts = sim_opts = dprod(
            scheme_opts,
            dprod(integrator_opts, res_opts)
        )

        # Unpack the simulation options
        self.schemes = get_all_unique_values(sim_opts, 'scheme')
        self.re_s = get_all_unique_values(sim_opts, 're')
        self.integrators = get_all_unique_values(sim_opts, 'integrator')
        self.nx = get_all_unique_values(sim_opts, 'nx')
        self.dt_mul_facs = get_all_unique_values(
            sim_opts, 'integrator_dt_mul_fac'
        )
        self.c0s = get_all_unique_values(sim_opts, 'c0_fac')
        self.pst_freqs = get_all_unique_values(sim_opts, 'pst_freq')

        self.case_info = {}
        for i in range(len(sim_opts)):
            sim_name = opts2path(
                sim_opts[i],
                kmap=dict(
                    integrator_dt_mul_fac='dtmul', c0_fac='c0', pst_freq='pst',
                    adaptive_timestep='adapt'
                )
            )
            self.case_info[sim_name] = sim_opts[i]

        cmd = 'python ' + self._get_file()
        self.cases = [
            Simulation(
                root=self.input_path(name),
                base_command=cmd,
                job_info=dict(n_core=4, n_thread=8),
                cache_nnps=None,
                **kw
            ) for name, kw in self.case_info.items()
        ]
        print(len(self.cases), 'cases created')
        for case in self.cases:
            self.case_info[case.name]['case'] = case

    def run(self):
        self.make_output_dir()
        for c0 in self.c0s:
            for re in self.re_s:
                self._plot_convergence(c0=c0, re=re)
        for intg in self.integrators:
            for re in self.re_s:
                if intg == 'auto':
                    continue
                self._plot_convergence_c0(intg=intg, re=re)
        
        for re in self.re_s:
            for c0 in self.c0s:
                for pst in self.pst_freqs:
                    self._plot_rt_speedup(
                        re=re, c0=c0, pst=pst, largest_dtmf_only=True
                    )
                    self._plot_rt_speedup(
                        re=re, c0=c0, pst=pst, largest_dtmf_only=False
                    )

    def calculate_l1(self, cases):
        data = {}
        for case in cases:
            l1 = case.data['l1']
            l1 = np.mean(l1)
            data[case.params['nx']] = l1
        nx_arr = np.asarray(sorted(data.keys()), dtype=float)
        l1 = np.asarray([data[x] for x in nx_arr])
        return nx_arr, l1

    def _plot_rt_speedup(
        self, c0=None, re=None, pst=None, largest_dtmf_only=True
    ):
        if c0 is None:
            c0 = max(self.c0s)
        if re is None:
            re = max(self.re_s)
        if pst is None:
            pst = max(self.pst_freqs)

        plt.figure()
        dx = None
        marker = cycle(MARKER)
        scheme = 'tsph'
        for intg in self.integrators:
            if intg == 'auto':
                continue
            if largest_dtmf_only:
                temp = get_all_unique_values(
                    self.sim_opts, 'integrator_dt_mul_fac',
                    option=dict(integrator=intg)
                )
                if intg == 'pec':
                    r_list = [1, max(temp)]
                else:
                    r_list = [max(temp)]
            else:
                r_list = self.dt_mul_facs
            for dtmf in r_list:
                rts = []
                dts = []
                for dt in self.nx:
                    cases = filter_cases(
                        self.cases, scheme=scheme, integrator=intg, nx=dt,
                        c0_fac=c0, integrator_dt_mul_fac=dtmf, re=re, pst_freq=pst
                    )
                    if len(cases) < 1:
                        continue
                    case = cases[0]
                    rts.append(get_cpu_time(case))
                    dts.append(case.params['nx'])
                if len(rts) < 1:
                    continue
                label = fr'{intg} ({dtmf}$\Delta t$)'
                if intg == 'rk2' and dtmf == 1:
                    label += f' (Adaptive)'
                plt.plot(dts, rts, label=label, marker=next(marker))


        plt.grid()
        plt.legend(loc='best')
        plt.xlabel(r'$N_x$')
        plt.ylabel(r'$RT$')
        plt.title(
            fr'CPU time - L-IPST-C Scheme (pst={pst}, c0={c0}, Re={re}) ($t_f=0.1$)'
        )
        if largest_dtmf_only:
            fname = f'pois_rt_pst_{pst}_c0_{c0}_re_{re}_largest_dtmf.png'
        else:
            fname = f'pois_rt_pst_{pst}_c0_{c0}_re_{re}.png'
        plt.savefig(self.output_path(fname), dpi=300)

    def _plot_convergence(self, c0=None, re=None):
        if c0 is None:
            c0 = min(self.c0s)
        if re is None:
            re = max(self.re_s)
        plt.figure()
        marker = cycle(MARKER)
        for scheme in self.schemes:
            for intg in self.integrators:
                for dtmf in self.dt_mul_facs:
                    for pst in self.pst_freqs:
                        cases = filter_cases(
                            self.cases, scheme=scheme, integrator=intg,
                            c0_fac=c0, integrator_dt_mul_fac=dtmf, re=re,
                            pst_freq=pst
                        )
                        if len(cases) < 1:
                            continue
                        dts, l1 = self.calculate_l1(cases)
                        dts = 1./dts
                        label = get_label_from_scheme(scheme) +\
                            fr' ({intg}) ({dtmf}$\Delta t$) (pst={pst})'
                        if intg == 'rk2' and dtmf == 1:
                            label += f' (Adaptive)'
                        plt.loglog(dts, l1, label=label, marker=next(marker))

        plt.loglog(dts, l1[0]*(dts/dts[0])**2, 'k--', linewidth=2,
                label=r'$O(h^2)$')
        plt.loglog(dts, l1[0]*(dts/dts[0]), 'k:', linewidth=2,
                label=r'$O(h)$')
        plt.grid()
        plt.legend(loc='best')
        plt.xlabel(r'$h$')
        plt.ylabel(r'$L_1$ error')
        plt.title(
            fr'Scheme convergence - $L_1$ error $(c_0={c0})$ (Re={re}) ($t_f=0.1$)'
        )
        plt.savefig(
            self.output_path(f'dt_pois_conv_c0_{c0}_re_{re}.png'), dpi=300
        )
        plt.close()

    def _plot_convergence_c0(self, intg, re=None):
        if re is None:
            re = max(self.re_s)
        plt.figure()
        marker = cycle(MARKER)
        scheme = 'tsph'
        if intg not in self.integrators:
            return
        for c0 in self.c0s:
            for dtmf in self.dt_mul_facs:
                for pst in self.pst_freqs:
                    cases = filter_cases(
                        self.cases, scheme=scheme, integrator=intg, c0_fac=c0,
                        integrator_dt_mul_fac=dtmf, re=re, pst_freq=pst
                    )
                    if len(cases) < 1:
                        continue
                    dts, l1 = self.calculate_l1(cases)
                    dts = 1./dts
                    label = get_label_from_scheme(scheme) +\
                        fr' ({intg}) ({dtmf}$\Delta t$) (pst={pst})'
                    label += f' (c0={c0})'
                    if intg == 'rk2' and dtmf == 1:
                        label += f' (Adaptive)'
                    plt.loglog(dts, l1, label=label, marker=next(marker))
        
        plt.loglog(dts, l1[0]*(dts/dts[0])**2, 'k--', linewidth=2,
                label=r'$O(h^2)$')
        plt.loglog(dts, l1[0]*(dts/dts[0]), 'k:', linewidth=2,
                label=r'$O(h)$')
        plt.grid()
        plt.legend(loc='best')
        plt.xlabel(r'$h$')
        plt.ylabel(r'$L_1$ error')
        plt.title(
            fr'Scheme convergence - $L_1$ error (Re={re}) ($t_f=0.1$)'
        )
        fname = self.output_path(f'dt_pois_conv_c0_{intg}_re_{re}.png')
        plt.savefig(fname, dpi=300)
        plt.close()

class KEpsModelTesting(PySPHProblem):
    def get_name(self):
        return "k_eps_model_testing"

    def _get_file(self):
        return "code/test_k_eps.py --max-steps=1 "

    def setup(self):
        scheme_opts = mdict(k_eps_expand=['yes', 'no'])
        tc_opts = mdict(k_eps_test_case=[0, 1, 2])
        res_opts = mdict(
            nx=[200], re=[100], c0_fac=[10, 20, 40]
        )

        self.sim_opts = sim_opts = dprod(
            scheme_opts, dprod(tc_opts, res_opts)
        )

        self.case_info = {}
        for i in range(len(sim_opts)):
            sim_name = opts2path(
                sim_opts[i],
                kmap=dict(
                    k_eps_expand='ke_expd', k_eps_test_case='ketc',
                    c0_fac='c0',
                )
            )
            self.case_info[sim_name] = sim_opts[i]
        
        cmd = 'python ' + self._get_file()
        self.cases = [
            Simulation(
                root=self.input_path(name),
                base_command=cmd,
                job_info=dict(n_core=1, n_thread=1),
                cache_nnps=None,
                **kw
            ) for name, kw in self.case_info.items()
        ]
        print(len(self.cases), 'cases created')
        for case in self.cases:
            self.case_info[case.name]['case'] = case
    
    def run(self):
        self.make_output_dir()


if __name__ == "__main__":
    PROBLEMS = [
        SineVelProfile,
        TGV2DSchemeComparison,
        TGV2DExtForceSchemeComparison,
        TB3DExtForceSchemeComparison,
        RunTimeDomainManager,
        TGV2DIntegratorComparison,
        KEpsModelTesting
    ]
    automator = Automator(
        simulation_dir='outputs',
        output_dir=os.path.join('manuscript', 'code_figures'),
        all_problems=PROBLEMS,
        cluster_manager_factory=CondaClusterManager
    )
    tic = time.perf_counter()
    automator.run()
    toc = time.perf_counter()
    print(f"Total time taken = {toc-tic:.2f} seconds")

