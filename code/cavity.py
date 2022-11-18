"""Lid driven cavity using the Transport Velocity formulation. (10 minutes)
"""

import os

# PySPH imports
from pysph.base.utils import get_particle_array
from pysph.solver.application import Application

from pysph.sph.scheme import TVFScheme, SchemeChooser
from pysph.sph.wc.edac import EDACScheme

# numpy
import numpy as np

# domain and reference values
L = 1.0
Umax = 1.0
c0 = 10 * Umax
rho0 = 1.0
p0 = c0 * c0 * rho0

# Numerical setup
hdx = 1.0


class LidDrivenCavity(Application):
    def add_user_options(self, group):
        group.add_argument(
            "--nx", action="store", type=int, dest="nx", default=50,
            help="Number of points along x direction."
        )
        group.add_argument(
            "--re", action="store", type=float, dest="re", default=100,
            help="Reynolds number (defaults to 100)."
        )
        self.n_avg = 5
        group.add_argument(
            "--n-vel-avg", action="store", type=int, dest="n_avg",
            default=None,
            help="Average velocities over these many saved timesteps."
        )

    def consume_user_options(self):
        nx = self.options.nx
        if self.options.n_avg is not None:
            self.n_avg = self.options.n_avg
        self.dx = L / nx
        self.re = self.options.re
        h0 = hdx * self.dx
        self.nu = Umax * L / self.re
        dt_cfl = 0.25 * h0 / (c0 + Umax)
        dt_viscous = 0.125 * h0**2 / self.nu
        dt_force = 1.0
        self.tf = 10.0
        self.dt = min(dt_cfl, dt_viscous, dt_force)

    def configure_scheme(self):
        h0 = hdx * self.dx
        if self.options.scheme == 'tvf':
            self.scheme.configure(h0=h0, nu=self.nu)
        elif self.options.scheme == 'edac':
            self.scheme.configure(h=h0, nu=self.nu)
        self.scheme.configure_solver(tf=self.tf, dt=self.dt, pfreq=500)

    def create_scheme(self):
        tvf = TVFScheme(
            ['fluid'], ['solid'], dim=2, rho0=rho0, c0=c0, nu=None,
            p0=p0, pb=p0, h0=hdx
        )
        edac = EDACScheme(
            fluids=['fluid'], solids=['solid'], dim=2, c0=c0, rho0=rho0,
            nu=0.0, pb=p0, eps=0.0, h=0.0,
        )
        s = SchemeChooser(default='tvf', tvf=tvf, edac=edac)
        return s

    def create_particles(self):
        dx = self.dx
        ghost_extent = 5 * dx
        # create all the particles
        _x = np.arange(-ghost_extent - dx / 2, L + ghost_extent + dx / 2, dx)
        x, y = np.meshgrid(_x, _x)
        x = x.ravel()
        y = y.ravel()

        # sort out the fluid and the solid
        indices = []
        for i in range(x.size):
            if ((x[i] > 0.0) and (x[i] < L)):
                if ((y[i] > 0.0) and (y[i] < L)):
                    indices.append(i)

        # create the arrays
        solid = get_particle_array(name='solid', x=x, y=y)

        # remove the fluid particles from the solid
        fluid = solid.extract_particles(indices)
        fluid.set_name('fluid')
        solid.remove_particles(indices)

        print("Lid driven cavity :: Re = %d, dt = %g" % (self.re, self.dt))

        # setup the particle properties
        volume = dx * dx

        # mass is set to get the reference density of rho0
        fluid.m[:] = volume * rho0
        solid.m[:] = volume * rho0
        # Set a reference rho also, some schemes will overwrite this with a
        # summation density.
        fluid.rho[:] = rho0
        solid.rho[:] = rho0

        # smoothing lengths
        fluid.h[:] = hdx * dx
        solid.h[:] = hdx * dx

        # imposed horizontal velocity on the lid
        solid.u[:] = 0.0
        solid.v[:] = 0.0

        for i in range(solid.get_number_of_particles()):
            if solid.y[i] > L:
                solid.u[i] = Umax

        # add requisite properties to the arrays:
        self.scheme.setup_properties([fluid, solid])

        # volume is set as dx^2
        fluid.V[:] = 1. / volume
        solid.V[:] = 1. / volume

        return [fluid, solid]

    def post_process(self, info_fname):
        try:
            import matplotlib
            matplotlib.use('Agg')
            from matplotlib import pyplot as plt
        except ImportError:
            print("Post processing requires matplotlib.")
            return
        if self.rank > 0:
            return
        info = self.read_info(info_fname)
        if len(self.output_files) == 0:
            return

        # Plot KE history
        t, ke = self._plot_ke_history()

        # Plot the velocity profile
        tf, x, ui, vi, ui_c, vi_c = self._plot_velocity()

        # Plot energy spectrum
        k, Ek0, Ekf = self._plot_energy_spectrum()

        # Save data
        res = os.path.join(self.output_dir, "results.npz")
        np.savez(
            res, t=t, ke=ke, x=x, u=ui, v=vi, u_c=ui_c, v_c=vi_c, k=k, Ek0=Ek0,
            Ekf=Ekf
        )

    def _plot_ke_history(self):
        from pysph.tools.pprocess import get_ke_history
        from matplotlib import pyplot as plt
        t, ke = get_ke_history(self.output_files, 'fluid')
        plt.clf()
        plt.plot(t, ke)
        plt.xlabel('t')
        plt.ylabel('Kinetic energy')
        fig = os.path.join(self.output_dir, "ke_history.png")
        plt.savefig(fig, dpi=300)
        return t, ke

    def _plot_velocity(self):
        from pysph.tools.interpolator import Interpolator
        from pysph.solver.utils import load
        from pysph.examples.ghia_cavity_data import get_u_vs_y, get_v_vs_x
        # interpolated velocities
        _x = np.linspace(0, 1, 101)
        xx, yy = np.meshgrid(_x, _x)

        # take the last solution data
        fname = self.output_files[-1]
        data = load(fname)
        tf = data['solver_data']['t']
        interp = Interpolator(list(data['arrays'].values()), x=xx, y=yy)
        ui = np.zeros_like(xx)
        vi = np.zeros_like(xx)
        # Average out the velocities over the last n_avg timesteps
        for fname in self.output_files[-self.n_avg:]:
            data = load(fname)
            tf = data['solver_data']['t']
            interp.update_particle_arrays(list(data['arrays'].values()))
            _u = interp.interpolate('u')
            _v = interp.interpolate('v')
            _u.shape = 101, 101
            _v.shape = 101, 101
            ui += _u
            vi += _v

        ui /= self.n_avg
        vi /= self.n_avg

        # velocity magnitude
        self.vmag = vmag = np.sqrt(ui**2 + vi**2)
        import matplotlib.pyplot as plt

        f = plt.figure()

        plt.streamplot(
            xx, yy, ui, vi, density=(2, 2),  # linewidth=5*vmag/vmag.max(),
            color=vmag
        )
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.colorbar()
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.title('Streamlines at %s seconds' % tf)
        fig = os.path.join(self.output_dir, 'streamplot.png')
        plt.savefig(fig, dpi=300)

        f = plt.figure()

        ui_c = ui[:, 50]
        vi_c = vi[50]

        s1 = plt.subplot(211)
        s1.plot(ui_c, _x, label='Computed')

        y, data = get_u_vs_y()
        if self.re in data:
            s1.plot(data[self.re], y, 'o', fillstyle='none',
                    label='Ghia et al.')
        s1.set_xlabel(r'$v_x$')
        s1.set_ylabel(r'$y$')
        s1.legend()

        s2 = plt.subplot(212)
        s2.plot(_x, vi_c, label='Computed')
        x, data = get_v_vs_x()
        if self.re in data:
            s2.plot(x, data[self.re], 'o', fillstyle='none',
                    label='Ghia et al.')
        s2.set_xlabel(r'$x$')
        s2.set_ylabel(r'$v_y$')
        s2.legend()

        fig = os.path.join(self.output_dir, 'centerline.png')
        plt.savefig(fig, dpi=300)
        return tf, _x, ui, vi, ui_c, vi_c

    def _plot_energy_spectrum(self, Ni=101):
        from energy_spectrum import (
            compute_energy_spectrum, compute_scalar_energy_spectrum,
            velocity_intepolator
        )
        from pysph.base.kernels import WendlandQuinticC4

        # Interpolate initial and final states of velocity.
        t0, u0, v0 = velocity_intepolator(
            self.output_files[0], dim=2, Ni=Ni,
            kernel=WendlandQuinticC4(dim=2),
            domain_manager=self.create_domain()
        )

        tf, uf, vf = velocity_intepolator(
            self.output_files[-1], dim=2, Ni=Ni,
            kernel=WendlandQuinticC4(dim=2),
            domain_manager=self.create_domain()
        )

        # Inital energy spectrum
        EK_U0, EK_V0, _ = compute_energy_spectrum(u0, v0, w=None, U0=1)
        k0, Ek0 = compute_scalar_energy_spectrum(EK_U0, EK_V0, EK_W=None)

        # Final energy spectrum
        EK_Uf, EK_Vf, _ = compute_energy_spectrum(uf, vf, w=None, U0=1)
        kf, Ekf = compute_scalar_energy_spectrum(EK_Uf, EK_Vf, EK_W=None)

        # Save npz file
        fname = os.path.join(self.output_dir, 'energy_spectrum.npz')
        np.savez(
            fname,
            Ni=Ni, h=self.h0,
            t0=t0, u0=u0, v0=v0, EK_U0=EK_U0, EK_V0=EK_V0, k0=k0, Ek0=Ek0,
            tf=tf, uf=uf, vf=vf, EK_Uf=EK_Uf, EK_Vf=EK_Vf, kf=kf, Ekf=Ekf
        )

        # Plotting
        import matplotlib.pyplot as plt
        plt.clf()
        plt.loglog(k0, Ek0, 'k--', label=f't={t0:.2f}')
        plt.loglog(kf, Ekf, 'k-', label=f't={tf:.2f}')
        plt.xlabel(r'$k$')
        plt.ylabel(r'$E(k)$')
        plt.legend(loc='lower left')
        plt.title(f'Energy spectrum comparison (Re={self.options.re})')
        fig = os.path.join(self.output_dir, 'energy_spectrum.png')
        plt.savefig(fig, dpi=300)
        return kf, Ek0, Ekf


if __name__ == '__main__':
    app = LidDrivenCavity()
    app.run()
    app.post_process(app.info_filename)
