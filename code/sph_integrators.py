###########################################################################
# Imports
###########################################################################
# PySPH Import
from pysph.sph.integrator import Integrator
from pysph.sph.integrator_step import IntegratorStep

# Math imports
from math import sqrt

###########################################################################
# Code
###########################################################################
### Predict-Evaluate-Correct Integrator------------------------------------
class PECIntegrator(Integrator):
    def one_timestep(self, t, dt):
        # Initialise `q^{n}` & `v^{n}`
        self.initialize()

        # Predict
        self.stage1()
        self.update_domain()
        # Call any post-stage functions
        self.do_post_stage(0.5*dt, 1)

        # Correct
        self.compute_accelerations()        
        self.stage2()
        self.update_domain()
        # Call any post-stage functions.
        self.do_post_stage(dt, 2)

### Runge-Kutta Second-Order Integrator------------------------------------
class RK2Integrator(Integrator):
    def one_timestep(self, t, dt):
        # Initialise `U^{n}`
        self.initialize()

        # Stage 1 - Compute and store `U^{1}`
        self.compute_accelerations()
        self.stage1()
        self.update_domain()
        # Call any post-stage functions
        self.do_post_stage(0.5*dt, 1)

        # Stage 2 - Compute and store `U^{n+1}` 
        self.compute_accelerations()
        self.stage2()
        self.update_domain()
        # Call any post-stage functions
        self.do_post_stage(dt, 2)

class RK2Stepper(IntegratorStep):
    def initialize(
        self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
        d_u0, d_v0, d_w0, d_u, d_v, d_w,
        d_rhoc, d_rhoc0
    ):
        # Initialise `U^{n}`
        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]
        d_z0[d_idx] = d_z[d_idx]

        d_u0[d_idx] = d_u[d_idx]
        d_v0[d_idx] = d_v[d_idx]
        d_w0[d_idx] = d_w[d_idx]

        d_rhoc0[d_idx] = d_rhoc[d_idx]

    def stage1(
        self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
        d_u0, d_v0, d_w0, d_u, d_v, d_w, d_au, d_av, d_aw,
        d_rhoc, d_arho, d_rhoc0,
        dt
    ):
        dtb2 = 0.5*dt

        # Compute `U^{1}`
        d_x[d_idx] = d_x0[d_idx] + dtb2*d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dtb2*d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dtb2*d_w[d_idx]

        d_u[d_idx] = d_u0[d_idx] + dtb2*d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dtb2*d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dtb2*d_aw[d_idx]

        d_rhoc[d_idx] = d_rhoc0[d_idx] + dtb2*d_arho[d_idx]

    def stage2(
        self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
        d_u0, d_v0, d_w0, d_u, d_v, d_w, d_au, d_av, d_aw,
        dt, d_rhoc, d_arho, d_rhoc0
    ):
        # Compute `U^{n+1}`
        d_x[d_idx] = d_x0[d_idx] + dt*d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dt*d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dt*d_w[d_idx]

        d_u[d_idx] = d_u0[d_idx] + dt*d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dt*d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dt*d_aw[d_idx]

        d_rhoc[d_idx] = d_rhoc0[d_idx] + dt*d_arho[d_idx]

class RK2StepperEDAC(IntegratorStep):
    def initialize(
        self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
        d_u0, d_v0, d_w0, d_u, d_v, d_w, d_p, d_p0
    ):
        # Initialise `U^{n}`
        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]
        d_z0[d_idx] = d_z[d_idx]

        d_u0[d_idx] = d_u[d_idx]
        d_v0[d_idx] = d_v[d_idx]
        d_w0[d_idx] = d_w[d_idx]

        d_p0[d_idx] = d_p[d_idx]

    def stage1(
        self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
        d_u0, d_v0, d_w0, d_u, d_v, d_w, d_au, d_av, d_aw,
        dt, d_p, d_ap, d_p0
    ):
        dtb2 = 0.5*dt

        # Compute `U^{1}`
        d_x[d_idx] = d_x0[d_idx] + dtb2*d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dtb2*d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dtb2*d_w[d_idx]

        d_u[d_idx] = d_u0[d_idx] + dtb2*d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dtb2*d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dtb2*d_aw[d_idx]

        d_p[d_idx] = d_p0[d_idx] + dtb2*d_ap[d_idx]

    def stage2(
        self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
        d_u0, d_v0, d_w0, d_u, d_v, d_w, d_au, d_av, d_aw,
        dt, d_p, d_ap, d_p0
    ):
        # Compute `U^{n+1}`
        d_x[d_idx] = d_x0[d_idx] + dt*d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dt*d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dt*d_w[d_idx]

        d_u[d_idx] = d_u0[d_idx] + dt*d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dt*d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dt*d_aw[d_idx]

        d_p[d_idx] = d_p0[d_idx] + dt*d_ap[d_idx]

### Runge-Kutta Third-Order Integrator-------------------------------------
class RK3Integrator(Integrator):
    def one_timestep(self, t, dt):
        # Initialise `U^{n}`
        self.initialize()

        # Stage 1 - Compute and store `U^{1}`
        self.compute_accelerations()
        self.stage1()
        self.update_domain()
        # Call any post-stage functions
        self.do_post_stage(dt/3., 1)

        # Stage 2 - Compute and store `U^{2}` 
        self.compute_accelerations()
        self.stage2()
        self.update_domain()
        # Call any post-stage functions
        self.do_post_stage((2./3.)*dt, 2)

        # Stage 3 - Compute and store `U^{n+1}` 
        self.compute_accelerations()
        self.stage3()
        self.update_domain()
        # Call any post-stage functions
        self.do_post_stage(dt, 3)

### Runge-Kutta Third-Order Integrator Stepper-----------------------------
class RK3Stepper(IntegratorStep):
    def initialize(
        self, d_idx, d_x0, d_y0, d_z0, d_xi, d_yi, d_zi, d_x, d_y, d_z, 
        d_u0, d_v0, d_w0, d_ui, d_vi, d_wi, d_u, d_v, d_w, 
        d_rhoc, d_rhoc0, d_rhoci,
    ):
        # Initialise `U^{n}`
        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]
        d_z0[d_idx] = d_z[d_idx]

        d_u0[d_idx] = d_u[d_idx]
        d_v0[d_idx] = d_v[d_idx]
        d_w0[d_idx] = d_w[d_idx]

        d_rhoc0[d_idx] = d_rhoc[d_idx]

        # Initialise `U^{i}`
        d_xi[d_idx] = 0.0
        d_yi[d_idx] = 0.0
        d_zi[d_idx] = 0.0

        d_ui[d_idx] = 0.0
        d_vi[d_idx] = 0.0
        d_wi[d_idx] = 0.0
        
        d_rhoci[d_idx] = 0.0

    def stage1(
        self, d_idx, d_x0, d_y0, d_z0, d_xi, d_yi, d_zi, d_x, d_y, d_z,
        d_u0, d_v0, d_w0, d_ui, d_vi, d_wi, d_u, d_v, d_w, d_au, d_av, d_aw,
        d_rhoc, d_arho, d_rhoc0, d_rhoci,
        dt
    ):
        dtb3 = dt/3.

        # Store `f(U^{n})`
        d_xi[d_idx] = d_u[d_idx]
        d_yi[d_idx] = d_v[d_idx]
        d_zi[d_idx] = d_w[d_idx]

        d_ui[d_idx] = d_au[d_idx]
        d_vi[d_idx] = d_av[d_idx]
        d_wi[d_idx] = d_aw[d_idx]

        d_rhoci[d_idx] = d_arho[d_idx]

        # Compute `U^{1}`
        d_x[d_idx] = d_x0[d_idx] + dtb3*d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dtb3*d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dtb3*d_w[d_idx]

        d_u[d_idx] = d_u0[d_idx] + dtb3*d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dtb3*d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dtb3*d_aw[d_idx]

        d_rhoc[d_idx] = d_rhoc0[d_idx] + dtb3*d_arho[d_idx]

    def stage2(
        self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
        d_u0, d_v0, d_w0, d_u, d_v, d_w, d_au, d_av, d_aw,
        d_rhoc, d_arho, d_rhoc0,
        dt,
    ):
        dt2b3 = (2./3.)*dt

        # Compute `U^{2}`
        d_x[d_idx] = d_x0[d_idx] + dt2b3*d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dt2b3*d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dt2b3*d_w[d_idx]

        d_u[d_idx] = d_u0[d_idx] + dt2b3*d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dt2b3*d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dt2b3*d_aw[d_idx]

        d_rhoc[d_idx] = d_rhoc0[d_idx] + dt2b3*d_arho[d_idx]

    def stage3(
        self, d_idx, d_x0, d_y0, d_z0, d_xi, d_yi, d_zi, d_x, d_y, d_z,
        d_u0, d_v0, d_w0, d_ui, d_vi, d_wi, d_u, d_v, d_w, d_au, d_av, d_aw,
        d_rhoc, d_arho, d_rhoc0, d_rhoci,
        dt,
    ):
        dtb4 = dt/4.0
        
        # Compute `U^{n+1}`
        d_x[d_idx] = d_x0[d_idx] + dtb4*(d_xi[d_idx] + 3.*d_u[d_idx])
        d_y[d_idx] = d_y0[d_idx] + dtb4*(d_yi[d_idx] + 3.*d_v[d_idx])
        d_z[d_idx] = d_z0[d_idx] + dtb4*(d_zi[d_idx] + 3.*d_w[d_idx])

        d_u[d_idx] = d_u0[d_idx] + dtb4*(d_ui[d_idx] + 3.*d_au[d_idx])
        d_v[d_idx] = d_v0[d_idx] + dtb4*(d_vi[d_idx] + 3.*d_av[d_idx])
        d_w[d_idx] = d_w0[d_idx] + dtb4*(d_wi[d_idx] + 3.*d_aw[d_idx])

        d_rhoc[d_idx] = d_rhoc0[d_idx] + dtb4*(d_rhoci[d_idx] + 3.*d_arho[d_idx])

### Runge-Kutta Fourth-Order Integrator------------------------------------
class RK4Integrator(Integrator):
    def one_timestep(self, t, dt):
        # Initialise `U^{n}`
        self.initialize()

        # Stage 1 - Compute and store `U^{1}`
        self.compute_accelerations()
        self.stage1()
        self.update_domain()
        # Call any post-stage functions
        self.do_post_stage(0.5*dt, 1)

        # Stage 2 - Compute and store `U^{2}` 
        self.compute_accelerations()
        self.stage2()
        self.update_domain()
        # Call any post-stage functions
        self.do_post_stage(0.5*dt, 2)

        # Stage 3 - Compute and store `U^{3}` 
        self.compute_accelerations()
        self.stage3()
        self.update_domain()
        # Call any post-stage functions
        self.do_post_stage(dt, 3)

        # Stage 4 - Compute and store `U^{n+1}` 
        self.compute_accelerations()
        self.stage4()
        self.update_domain()
        # Call any post-stage functions
        self.do_post_stage(dt, 4)

### Runge-Kutta Fourth-Order Integrator Stepper----------------------------
class RK4Stepper(IntegratorStep):
    def initialize(
        self, d_idx, d_x0, d_y0, d_z0, d_xi, d_yi, d_zi, d_x, d_y, d_z, 
        d_u0, d_v0, d_w0, d_ui, d_vi, d_wi, d_u, d_v, d_w, 
        d_rhoc, d_rhoc0, d_rhoci,
    ):
        # Initialise `U^{n}`
        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]
        d_z0[d_idx] = d_z[d_idx]

        d_u0[d_idx] = d_u[d_idx]
        d_v0[d_idx] = d_v[d_idx]
        d_w0[d_idx] = d_w[d_idx]

        d_rhoc0[d_idx] = d_rhoc[d_idx]

        # Initialise `U^{i}`
        d_xi[d_idx] = 0.0
        d_yi[d_idx] = 0.0
        d_zi[d_idx] = 0.0

        d_ui[d_idx] = 0.0
        d_vi[d_idx] = 0.0
        d_wi[d_idx] = 0.0
        
        d_rhoci[d_idx] = 0.0

    def stage1(
        self, d_idx, d_x0, d_y0, d_z0, d_xi, d_yi, d_zi, d_x, d_y, d_z,
        d_u0, d_v0, d_w0, d_ui, d_vi, d_wi, d_u, d_v, d_w, d_au, d_av, d_aw,
        d_rhoc, d_arho, d_rhoc0, d_rhoci,
        dt
    ):
        dtb2 = 0.5*dt

        # Store `f(U^{n})`
        d_xi[d_idx] = d_u[d_idx]
        d_yi[d_idx] = d_v[d_idx]
        d_zi[d_idx] = d_w[d_idx]

        d_ui[d_idx] = d_au[d_idx]
        d_vi[d_idx] = d_av[d_idx]
        d_wi[d_idx] = d_aw[d_idx]

        d_rhoci[d_idx] = d_arho[d_idx]

        # Compute `U^{1}`
        d_x[d_idx] = d_x0[d_idx] + dtb2*d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dtb2*d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dtb2*d_w[d_idx]

        d_u[d_idx] = d_u0[d_idx] + dtb2*d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dtb2*d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dtb2*d_aw[d_idx]

        d_rhoc[d_idx] = d_rhoc0[d_idx] + dtb2*d_arho[d_idx]

    def stage2(
        self, d_idx, d_x0, d_y0, d_z0, d_xi, d_yi, d_zi, d_x, d_y, d_z,
        d_u0, d_v0, d_w0, d_ui, d_vi, d_wi, d_u, d_v, d_w, d_au, d_av, d_aw,
        d_rhoc, d_arho, d_rhoc0, d_rhoci,
        dt
    ):
        dtb2 = 0.5*dt

        # Store `f(U^{n}) + 2f(U^{1})`
        d_xi[d_idx] += 2.*d_u[d_idx]
        d_yi[d_idx] += 2.*d_v[d_idx]
        d_zi[d_idx] += 2.*d_w[d_idx]

        d_ui[d_idx] += 2.*d_au[d_idx]
        d_vi[d_idx] += 2.*d_av[d_idx]
        d_wi[d_idx] += 2.*d_aw[d_idx]

        d_rhoci[d_idx] += 2.*d_arho[d_idx]

        # Compute `U^{2}`
        d_x[d_idx] = d_x0[d_idx] + dtb2*d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dtb2*d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dtb2*d_w[d_idx]

        d_u[d_idx] = d_u0[d_idx] + dtb2*d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dtb2*d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dtb2*d_aw[d_idx]

        d_rhoc[d_idx] = d_rhoc0[d_idx] + dtb2*d_arho[d_idx]

    def stage3(
        self, d_idx, d_x0, d_y0, d_z0, d_xi, d_yi, d_zi, d_x, d_y, d_z,
        d_u0, d_v0, d_w0, d_ui, d_vi, d_wi, d_u, d_v, d_w, d_au, d_av, d_aw,
        d_rhoc, d_arho, d_rhoc0, d_rhoci,
        dt
    ):
        # Store `f(U^{n}) + 2f(U^{1}) + 2f(U^{2})`
        d_xi[d_idx] += 2.*d_u[d_idx]
        d_yi[d_idx] += 2.*d_v[d_idx]
        d_zi[d_idx] += 2.*d_w[d_idx]

        d_ui[d_idx] += 2.*d_au[d_idx]
        d_vi[d_idx] += 2.*d_av[d_idx]
        d_wi[d_idx] += 2.*d_aw[d_idx]

        d_rhoci[d_idx] += 2.*d_arho[d_idx]

        # Compute `U^{3}`
        d_x[d_idx] = d_x0[d_idx] + dt*d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dt*d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dt*d_w[d_idx]

        d_u[d_idx] = d_u0[d_idx] + dt*d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dt*d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dt*d_aw[d_idx]

        d_rhoc[d_idx] = d_rhoc0[d_idx] + dt*d_arho[d_idx]

    def stage4(
        self, d_idx, d_x0, d_y0, d_z0, d_xi, d_yi, d_zi, d_x, d_y, d_z,
        d_u0, d_v0, d_w0, d_ui, d_vi, d_wi, d_u, d_v, d_w, d_au, d_av, d_aw,
        d_rhoc, d_arho, d_rhoc0, d_rhoci,
        dt,
    ):
        dtb6 = dt/6.
        
        # Compute `U^{n+1}`
        d_x[d_idx] = d_x0[d_idx] + dtb6*(d_xi[d_idx] + d_u[d_idx])
        d_y[d_idx] = d_y0[d_idx] + dtb6*(d_yi[d_idx] + d_v[d_idx])
        d_z[d_idx] = d_z0[d_idx] + dtb6*(d_zi[d_idx] + d_w[d_idx])

        d_u[d_idx] = d_u0[d_idx] + dtb6*(d_ui[d_idx] + d_au[d_idx])
        d_v[d_idx] = d_v0[d_idx] + dtb6*(d_vi[d_idx] + d_av[d_idx])
        d_w[d_idx] = d_w0[d_idx] + dtb6*(d_wi[d_idx] + d_aw[d_idx])

        d_rhoc[d_idx] = d_rhoc0[d_idx] + dtb6*(d_rhoci[d_idx] + d_arho[d_idx])