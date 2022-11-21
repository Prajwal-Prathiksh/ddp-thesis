#NOQA
r"""
SPH-LES Model
#####################
References
-----------
    .. [Okraschevski2022] M. Okraschevski, N. Bürkle, R. Koch, and H.-J. Bauer,
    “Smoothed Particle Hydrodynamics Physically Reconsidered -- The Relation to
    Explicit Large Eddy Simulation and the Issue of Particle Duality,” 2022,
    [Online]. Available: http://arxiv.org/abs/2206.10127.
"""
###########################################################################
# Import
###########################################################################
from compyle.api import declare
from math import sqrt
from pysph.sph.equation import Equation, MultiStageEquations, Group
from pysph.base.utils import get_particle_array


###########################################################################
# Particle Array
###########################################################################
def get_particle_array_sph_les_fluid(constants=None, **props):
    """Returns a fluid particle array for the SPH-LES Scheme

        This sets the default properties to be::

            [
                # PySPH Default Properties
                'x', 'y', 'z', 'u', 'v', 'w', 'm', 'h', 'rho', 'p', 'au', 'av',
                'aw', 'gid', 'pid', 'tag',

                # SPH-LES Properties
                'ax', 'ay', 'az', 'V', 'tau', 'J', 'nu_t'
            ]

        Parameters:
        -----------
        constants : dict
            Dictionary of constants

        Other Parameters
        ----------------
        props : dict
            Additional keywords passed are set as the property arrays.

        See Also
        --------
        get_particle_array
    """
    # Properties required for SPH-LES Scheme
    sph_les_props = [
        'ax', 'ay', 'az', 'V', 'nu_t'
    ]

    pa = get_particle_array(
        constants=constants, additional_props=sph_les_props, **props
    )

    pa.add_property('tau', stride=9)
    pa.add_property('J', stride=9)

    # default property arrays to save out
    pa.set_output_arrays([
        'x', 'y', 'z', 'u', 'v', 'w', 'rho', 'p', 'h', 'm',
        'pid', 'gid', 'tag',
    ])
    return pa


###########################################################################
# Equations & Respective Imports
###########################################################################
# Equation of State------------------------------------------------------
class LinearBarotropicEOS(Equation):
    r"""**Linear Barotropic Equation of State**

    .. math::

        p(\rho) = p_0 + K \bigg(\frac{\rho}{\rho_0} - 1 \bigg

    where,

    .. math::

        K = \rho_0 c^2

    .. math::

        \rho_0, p_0 \in \mathbb{R}^+


    """

    def __init__(self, dest, sources, p0, rho0, K):
        r"""
        Parameters
        ----------
        p0 : float
            Reference Pressure
        rho0 : float
            Reference Density
        K : float
            Constant
        """

        self.K = K
        self.p0 = p0
        self.rho0 = rho0
        super(LinearBarotropicEOS, self).__init__(dest, sources)

    def loop(self, d_idx, d_p, d_rho):
        d_p[d_idx] = self.p0 + self.K * (d_rho[d_idx] / self.rho0 - 1.)


# Summation Density Equation----------------------------------------------
class SummationDensity(Equation):
    r"""Summation density:

    :math:`\rho_i = m_i \sum_j W_{ij}`

    :math:`V_i = (\sum_j W_{ij})^{-1}`

    """

    def initialize(self, d_idx, d_rho):
        d_rho[d_idx] = 0.0

    def loop(self, d_idx, d_rho, WIJ):
        d_rho[d_idx] += WIJ

    def post_loop(self, d_idx, d_rho, d_V, d_m):
        d_V[d_idx] /= d_rho[d_idx]
        d_rho[d_idx] *= d_m[d_idx]


# Momentum Equation------------------------------------------------------


class PreMomentumEquation(Equation):
    def __init__(
        self, dest, sources, dim, nu, rho0, turb_visc_model, DELTA, Cs=0.15,
        C_sigma=1.35
    ):
        r"""
        Parameters
        ----------
        dim : int
            Dimensions of the problem
        nu : float
            Kinematic Viscosity
        rho0 : float
            Reference Density
        turb_visc_model : str
            Turbulent Viscosity Model to be used: SMAG, SIGMA, SMAG_MCG
        DELTA : float
            Filter Length-scale
        Cs : float
            Smagorinsky Constant
        Csigma : float
            Sigma Model Constant
        """

        self.dim = int(dim)
        self.nu = nu
        self.rho0 = rho0
        self.turb_visc_model = turb_visc_model
        self.DELTA = DELTA
        self.Cs = Cs
        self.C_sigma = C_sigma

        self.lap_vel_prefactor = 2. * (2. + self.dim) * self.nu * self.rho0
        self.SMAG_prefactor = (self.Cs * self.DELTA)**2
        self.SIGMA_prefactor = (self.C_sigma * self.DELTA)**2

        turb_visc_models = ['SMAG', 'SIGMA', 'SMAG_MCG']
        if self.turb_visc_model not in turb_visc_models:
            raise Exception("Invalid Turbulent Viscosity Model.")

        super(PreMomentumEquation, self).__init__(dest, sources)

    def initialize(self, d_idx, d_J, d_tau):
        i = declare('int', 1)
        for i in range(9):
            d_J[d_idx * 9 + i] = 0.
            d_tau[d_idx * 9 + i] = 0.

    def loop(self, d_idx, s_idx, d_J, s_V, VIJ, DWIJ):

        Vj = s_V[s_idx]

        i, j = declare('int', 2)

        for i in range(3):
            for j in range(3):
                d_J[d_idx * 9 + i * 3 + j] = - VIJ[i] * DWIJ[j] * Vj

        # d_J[d_idx*9 + 0] += - VIJ[0]*DWIJ[0]*Vj # 2D
        # d_J[d_idx*9 + 1] += - VIJ[0]*DWIJ[1]*Vj # 2D
        # d_J[d_idx*9 + 2] += - VIJ[0]*DWIJ[2]*Vj

        # d_J[d_idx*9 + 3] += - VIJ[1]*DWIJ[0]*Vj # 2D
        # d_J[d_idx*9 + 4] += - VIJ[1]*DWIJ[1]*Vj # 2D
        # d_J[d_idx*9 + 5] += - VIJ[1]*DWIJ[2]*Vj

        # d_J[d_idx*9 + 6] += - VIJ[2]*DWIJ[0]*Vj
        # d_J[d_idx*9 + 7] += - VIJ[2]*DWIJ[1]*Vj
        # d_J[d_idx*9 + 8] += - VIJ[2]*DWIJ[2]*Vj

    def post_loop(self, d_idx, d_J, d_nu_t, s_idx, d_tau, s_tau):
        i, j = declare('int', 2)
        n = 3
        D = declare('matrix((3,3))')

        for i in range(n):
            for j in range(n):
                D[i * n + j] = 0.5 * (
                    d_J[d_idx * 9 + i * n + j] + d_J[d_idx * 9 + i + j * n]
                )

        if self.dim == 2:
            trace_D_square = D[0]**2 + D[4]**2 + 2 * D[1]**2
        elif self.dim == 3:
            trace_D_square = D[0]**2 + D[4]**2 + D[8]**2
            trace_D_square += 2 * (D[1]**2 + D[2]**2 + D[5]**2)

        if self.turb_visc_model == "SMAG":
            d_nu_t[d_idx] = self.SMAG_prefactor * sqrt(2 * trace_D_square)

        elif self.turb_visc_model == "SIGMA":
            if self.dim == 2:
                sigma1 = D[0] + D[4]
                sigma2 = 0.5 * (sigma1**2 - trace_D_square)
                sigma3 = D[0] * D[4] - D[1]**2
            elif self.dim == 3:
                sigma1 = D[0] + D[4] + D[8]
                sigma2 = 0.5 * (sigma1**2 - trace_D_square)

                A = D[0] * (D[4] * D[8] - D[5] * D[7])
                B = D[1] * (D[3] * D[8] - D[5] * D[6])
                C = D[2] * (D[3] * D[7] - D[4] * D[6])
                sigma3 = A - B + C

            d_nu_t[d_idx] = sigma3 * \
                (sigma1 - sigma2) * (sigma2 - sigma3) / sigma1**2
            d_nu_t[d_idx] *= self.SIGMA_prefactor

        fac = -2. * d_nu_t[d_idx] * self.rho0
        if self.dim == 2:
            d_tau[d_idx * 9 + 0] = fac * D[0]
            d_tau[d_idx * 9 + 1] = fac * D[1]
            d_tau[d_idx * 9 + 3] = fac * D[3]
            d_tau[d_idx * 9 + 4] = fac * D[4]
        elif self.dim == 3:
            for i in range(3):
                for j in range(3):
                    d_tau[d_idx * 9 + i * n + j] = fac * D[i * n + j]


class MomentumEquation(Equation):
    r"""**Momentum equation**

    .. math::

        \bar{\rho}_i \frac{\mathrm{d} \tilde{\mathbf{v}}_i}{\mathrm{~d} t} =
        -\sum_j \left(\bar{p}_j+\bar{p}_i\right) \nabla W_{ij} V_j+2(2+n) \eta
        \sum_j \frac{\tilde{\mathbf{v}}_{ij} \cdot
        \mathbf{r}_{ij}}{|\mathbf{r}_{ij}|^2} \nabla W_{ij} V_j-
        \operatorname{div}\left[\boldsymbol{\tau}_{S F S}\right]_i

    where

    .. math::

        \eta = \nu \bar{\rho}

    .. math::

        \operatorname{div} \left[\boldsymbol{\tau}_{S F S}\right]
        (\mathbf{r}, t) \approx \sum_j (\tau_{SFS, j} + \tau_{SFS, i})
        \nabla W_{ij} V_j

    .. math::

        \tau_{SFS}(\mathbf{r}, t) \approx  -2 \nu_t \bar{\rho}
        \tilde{\mathbb{D}}(\mathbf{r}, t)

    .. math::

        \tilde{\mathbb{D}}(\mathbf{r}, t) = \frac{1}{2} (\tilde{\mathbb{J}} +
        \tilde{\mathbb{J}}^T) (\mathbf{r}, t)

    .. math::

        \tilde{\mathbb{J}}(\mathbf{r}_i, t) \approx  -
        \sum_j \tilde{\mathbf{v}}_{ij} \nabla W_{ij}^T V_j

    The turbulence eddy viscosity term can be modelled through either the
    Smagorinsky model or the \sigma model, given below respectively:

    .. math::

        \nu_t = (C_S \Delta)^2 \sqrt{2
        \operatorname{tr}[\tilde{\mathbb{D}}^2]}, C_S = 0.15

    .. math::

        \nu_t = (C_{\sigma} \Delta)^2 \frac{\sigma_3(\sigma_1 -
        \sigma_2)(\sigma_2 - \sigma_3)}{\sigma_1^2}, C_{\sigma} = 1.35

    where \sigma refers to the singular values of the tensor
    \tilde{\mathbb{J}}.

    The Monoghan-Cleary-Gingold (MCG) form of the \operatorname{div}
    \left[\boldsymbol{\tau}_{S F S}\right] is given as:

    .. math::

        \operatorname{div} \left[\boldsymbol{\tau}_{S F S}\right]
        (\mathbf{r}, t) \approx 2(2+n) \sum_j \bar{\rho}_i \bar{\rho}_j
        \frac{\nu_{t,i} + \nu_{t,j}}{\bar{\rho}_i + \bar{\rho}_j}
        \frac{\tilde{\mathbf{v}}_{ij} \cdot
        \mathbf{r}_{ij}}{|\mathbf{r}_{ij}|^2}\nabla W_{ij} Vj

    """

    def __init__(self, dest, sources, dim, nu, rho0, turb_visc_model):
        r"""
        Parameters
        ----------
        dim : int
            Dimensions of the problem
        nu : float
            Kinematic Viscosity
        rho0 : float
            Reference Density
        turb_visc_model : str
            Turbulent Viscosity Model to be used: SMAG, SIGMA, SMAG_MCG
        """

        self.dim = int(dim)
        self.nu = nu
        self.rho0 = rho0
        self.lap_vel_prefactor = 2. * (2. + self.dim) * self.nu * self.rho0
        self.turb_visc_model = turb_visc_model

        turb_visc_models = ['SMAG', 'SIGMA', 'SMAG_MCG']
        if self.turb_visc_model not in turb_visc_models:
            raise Exception("Invalid Turbulent Viscosity Model.")

        super(MomentumEquation, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.
        d_av[d_idx] = 0.
        d_aw[d_idx] = 0.

    def loop(
        self, d_idx, s_idx, d_rho, d_tau, d_nu_t, d_p, d_au, d_av, d_aw, s_rho,
        s_V, s_p, s_tau, s_nu_t, VIJ, XIJ, R2IJ, EPS, DWIJ
    ):
        Vj = s_V[s_idx]
        rhoi = d_rho[d_idx]
        rhoj = s_rho[s_idx]
        Pi = d_p[d_idx]
        Pj = s_p[s_idx]

        # Pressure Gradient Term
        gradp_fac = (Pi + Pj) * Vj

        # Velocity Laplacian Term
        vijdotxij = VIJ[0] * XIJ[0] + VIJ[1] * XIJ[1] + VIJ[2] * XIJ[2]
        lap_vel_fac = self.lap_vel_prefactor * vijdotxij * Vj / (R2IJ + EPS)

        # Subfilter Stress Term
        if not self.turb_visc_model == "SMAG_MCG":
            i, j = declare('int', 3)
            n = 3
            tau_ij = declare('matrix((3,3))')
            for i in range(n):
                for j in range(n):
                    tau_ij[i * n + j] = d_tau[d_idx * 9 + i * n + j] + \
                        s_tau[s_idx * 9 + i * n + j]

            div_tau = declare('matrix((3,1))')

            if self.dim == 2:
                div_tau[0] = Vj * (tau_ij[0] * DWIJ[0] + tau_ij[1] * DWIJ[1])
                div_tau[1] = Vj * (tau_ij[3] * DWIJ[0] + tau_ij[4] * DWIJ[1])
                div_tau[2] = 0.
            elif self.dim == 3:
                for i in range(n):
                    for j in range(n):
                        div_tau[i] = Vj * (tau_ij[i * n + j] * DWIJ[j])

            tmp = gradp_fac + lap_vel_fac
            # Accelerations
            d_au[d_idx] += tmp * DWIJ[0] + div_tau[0]
            d_av[d_idx] += tmp * DWIJ[1] + div_tau[1]
            d_aw[d_idx] += tmp * DWIJ[2] + div_tau[2]

        else:
            nu_ti = d_nu_t[d_idx]
            nu_tj = s_nu_t[s_idx]
            div_tau_fac = 2 * (2 + self.dim) * rhoi * rhoj
            div_tau_fac *= (nu_ti + nu_tj) / (rhoi + rhoj)
            div_tau_fac *= vijdotxij * Vj / (R2IJ + EPS)

            tmp = gradp_fac + lap_vel_fac + div_tau_fac
            # Accelerations
            d_au[d_idx] += tmp * DWIJ[0]
            d_av[d_idx] += tmp * DWIJ[1]
            d_aw[d_idx] += tmp * DWIJ[2]
