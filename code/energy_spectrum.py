r"""
Energy Spectrum of the Flow
############################
References
-----------
    .. [energyspectrum] Energy_Spectrum: Script (with Example) to Compute the
    Kinetic Energy Spectrum of Periodic Turbulent Flows. Accessed 7 Nov. 2022.
"""
# Library imports.
import numpy as np
from pysph.base.kernels import WendlandQuinticC4
from compyle.api import annotate, Elementwise, wrap
from pysph.tools.interpolator import Interpolator
from pysph.solver.utils import load

# TODO: Compyle iterative functions


def compute_energy_spectrum(
    u: np.ndarray, v: np.ndarray = None, w: np.ndarray = None, U0: float = 1.0,
    debug: bool = False
):
    """
    Calculate the point-wise energy spectrum of the flow E(kx, ky, kz), from
    the normalised velocity spectrum of a flow.
    Note: For the calculation of the velocity spectrum, the flow is assumed
    to be periodic, and the velocity field data is from an equidistant grid of
    points.
    Parameters
    ----------
    u : np.ndarray
        Velocity field in x-direction.
    v : np.ndarray, optional
        Velocity field in y-direction.
    w : np.ndarray, optional
        Velocity field in z-direction.
    U0 : float, optional
        Reference velocity. The default is 1.
    debug : bool, optional
        Return the velocity spectrum as well. The default is False.
    Returns
    -------
    EK_U : np.ndarray
        Point-wise energy spectrum of the flow in x-direction.
    EK_V : np.ndarray
        Point-wise energy spectrum of the flow in y-direction.
    EK_W : np.ndarray
        Point-wise energy spectrum of the flow in z-direction.
    """
    # Import FFT-functions
    from numpy.fft import fftn as fftn
    from numpy.fft import fftshift as fftshift

    # Check shape of velocity components for given dimensions
    dim = len(u.shape)
    if dim == 1:
        if v is not None or w is not None:
            raise ValueError(
                "Velocity components v and w should be None for 1D data."
            )
    elif dim == 2:
        if v is None:
            raise ValueError(
                "Velocity component v should not be None for 2D data."
            )
        if w is not None:
            raise ValueError(
                "Velocity component w should be None for 2D data."
            )
    elif dim == 3:
        if v is None or w is None:
            raise ValueError(
                "Velocity components v or w should not be None for 3D data."
            )

    # Velocity field data
    v = v if v is not None else np.array([0.])
    w = w if w is not None else np.array([0.])

    # Get normalised velocity spectrum
    u_spectrum = np.abs(fftn(u / U0) / u.size)
    v_spectrum = np.abs(fftn(v / U0) / v.size)
    w_spectrum = np.abs(fftn(w / U0) / w.size)

    EK_U = fftshift(u_spectrum**2)
    EK_V = fftshift(v_spectrum**2)
    EK_W = fftshift(w_spectrum**2)

    if debug:
        return EK_U, EK_V, EK_W, u_spectrum, v_spectrum, w_spectrum
    else:
        return EK_U, EK_V, EK_W

def compute_scalar_energy_spectrum(
    EK_U: np.ndarray, EK_V: np.ndarray = None, EK_W: np.ndarray = None,
    debug: bool = False
):
    """
    Calculate 1D energy spectrum of the flow E(k), from the point-wise energy
    spectrum E(kx, ky, kz), by integrating it over the surface of a sphere of
    radius k = (kx**2 + ky**2 + kz**2)**0.5.
    Parameters
    ----------
    EK_U : np.ndarray
        Point-wise energy spectrum of the flow in x-direction.
    EK_V : np.ndarray, optional
        Point-wise energy spectrum of the flow in y-direction.
    EK_W : np.ndarray, optional
        Point-wise energy spectrum of the flow in z-direction.
    debug : bool, optional
        Return the averaged energy spectrum as well. The default is False.
    Returns
    -------
    k : np.ndarray
        1D array of wave numbers.
    Ek : np.ndarray
        1D array of energy spectrum.
    """
    # Import numpy functions
    from numpy.linalg import norm as norm

    # Check shape of velocity components for given dimensions
    dim = len(np.shape(EK_U))
    if dim == 1:
        if EK_V is not None or EK_W is not None:
            raise ValueError(
                "Energy components EK_V and EK_W should be None for 1D data."
            )
        EK_U = np.array(EK_U)
    elif dim == 2:
        if EK_V is None:
            raise ValueError(
                "Energy component EK_V should not be None for 2D data."
            )
        if EK_W is not None:
            raise ValueError(
                "Energy component EK_W should be None for 2D data."
            )
        EK_U, EK_V = np.array(EK_U), np.array(EK_V)
    elif dim == 3:
        if EK_V is None or EK_W is None:
            raise ValueError(
                "Energy component EK_V or EK_W should not be None for 3D data."
            )
        EK_U, EK_V, EK_W = np.array(EK_U), np.array(EK_V), np.array(EK_W)

    eps = 1e-50

    box_side_x = np.shape(EK_U)[0]
    box_side_y = np.shape(EK_U)[1] if dim > 1 else 0
    box_side_z = np.shape(EK_U)[2] if dim > 2 else 0

    box_radius = int(
        1 + np.ceil(
            norm(np.array([box_side_x, box_side_y, box_side_z])) / 2
        )
    )

    center_x = int(box_side_x / 2)
    center_y = int(box_side_y / 2)
    center_z = int(box_side_z / 2)

    EK_U_sphere = np.zeros((box_radius, )) + eps
    EK_V_sphere = np.zeros((box_radius, )) + eps
    EK_W_sphere = np.zeros((box_radius, )) + eps

    if dim == 1:
        for i in range(box_side_x):
            wn = np.round(norm(i - center_x))
            wn = int(wn)

            EK_U_sphere[wn] += EK_U[i]

    elif dim == 2:
        for i in range(box_side_x):
            for j in range(box_side_y):
                wn = np.round(norm([i - center_x, j - center_y]))
                wn = int(wn)

                EK_U_sphere[wn] += EK_U[i, j]
                EK_V_sphere[wn] += EK_V[i, j]

    elif dim == 3:
        for i in range(box_side_x):
            for j in range(box_side_y):
                for k in range(box_side_z):
                    wn = np.round(
                        norm([i - center_x, j - center_y, k - center_z]))
                    wn = int(wn)

                    EK_U_sphere[wn] += EK_U[i, j, k]
                    EK_V_sphere[wn] += EK_V[i, j, k]
                    EK_W_sphere[wn] += EK_W[i, j, k]

    Ek = 0.5 * (EK_U_sphere + EK_V_sphere + EK_W_sphere)
    k = np.arange(0, len(Ek))

    if debug:
        return k, Ek, EK_U_sphere, EK_V_sphere, EK_W_sphere
    else:
        return k, Ek

def velocity_intepolator(
    fname: str, dim: int, kernel: object = None, nx_i: int = 101,
    domain_manager: object = None, **kwargs
):
    """
    Interpolate the energy spectrum of the flow from the given file.

    Parameters
    ----------
    fname : str
        Name of the file containing the flow data.
    dim : int
        Dimension of the flow.
    kernel : object, optional
        Kernel object. The default is WendlandQuinticC4.
    nx_i : int, optional
        Number of points to interpolate the energy spectrum (nx_i**2 for 2D data,
        nx_i**3 for 3D data). The default is 101.
    domain_manager : object, optional
        DomainManager object. The default is None.
    **kwargs : dict, optional
        Additional keyword arguments to pass to the interpolator.

    Returns
    -------
    t : float
        Time of the flow data.
    res : list[np.ndarray]
        List of interpolated energy spectrum of the flow for each direction.
    """
    # Load data
    data = load(fname)
    t = data["solver_data"]["t"]

    # Create meshgrid based on dimension
    _x = np.linspace(0, 1, nx_i)
    if dim == 1:
        x = _x
        y = z = None
    elif dim == 2:
        x, y = np.meshgrid(_x, _x)
        z = None
    elif dim == 3:
        x, y, z = np.meshgrid(_x, _x, _x)
    else:
        raise ValueError("Dimension should be 1, 2 or 3.")

    # Setup default interpolator properties
    if kernel is None:
        kernel = WendlandQuinticC4(dim=dim)

    # Interpolate velocity
    interp = Interpolator(
        list(data['arrays'].values()), x=x, y=y, z=z,
        kernel=kernel, domain_manager=domain_manager, **kwargs
    )
    if dim == 1:
        _u = interp.interpolate('u')
        ui = _u
        res = [ui, None, None]
    elif dim == 2:
        _u = interp.interpolate('u')
        _v = interp.interpolate('v')
        ui = _u.reshape(nx_i, nx_i)
        vi = _v.reshape(nx_i, nx_i)
        res = [ui, vi, None]
    elif dim == 3:
        _u = interp.interpolate('u')
        _v = interp.interpolate('v')
        _w = interp.interpolate('w')
        ui = _u.reshape(nx_i, nx_i, nx_i)
        vi = _v.reshape(nx_i, nx_i, nx_i)
        wi = _w.reshape(nx_i, nx_i, nx_i)
        res = [ui, vi, wi]
    return t, res


class EnergySpectrum(object):
    """
    Class to compute the energy spectrum of the flow.

    Initialization
    --------------
    1. From velocity field
        >>> EnergySpectrum(dim, u, v, w....)
    2. From PySPH file (Interpolates the velocity field implicitly)
        >>> EnergySpectrum.from_pysph_file(fname....)
    3. From an example
        >>> EnergySpectrum.from_example(dim....)

    Parameters
    ----------
    dim : int
        Dimension of the flow.
    u : np.ndarray
        Velocity field in x-direction.
    v : np.ndarray
        Velocity field in y-direction. None for 1D data.
    w : np.ndarray
        Velocity field in z-direction. None for 1D and 2D data.
    t : float, optional
        Time of the flow data. Default is 0. Optional, required for plotting.
    U0: float, optional
        Reference velocity. Default is 1.
    """
    def __init__(
        self, dim:int, u:np.ndarray, v:np.ndarray=None, w:np.ndarray=None,
        t:float=0., U0:float=1.
    ):
        """
        Initialize the class.
        """
        self.dim = dim
        self.u, self.v, self.w = u, v, w
        self.t = t
        self.U0 = U0

        self.n_1d = u.shape[0]

        if dim not in [1, 2, 3]:
            raise ValueError("Dimension should be 1, 2 or 3.")

        self._check_format_of_list_data([u,v,w])

        self.k, self.Ek = None, None

    # Class methods
    @classmethod
    def from_pysph_file(
        cls, fname:str, dim:int, L:float, nx_i:int, kernel:object,
        domain_manager:object=None, U0=1., **kwargs
    ):
        """
        Create an EnergySpectrum object from a PySPH output file.

        Parameters
        ----------
        fname : str
            Name of the file containing the flow data.
        dim : int
            Dimension of the flow.
        L : float
            Length of the domain.
        nx_i : int
            Number of points to interpolate the energy spectrum (nx_i**2 for 2D
            data, nx_i**3 for 3D data).
        kernel : object
            Kernel object.
        domain_manager : object, optional
            DomainManager object. Default is None.
        U0 : float, optional
            Reference velocity. Default is 1.
        **kwargs : dict, optional
            Additional keyword arguments to pass to the PySPH interpolator.

        Returns
        -------
        EnergySpectrum object.
        """
        try:
            from pysph.tools.geometry import load
            from pysph.tools.interpolator import Interpolator
            from pysph.base.kernels import WendlandQuinticC4
        except ImportError:
            raise ImportError(
                "PySPH is not installed."
            )
        
        data = load(fname)
        t = data["solver_data"]["t"]
        dim = data["solver_data"]["dim"]

        # Create meshgrid based on dimension
        _x = np.linspace(0, L, nx_i)
        if dim == 1:
            x = _x
            y = z = None
        elif dim == 2:
            x, y = np.meshgrid(_x, _x)
            z = None
        elif dim == 3:
            x, y, z = np.meshgrid(_x, _x, _x)
        
        # Setup default interpolator properties
        if kernel is None:
            kernel = WendlandQuinticC4(dim=dim)
        
        # Interpolate velocity
        interp = Interpolator(
            list(data['arrays'].values()), x=x, y=y, z=z,
            kernel=kernel, domain_manager=domain_manager, **kwargs
        )
        if dim == 1:
            _u = interp.interpolate('u')
            ui = _u
            vi = wi = None
        elif dim == 2:
            _u = interp.interpolate('u')
            _v = interp.interpolate('v')
            ui = _u.reshape(nx_i, nx_i)
            vi = _v.reshape(nx_i, nx_i)
            wi = None
        elif dim == 3:
            _u = interp.interpolate('u')
            _v = interp.interpolate('v')
            _w = interp.interpolate('w')
            ui = _u.reshape(nx_i, nx_i, nx_i)
            vi = _v.reshape(nx_i, nx_i, nx_i)
            wi = _w.reshape(nx_i, nx_i, nx_i)

        return cls(
            dim=dim, u=ui, v=vi, w=wi, t=t, U0=U0
        )

    @classmethod
    def from_example(
        cls, dim:int, nx:int, custom_formula:list=None
    ):
        """
        Create an EnergySpectrum object from an example.

        Parameters
        ----------
        dim : int
            Dimension of the flow.
        nx : int
            Number of points in each direction.
        custom_formula : list, optional
            Custom formula to generate the velocity field for each dimension.
            Default is None.
            Numpy functions can be used. Spatial coordinates are x, y and z.
            pi, twopi, cos and sin can be used as well.        

        Returns
        -------
        EnergySpectrum object.

        Notes
        -----
        dim = 1:
            x = arange(0, 1, 1/nx)
            u = - cos(2πx)
        dim = 2:
            _x = arange(0, 1, 1/nx)
            x, y = meshgrid(_x, _x)
            u = + cos(2πx) * sin(2πy)
            v = - sin(2πx) * cos(2πy)
        dim = 3:
            _x = arange(0, 1, 1/nx)
            x, y, z = meshgrid(_x, _x, _x)
            u = + sin(2πx) * cos(2πy) * cos(2πz)
            v = - cos(2πx) * sin(2πy) * cos(2πz)
            w = 0.0
        """
        pi = np.pi
        twopi = 2*pi
        cos, sin = np.cos, np.sin

        _x = np.arange(0., 1., 1./nx)
        if dim == 1:
            if custom_formula is None:
                x = _x
                u = - cos(twopi*x)
            else:
                u = eval(custom_formula[0])
            v = w = None
        elif dim == 2:
            x, y = np.meshgrid(_x, _x)
            if custom_formula is None:
                u = + cos(twopi*x) * sin(twopi*y)
                v = - sin(twopi*x) * cos(twopi*y)
            else:
                u = eval(custom_formula[0])
                v = eval(custom_formula[1])
            w = None
        elif dim == 3:
            x, y, z = np.meshgrid(_x, _x, _x)
            if custom_formula is None:
                u = + sin(twopi*x) * cos(twopi*y) * cos(twopi*z)
                v = - cos(twopi*x) * sin(twopi*y) * cos(twopi*z)
                w = np.zeros_like(u)
            else:
                u = eval(custom_formula[0])
                v = eval(custom_formula[1])
                w = eval(custom_formula[2])
        else:
            raise ValueError("Dimension should be 1, 2 or 3.")
        
        return cls(
            dim=dim, u=u, v=v, w=w, t=0.0, U0=1.0
        )


    # Private methods
    def _check_format_of_list_data(self, data:list):
        """
        Check the format of the list data.
        For 1D data, the list should be of the form [u, None, None].
        For 2D data, the list should be of the form [u, v, None].
        For 3D data, the list should be of the form [u, v, w].

        Parameters
        ----------
        data : list
            List of data to check.
        """
        if len(data) != 3:
            raise ValueError("The data should be a list of 3 arrays.")

        if self.dim == 1:
            if data[1] is not None or data[2] is not None:
                raise ValueError(
                    "The data should be a list of 1 array for dim = 1."
                )
            if data[0] is None:
                raise ValueError(f"{data[0]} is None.")
        elif self.dim == 2:
            if data[2] is not None:
                raise ValueError(
                    "The data should be a list of 2 arrays for dim = 2."
                )
            if data[0] is None or data[1] is None:
                raise ValueError(f"{data[0]} or {data[1]} is None.")
        elif self.dim == 3:
            if data[0] is None or data[1] is None or data[2] is None:
                raise ValueError(
                    f"{data[0]} or {data[1]} or {data[2]} is None."
            )
    
    def _correct_format_of_list_data(self, data:list):
        """
        Correct the format of the list data.

        Parameters
        ----------
        data : list
            List of data to correct.
        
        Returns
        -------
        Corrected list of data.
        """
        if self.dim == 1:
            corrected_data = [data[0], None, None]
        elif self.dim == 2:
            corrected_data = [data[0], data[1], None]
        elif self.dim == 3:
            corrected_data = data
        
        self._check_format_of_list_data(corrected_data)
        return corrected_data

    def _compute_energy_spectrum(self):
        """
        Compute the energy spectrum of the flow.
        """
        EK_U, EK_V, EK_W, u_spectrum, v_spectrum, w_spectrum =\
            compute_energy_spectrum(
                u=self.u,
                v=self.v,
                w=self.w,
                U0=self.U0,
                debug=True
            )

        return EK_U, EK_V, EK_W, u_spectrum, v_spectrum, w_spectrum

    def _compute_scalar_energy_spectrum(self):
        """
        Compute the energy spectrum of the flow.
        """
        k, Ek, EK_U_sphere, EK_V_sphere, EK_W_sphere =\
            compute_scalar_energy_spectrum(
                EK_U=self.EK_U,
                EK_V=self.EK_V,
                EK_W=self.EK_W,
                debug=True
            )
        return k, Ek, EK_U_sphere, EK_V_sphere, EK_W_sphere

    # Public methods
    def compute(self):
        """
        Compute the energy spectrum of the flow.
        """
        # Compute energy spectrum
        res = self._compute_energy_spectrum()
        self.EK_U, self.EK_V, self.EK_W, self.u_spectrum, self.v_spectrum,\
            self.w_spectrum = res

        self.EK_U, self.EK_V, self.EK_W = self._correct_format_of_list_data(
            [self.EK_U, self.EK_V, self.EK_W]
        )
        self.u_spectrum, self.v_spectrum, self.w_spectrum =\
            self._correct_format_of_list_data(
                [self.u_spectrum, self.v_spectrum, self.w_spectrum]
            )
        
        # Compute scalar energy spectrum
        res = self._compute_scalar_energy_spectrum()
        self.k, self.Ek, self.EK_U_sphere, self.EK_V_sphere, self.EK_W_sphere \
            = res
        self.EK_U_sphere, self.EK_V_sphere, self.EK_W_sphere =\
            self._correct_format_of_list_data(
                [self.EK_U_sphere, self.EK_V_sphere, self.EK_W_sphere]
            )

    def plot_scalar_Ek(
        self, show=False, savefig=False, fname=None, dpi=300, plot_type="log",
    ):
        """
        Plot the scalar energy spectrum of the flow.

        Parameters
        ----------
        show : bool, optional
            Show the plot. Default is False.
        savefig : bool, optional
            Save the figure. Default is False.
        fname : str, optional
            Filename to save the figure. Default is "./energy_spectrum.png".
        dpi : int, optional
            Dots per inch. Default is 300.
        plot_type : str, optional
            Type of plot. Default is "log". Options: "log", "stem".
        """
        if self.k is None:
            self.compute()
        
        if fname is None:
            fname = "./energy_spectrum.png"
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib is not installed.")

        n = self.n_1d
        plt.clf()
        if plot_type == "log":
            plt.loglog(self.k[0:n], self.Ek[0:n], 'k')
            plt.loglog(self.k[n:], self.Ek[n:], 'k--')
        elif plot_type == "stem":
            plt.stem(self.Ek)
        plt.xlabel(r'$k$')
        plt.ylabel(r'$E(k)$')
        plt.grid()
        plt.tight_layout()
        plt.title(f"Energy spectrum at t = {self.t:.2f}")

        if savefig:
            plt.savefig(fname, dpi=dpi)
        if show:
            plt.show()

    def plot_EK(
        self, show=False, savefig=False, fname=None, dpi=300, shift_fft=False
    ):
        """
        Plot each component of the energy spectrum of the flow.

        Parameters
        ----------
        show : bool, optional
            Show the plot. Default is False.
        savefig : bool, optional
            Save the figure. Default is False.
        fname : str, optional
            Filename to save the figure.
            Default is "./EK_{dim}_{component}.png".
        dpi : int, optional
            Dots per inch. Default is 300.
        shift_fft : bool, optional
            Shift the FFT. Default is False.
        """

        if self.k is None:
            self.compute()
        
        dim = self.dim        
        if fname is None:
            fname = f"./EK_{dim}_"
        else:
            fname = f"{fname}_"

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib is not installed.")

        if shift_fft:
            fftshift = np.fft.fftshift
        else:
            fftshift = lambda x: x

        if dim == 1:
            plt.clf()
            plt.stem(fftshift(self.EK_U))
            plt.xlabel(r'$k$')
            plt.ylabel(r'$E_{u}(k)$')
            plt.grid()
            plt.tight_layout()
            plt.title(r"$E_u(k)$ at t = {:.2f}".format(self.t))
            if savefig:
                plt.savefig(fname + "U.png", dpi=dpi)
            if show:
                plt.show()
            
        elif dim == 2:
            plt.clf()

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(fftshift(self.EK_U))
            axes[0].set_title(r"$E_u(k)$ at t = {:.2f}".format(self.t))
            axes[0].set_xlabel(r"$k_x$")
            axes[0].set_ylabel(r"$k_y$")
            axes[0].invert_yaxis()

            axes[1].imshow(fftshift(self.EK_V))
            axes[1].set_title(r"$E_v(k)$ at t = {:.2f}".format(self.t))
            axes[1].set_xlabel(r"$k_x$")
            axes[1].set_ylabel(r"$k_y$")
            axes[1].invert_yaxis()

            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(axes[0].images[0], cax=cbar_ax)

            if savefig:
                plt.savefig(fname + "UV.png", dpi=dpi)
            if show:
                plt.show()

        elif dim == 3:
            raise NotImplementedError("3D not implemented yet.")





        