r"""
Test-cases for energy_spectrum.py
Author: K T Prajwal Prathiksh
###
"""
# Library imports
import unittest
import numpy as np

# Local imports.
from energy_spectrum import (
    compute_energy_spectrum, compute_scalar_energy_spectrum_python,
    compute_scalar_energy_spectrum_numba,
    compute_scalar_energy_spectrum_compyle
)


def check_if_coords_are_in_indices(coords, indices):
    """
    Check if given coordinates are in indices.

    Parameters
    ----------
    coords : array_like
        Coordinates to check. Format: [[x1, y1, z1], [x2, y2, z2], ...].
    indices : array_like
        Indices to check. Format: [[x1, ...], [y1, ...], [z1, ...]].

    Returns
    -------
    bool
        True if the coordinates are in the indices, False otherwise.
    """
    ind = np.vstack(indices).T
    for coord in coords:
        if coord not in ind:
            return False
    return True


class TestComputeEnergySpectrum(unittest.TestCase):
    """
    Test the function calculate_energy_spectrum.
    """

    def test_should_work_for_1d_data(self):
        """
        Test that the function works for 1D data.
        """

        # Given
        sr = 30
        x = np.arange(0, 1., 1. / sr)
        u = np.sin(2 * np.pi * x)
        U0 = 1.

        # When
        ek_u, ek_v, ek_w, u_spectrum, v_spectrum, w_spectrum =\
            compute_energy_spectrum(
                u, v=None, w=None, U0=U0, debug=True
            )

        # Then
        # Shape check
        self.assertEqual(ek_u.shape, u.shape)
        self.assertEqual(ek_v.shape[0], 1)
        self.assertEqual(ek_w.shape[0], 1)
        self.assertEqual(u_spectrum.shape, u.shape)
        self.assertEqual(v_spectrum.shape[0], 1)
        self.assertEqual(w_spectrum.shape[0], 1)

        # Velocity spectrum check
        tol = 1e-10
        self.assertTrue(np.allclose(u_spectrum[0], 0., atol=tol))
        self.assertTrue(np.allclose(u_spectrum[1], 0.5, atol=tol))
        self.assertTrue(np.allclose(u_spectrum[2:-1], 0., atol=tol))
        self.assertTrue(np.allclose(u_spectrum[-1], 0.5, atol=tol))
        self.assertTrue(np.allclose(v_spectrum, 0., atol=tol))
        self.assertTrue(np.allclose(w_spectrum, 0., atol=tol))

        # Energy spectrum check
        ek_u_shifted = np.fft.fftshift(ek_u)
        self.assertTrue(np.allclose(ek_u_shifted[0], 0., atol=tol))
        self.assertTrue(np.allclose(ek_u_shifted[1], 0.125, atol=tol))
        self.assertTrue(np.allclose(ek_u_shifted[2:-1], 0., atol=tol))
        self.assertTrue(np.allclose(ek_u_shifted[-1], 0.125, atol=tol))
        self.assertTrue(np.allclose(ek_v, 0., atol=tol))
        self.assertTrue(np.allclose(ek_w, 0., atol=tol))

        # Given
        sr = 30
        x = np.arange(0, 1., 1. / sr)
        u = np.sin(2 * np.pi * x) + np.sin(2 * np.pi * 2 * x)
        U0 = 0.5

        # When
        ek_u, ek_v, ek_w, u_spectrum, v_spectrum, w_spectrum =\
            compute_energy_spectrum(
                u, v=None, w=None, U0=U0, debug=True
            )

        # Then
        # Shape check
        self.assertEqual(ek_u.shape, u.shape)

        # Velocity spectrum check
        tol = 1e-10
        self.assertTrue(np.allclose(u_spectrum[0], 0., atol=tol))
        self.assertTrue(np.allclose(u_spectrum[1:2], 1., atol=tol))
        self.assertTrue(np.allclose(u_spectrum[3:-2], 0., atol=tol))
        self.assertTrue(np.allclose(u_spectrum[-2:], 1., atol=tol))

        # Energy spectrum check
        ek_u_shifted = np.fft.fftshift(ek_u)
        self.assertTrue(np.allclose(ek_u_shifted[0], 0., atol=tol))
        self.assertTrue(np.allclose(ek_u_shifted[1:2], 0.5, atol=tol))
        self.assertTrue(np.allclose(ek_u_shifted[3:-2], 0., atol=tol))
        self.assertTrue(np.allclose(ek_u_shifted[-2:], 0.5, atol=tol))

    def test_should_work_for_2d_data(self):
        """
        Test that the function works for 2D data.
        """

        # Given
        sr = 30
        x = y = np.arange(0, 1., 1. / sr)
        X, Y = np.meshgrid(x, y)
        u = np.sin(2 * np.pi * X) + np.sin(2 * np.pi * 2 * Y)
        U0 = 1.

        # When
        ek_u, ek_v, ek_w, u_spectrum, v_spectrum, w_spectrum =\
            compute_energy_spectrum(
                u=u, v=u, w=None, U0=U0, debug=True
            )

        # Then
        # Shape check
        self.assertEqual(ek_u.shape, u.shape)
        self.assertEqual(ek_v.shape, u.shape)
        self.assertEqual(ek_w.shape[0], 1)
        self.assertEqual(u_spectrum.shape, u.shape)
        self.assertEqual(v_spectrum.shape, u.shape)
        self.assertEqual(w_spectrum.shape[0], 1)

        # Velocity spectrum check
        tol = 1e-10
        ind = np.where(np.abs(u_spectrum - 0.5) < tol)
        self.assertEqual(np.shape(ind), (2, 4))
        coords = [
            [0, 1], [0, sr - 1],  # Coefficients for x-axis
            [2, 0], [sr - 2, 0]  # Coefficients for y-axis
        ]
        self.assertTrue(
            check_if_coords_are_in_indices(coords=coords, indices=ind), True
        )

        # Energy spectrum check
        ek_u_shifted = np.fft.fftshift(ek_u)
        ind = np.where(np.abs(ek_u_shifted - 0.125) < tol)
        self.assertEqual(np.shape(ind), (2, 4))
        self.assertTrue(
            check_if_coords_are_in_indices(coords=coords, indices=ind), True
        )

        # Check that the spectrum is the same for u and v
        self.assertTrue(np.allclose(u_spectrum, v_spectrum, atol=tol))
        self.assertTrue(np.allclose(ek_u, ek_v, atol=tol))

    def test_should_work_for_3d_data(self):
        """
        Test that the function works for 3D data.
        """
        # Given
        sr = 30
        x = y = z = np.arange(0, 1., 1. / sr)
        X, Y, Z = np.meshgrid(x, y, z)
        u = np.sin(2 * np.pi * X) + np.sin(2 * np.pi * 2 * Y) +\
            np.sin(2 * np.pi * 3 * Z)
        U0 = 0.5

        # When
        ek_u, ek_v, ek_w, u_spectrum, v_spectrum, w_spectrum =\
            compute_energy_spectrum(
                u=u, v=u, w=u, U0=U0, debug=True
            )

        # Then
        # Shape check
        self.assertEqual(ek_u.shape, u.shape)
        self.assertEqual(ek_v.shape, u.shape)
        self.assertEqual(ek_w.shape, u.shape)
        self.assertEqual(u_spectrum.shape, u.shape)
        self.assertEqual(v_spectrum.shape, u.shape)
        self.assertEqual(w_spectrum.shape, u.shape)

        # Velocity spectrum check
        tol = 1e-10
        ind = np.where(np.abs(u_spectrum - 1.) < tol)
        self.assertEqual(np.shape(ind), (3, 6))
        coords = [
            [0, 0, 3], [0, 0, 27],  # Coefficients for x-axis
            [0, 1, 0], [0, 29, 0],  # Coefficients for y-axis
            [2, 0, 0], [28, 0, 0]  # Coefficients for z-axis
        ]
        self.assertTrue(
            check_if_coords_are_in_indices(coords=coords, indices=ind), True
        )

        # Energy spectrum check
        ek_u_shifted = np.fft.fftshift(ek_u)
        ind = np.where(np.abs(ek_u_shifted - 0.5) < tol)
        self.assertEqual(np.shape(ind), (3, 6))
        self.assertTrue(
            check_if_coords_are_in_indices(coords=coords, indices=ind), True
        )


class TestComputeScalarEnergySpectrum(unittest.TestCase):
    """
    Test the function calculate_scalar_energy_spectrum.
    """

    def _test_should_work_for_1d_data(self, func, ord, msg):
        """
        Test that the function works for 1D data.

        Parameters
        ----------
        func : function
            Function to test.
        ord : int
            Order of the norm.
        msg : str
            Message to display in case of failure.
        """
        # Given
        N = 20
        ek_u = [1] * N
        ek_u = np.array(ek_u[::-1] + [0.] + ek_u, dtype=np.float64)

        # When
        k, ek = func(
            ek_u=ek_u, debug=False, ord=ord
        )

        # Then
        # Shape check
        self.assertEqual(k.shape, ek.shape, msg=msg)
        self.assertEqual(k.shape, (N + 2,), msg=msg)

        # Check the energy spectrum
        tol = 1e-10
        self.assertTrue(np.allclose(ek[0], 0, atol=tol), msg=msg)
        self.assertTrue(np.allclose(ek[1:-1], 1, atol=tol), msg=msg)
        self.assertTrue(np.allclose(ek[-1], 0., atol=tol), msg=msg)

        # Check the wavenumber = arange(N + 2)
        self.assertTrue(np.allclose(k, np.arange(N + 2), atol=tol), msg=msg)

        # Given
        N = 20
        ek_u = list(np.arange(N) + 1)
        ek_u = np.array(ek_u[::-1] + [0.] + ek_u, dtype=np.float64)

        # When
        k, ek = func(
            ek_u=ek_u, debug=False, ord=ord
        )

        # Then
        # Shape check
        self.assertEqual(k.shape, ek.shape, msg=msg)
        self.assertEqual(k.shape, (N + 2,), msg=msg)

        # Energy spectrum check
        tol = 1e-10
        # Sum of ek and 0.5*(ek_u) must be same
        self.assertTrue(
            np.allclose(np.sum(ek), 0.5 * np.sum(ek_u), atol=tol), msg=msg
        )
        # Check the energy spectrum
        self.assertTrue(
            np.allclose(ek[:-1], np.arange(N + 1), atol=tol), msg=msg
        )
        self.assertTrue(np.allclose(ek[-1], 0., atol=tol), msg=msg)

        # Check the wavenumber = arange(N + 2)
        self.assertTrue(np.allclose(k, np.arange(N + 2), atol=tol), msg=msg)

    def _test_should_work_for_2d_data(self, func, ord, msg):
        """
        Test that the function works for 2D data.

        Parameters
        ----------
        func : function
            Function to test.
        ord : int
            Order of the norm.
        msg : str
            Message to display in case of failure.
        """
        # Given
        ek_u = np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ], dtype=np.float64)

        # When
        k, ek = func(
            ek_u=ek_u, ek_v=ek_u, debug=False, ord=ord
        )

        # Then
        # Shape check
        self.assertEqual(k.shape, ek.shape, msg=msg)
        self.assertEqual(k.shape, (4,), msg=msg)

        # Energy spectrum check
        tol = 1e-10
        # Sum of ek and 0.5*(ek_u + ek_v) must be same
        self.assertTrue(
            np.allclose(np.sum(ek), 0.5 * (np.sum(ek_u + ek_u)), atol=tol),
            msg=msg
        )
        # Check the energy spectrum
        self.assertTrue(np.allclose(ek, [1, 4, 0, 0], atol=tol), msg=msg)

        # Given
        ek_u = np.array([
            [0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 1., 0., 1., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0.]
        ], dtype=np.float64)

        # When
        k, ek = func(
            ek_u=ek_u, ek_v=ek_u, debug=False, ord=ord
        )

        # Then
        # Shape check
        self.assertEqual(k.shape, ek.shape, msg=msg)
        self.assertEqual(k.shape, (5,), msg=msg)

        # Energy spectrum check
        tol = 1e-10
        # Sum of ek and 0.5*(ek_u + ek_v) must be same
        self.assertTrue(
            np.allclose(np.sum(ek), 0.5 * np.sum(ek_u + ek_u), atol=tol),
            msg=msg
        )
        # Check the energy spectrum
        self.assertTrue(np.allclose(ek, [0, 2, 2, 0, 0], atol=tol), msg=msg)

    def _test_should_work_for_3d_data(self, func, ord, msg):
        """
        Test that the function works for 3D data.

        Parameters
        ----------
        func : function
            Function to test.
        ord : int
            Order of the norm.
        msg : str
            Message to display in case of failure.
        """
        # Given
        ek_u = np.array([
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]
            ],
            [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]
            ],
            [
                [1, 0, 1],
                [0, 1, 0],
                [1, 0, 1]
            ]
        ], dtype=np.float64)

        # When
        k, ek = func(
            ek_u=ek_u, ek_v=ek_u, ek_w=ek_u, debug=False, ord=ord
        )

        # Then
        # Shape check
        self.assertEqual(k.shape, ek.shape, msg=msg)
        self.assertEqual(k.shape, (4,), msg=msg)

        # Energy spectrum check
        tol = 1e-10
        # Sum of ek and 0.5*(ek_u + ek_v + ek_w) must be same
        self.assertTrue(
            np.allclose(
                np.sum(ek), 0.5 * np.sum(ek_u + ek_u + ek_u), atol=tol
            ),
            msg=msg
        )
        # Check the energy spectrum
        if ord == np.inf:
            ex_exact = [1, 18, 0, 0]
        if ord == 2:
            ex_exact = [1, 14, 4, 0]
        self.assertTrue(np.allclose(ek * 2. / 3., ex_exact, atol=tol), msg=msg)

    def _get_funcs(self):
        """
        Get the functions to test.

        Returns
        -------
        funcs : list
            List of functions to test.
        """
        FUNCS = [
            compute_scalar_energy_spectrum_python,
            compute_scalar_energy_spectrum_numba,
            compute_scalar_energy_spectrum_compyle
        ]
        return FUNCS

    def _get_orders(self):
        """
        Get the orders to test.

        Returns
        -------
        orders : list
            List of orders to test.
        """
        ORDERS = [np.inf, 2]
        return ORDERS

    def _get_error_msg(self, func, dim, ord):
        """
        Get the error message to display in case of failure.

        Parameters
        ----------
        func : function
            Function to test.
        dim : int
            Dimension of the data.
        ord : int
            Order of the norm.
        """
        msg = '\n\n' + '*' * 40
        msg += "\n\tERROR in --> Function: {0}".format(func.__name__)
        msg += "\n\tDimension: {0}".format(dim)
        msg += "\n\tOrder: {0}".format(ord)
        msg += "\n" + '*' * 40
        return msg

    def test_should_work_for_all_dimensions(self):
        """
        Test that the function works for data of all dimensions.
        """
        # Get every combination of functions and orders
        for func in self._get_funcs():
            for ord in self._get_orders():
                msg = self._get_error_msg(func=func, dim=1, ord=ord)
                self._test_should_work_for_1d_data(func=func, ord=ord, msg=msg)

                msg = self._get_error_msg(func=func, dim=2, ord=ord)
                self._test_should_work_for_2d_data(func=func, ord=ord, msg=msg)

                msg = self._get_error_msg(func=func, dim=3, ord=ord)
                self._test_should_work_for_3d_data(func=func, ord=ord, msg=msg)


if __name__ == '__main__':
    unittest.main()
