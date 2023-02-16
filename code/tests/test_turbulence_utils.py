r"""
Test-cases for turbulence_utils.py
Author: K T Prajwal Prathiksh
###
"""
# Library imports
import unittest
import numpy as np
try:
    import sympy as sp
except ImportError:
    raise ImportError("SymPy is required for testing.")

# Local imports
from turbulence_utils import compute_curl

def compute_symbolic_curl(
    Fx:sp.symbols, Fy:sp.symbols, Fz:sp.symbols,
    x:sp.symbols, y:sp.symbols, z:sp.symbols
):
    """
    Compute the curl of a cartesian vector field using sympy.

    Parameters
    ----------
    Fx, Fy, Fz : sp.symbols
        The components of the vector field.
    x, y, z : sp.symbols
        The symbols for the cartesian coordinates.
    
    Returns
    -------
    curl_x, curl_y, curl_z : sp.symbols
        The components of the curl of the cartesian vector field.
    """
    curl_x = sp.diff(Fz, y) - sp.diff(Fy, z)
    curl_y = sp.diff(Fx, z) - sp.diff(Fz, x)
    curl_z = sp.diff(Fy, x) - sp.diff(Fx, y)
    return curl_x, curl_y, curl_z

class TestComputeCurl(unittest.TestCase):
    """
    Test the function compute_curl.
    """
    
    def __init__(self, *args, **kwargs):
        super(TestComputeCurl, self).__init__(*args, **kwargs)
        self.x, self.y, self.z = sp.symbols('x y z')
        self.u, self.v, self.w = sp.symbols('u v w')
        
        # Number of grid points
        self.n = 100
        self.dx = 1 / self.n
        self.dx_by_2 = self.dx / 2.
        self._x = np.linspace(self.dx_by_2, 1 - self.dx_by_2, self.n)

    def _get_test_cases(self, dim: int):
        """
        Return a list of test cases.

        Parameters
        ----------
        dim : int
            The dimensionality of the vector field.

        Returns
        -------
        test_cases : list
            A list of test cases.
        """
        pi, sin, cos = sp.pi, sp.sin, sp.cos
        x, y, z = self.x, self.y, self.z
        twopi = 2 * pi
        twopi_x, twopi_y, twopi_z = twopi * x, twopi * y, twopi * z

        # 2D cases
        test_cases_2d = dict(
            c2d_1=dict(
                u = -cos(twopi_x) * sin(twopi_y),
                v = sin(twopi_x) * cos(twopi_y),
            ),
            c2d_2=dict(
                u = y,
                v = -x,
            ),
            c2d_3=dict(
                u = x,
                v = x**2,
            )
        )

        # 3D cases
        test_cases_3d = dict(
            c3d_1=dict(
                u = 3*y*z,
                v = x*z,
                w = 2*x*y,
            ),
            c3d_2=dict(
                u = cos(twopi_x)*sin(twopi_y)*sin(twopi_z),
                v = sin(twopi_x)*cos(twopi_y)*sin(twopi_z),
                w = sin(twopi_x)*sin(twopi_y)*cos(twopi_z),
            ),
            c3d_3=dict(
                u = cos(x),
                v = cos(x),
                w = x,
            ),
        )

        if dim == 2:
            return self._append_sympy_lambda_functions(test_cases_2d, dim)
        elif dim == 3:
            return self._append_sympy_lambda_functions(test_cases_3d, dim)

    def _append_sympy_lambda_functions(self, test_cases, dim: int):
        """
        Append sympy lambda functions to the each test case.

        Parameters
        ----------
        test_cases : list
            A list of test cases.
        dim : int
            The dimensionality of the vector field.

        Returns
        -------
        test_cases : list
            A list of test cases with sympy lambda functions.
        """
        x, y, z = self.x, self.y, self.z
        if dim == 2:
            args = (x, y)
        elif dim == 3:
            args = (x, y, z)

        for case in test_cases:
            u = test_cases[case]['u']
            v = test_cases[case]['v']
            if dim == 3:
                w = test_cases[case]['w']
            else:
                w = 0
            
            test_cases[case]['u_func'] = sp.lambdify(args, u)
            test_cases[case]['v_func'] = sp.lambdify(args, v)
            if dim == 3:
                test_cases[case]['w_func'] = sp.lambdify(args, w)

            curl_x, curl_y, curl_z = compute_symbolic_curl(
                Fx=u, Fy=v, Fz=w, x=x, y=y, z=z
            )
            test_cases[case]['curl_x'] = curl_x
            test_cases[case]['curl_y'] = curl_y
            test_cases[case]['curl_z'] = curl_z

            if dim == 3:
                test_cases[case]['curl_x_func'] = sp.lambdify(args, curl_x)
                test_cases[case]['curl_y_func'] = sp.lambdify(args, curl_y)
            test_cases[case]['curl_z_func'] = sp.lambdify(args, curl_z)
                
        return test_cases

    def _get_error_msg(self, case, dim):
        """
        Return a formatted error message.

        Parameters
        ----------
        case : str
            The test case.
        dim : int
            The dimensionality of the vector field.
        
        Returns
        -------
        msg : str
            The formatted error message.
        """
        test_cases = self._get_test_cases(dim=dim)
        msg = '\n\n' + '*' * 40
        msg += f'\nERROR in --> Test case: {case}'
        msg += f'\n\nDimension: {dim}D'
        msg += f'\nTest case details:'
        for key in test_cases[case]:
            msg += f'\n\t{key}: {test_cases[case][key]}'
        msg += '\n\n' + '*' * 40
        return msg
    
    def test_should_work_for_2d(self, max_error=1e-1):
        """
        Test if the function works for 2D.
        """
        # Get test cases
        test_cases = self._get_test_cases(dim=2)

        # Create meshgrid
        x, y = np.meshgrid(self._x, self._x)

        for tc in test_cases:
            msg = self._get_error_msg(case=tc, dim=2)
            # Given
            u = test_cases[tc]['u_func'](x, y)
            v = test_cases[tc]['v_func'](x, y)

            # Test for edge_order=1
            # When
            curl_z = compute_curl(dx=self.dx, u=u, v=v, edge_order=1)
            curl_z_exact = test_cases[tc]['curl_z_func'](x, y)

            # Then
            error_z = np.abs(curl_z - curl_z_exact).max()
            self.assertTrue(error_z < max_error, msg=msg)

            # Test for edge_order=2
            # When
            curl_z = compute_curl(dx=self.dx, u=u, v=v, edge_order=2)
            curl_z_exact = test_cases[tc]['curl_z_func'](x, y)

            # Then
            error_z = np.abs(curl_z - curl_z_exact).max()
            self.assertTrue(error_z < max_error, msg=msg)
   
    def test_should_work_for_3d(self, max_error=1e-1):
        """
        Test if the function works for 3D.
        """
        # Get test cases
        test_cases = self._get_test_cases(dim=3)

        # Create meshgrid
        x, y, z = np.meshgrid(self._x, self._x, self._x)

        for tc in test_cases:
            msg = self._get_error_msg(case=tc, dim=3)
            # Given
            u = test_cases[tc]['u_func'](x, y, z)
            v = test_cases[tc]['v_func'](x, y, z)
            w = test_cases[tc]['w_func'](x, y, z)

            # When
            curl_x, curl_y, curl_z = compute_curl(
                dx=self.dx, u=u, v=v, w=w
            )

            curl_x_exact = test_cases[tc]['curl_x_func'](x, y, z)
            curl_y_exact = test_cases[tc]['curl_y_func'](x, y, z)
            curl_z_exact = test_cases[tc]['curl_z_func'](x, y, z)

            # Then
            error_x = np.abs(curl_x - curl_x_exact).max()
            error_y = np.abs(curl_y - curl_y_exact).max()
            error_z = np.abs(curl_z - curl_z_exact).max()

            self.assertTrue(error_x < max_error, msg=msg)
            self.assertTrue(error_y < max_error, msg=msg)
            self.assertTrue(error_z < max_error, msg=msg)

if __name__ == '__main__':
    unittest.main()