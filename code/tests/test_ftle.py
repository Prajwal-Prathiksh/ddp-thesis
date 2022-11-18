# Standard library imports
try:
    # This is for Python-2.6.x
    import unittest2 as unittest
except ImportError:
    import unittest

# Library imports.
import numpy as np

# Local imports.
from ftle import (
    rename_fnames_according_to_time
)

class TestRenameFnamesAccordingToTime(unittest.TestCase):
    def test_rename_fnames_according_to_time(self):
        """
        Test that the filenames of the two time instances of a flow field are
        renamed according to the time of the flow field.
        """
        # Given
        data1 = {
            'solver_data': {
                't': 0.
            }
        }
        data2 = {
            'solver_data': {
                't': 1.
            }
        }

        # When
        data1, data2 = rename_fnames_according_to_time(data1, data2)

        # Then
        self.assertEqual(data1['solver_data']['t'], 0.0)
        self.assertEqual(data2['solver_data']['t'], 1.0)

        # Given
        data1 = {
            'solver_data': {
                't': 1.
            }
        }
        data2 = {
            'solver_data': {
                't': 0.
            }
        }

        # When
        data1, data2 = rename_fnames_according_to_time(data1, data2)

        # Then
        self.assertEqual(data1['solver_data']['t'], 0.0)
        self.assertEqual(data2['solver_data']['t'], 1.0)