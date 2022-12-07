r"""
Test-cases for ftle.py
###
"""
# Library imports.
import unittest
import os

# Local imports.
from ftle import (
    rename_fnames_according_to_time
)


class TestRenameFnamesAccordingToTime(unittest.TestCase):
    def test_files_with_counter_of_same_length(self):
        """
        Test that the function works for files with counter of same length.
        """
        # Given
        data1 = os.path.join(
            'data', 'test_ftle_012.npz'
        )
        data2 = os.path.join(
            'data', 'test_ftle_010.npz'
        )


        # When
        data1_new, data2_new = rename_fnames_according_to_time(data1, data2)

        # Then
        self.assertEqual(data1_new, data2)
        self.assertEqual(data2_new, data1)

        # Given
        data1 = os.path.join(
            os.getcwd(), 'junk', 'junk_00000.npz'
        )
        data2 = os.path.join(
            os.getcwd(), 'junk', 'junk_00001.npz'
        )

        # When
        data1_new, data2_new = rename_fnames_according_to_time(data1, data2)

        # Then
        self.assertEqual(data1_new, data1)
        self.assertEqual(data2_new, data2)

    def test_files_with_counter_of_unequal_length(self):
        """
        Test that the function works for files with counter of unequal length.
        """
        # Given
        data1 = os.path.join(
            'data', 'test_ftle_12.npz'
        )
        data2 = os.path.join(
            'data', 'test_ftle_0100.npz'
        )

        # When
        data1_new, data2_new = rename_fnames_according_to_time(data1, data2)

        # Then
        self.assertEqual(data1_new, data1)
        self.assertEqual(data2_new, data2)

        # Given
        data1 = os.path.join(
            os.getcwd(), 'junk', 'junk_00500.npz'
        )
        data2 = os.path.join(
            os.getcwd(), 'junk', 'junk_0001.npz'
        )

        # When
        data1_new, data2_new = rename_fnames_according_to_time(data1, data2)

        # Then
        self.assertEqual(data1_new, data2)
        self.assertEqual(data2_new, data1)
    
    def test_files_with_invalid_format(self):
        """
        Test that the function throws an error for files with invalid format.
        """
        # Given
        data1 = os.path.join(
            'data', 'test_ftle_012.npz'
        )
        data2 = os.path.join(
            'data', 'file-0100.npz'
        )

        # When
        with self.assertRaises(ValueError):
            data1_new, data2_new = rename_fnames_according_to_time(data1, data2)

    
