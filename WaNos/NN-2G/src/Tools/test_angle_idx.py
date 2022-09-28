"""
##################################################

Module: Test Angle Index

Brief:
Test whether the angle index function is functioning

##################################################
"""


import unittest

from Calculator.CompileVec import get_angles, get_rads
from Calculator.Calculation import get_neighbour

class TestAngle(unittest.TestCase):

    def setUp(self):

        self.n_atoms = 10
        self.n_angs  = int(self.n_atoms * (self.n_atoms -1) * (self.n_atoms - 2) / 2 )
        self.n_rads  = int(self.n_atoms * (self.n_atoms-1) / 2)


    def test_angle_arr(self):
        """
        angle_arr should be consistent with the neighbour list
        """

        # First, Must be able to get the angles_arr
        angle_arr = get_angles(self.n_atoms, self.n_angs)

        print(angle_arr)
        print(len(angle_arr))

        # Get Neighbour List
        distance_arr = []
        neighbourlist_arr, neighbourpair_arr, neighbourlist_count, neighbourpair_count = get_neighbour(distance_arr, self.n_atoms)
        print(neighbourpair_arr)
        print(len(neighbourpair_arr[0]))

        # TODO: Compare all the elements in the angle_arr and the neighbour list arr
        # Currently they have the same length, and therefore should be equivalent

        # Second, test the rad_arr
        rad_arr = get_rads(self.n_atoms, self.n_rads)
        print(rad_arr)
        print(len(rad_arr))
        print(len(neighbourlist_arr[0]))
        print(neighbourlist_arr[1])


if __name__ == '__main__':
    unittest.main()
