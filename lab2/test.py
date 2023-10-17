import unittest
import PdfEstimation
import numpy as np


# Test the get_elem function
class MyTestCase(unittest.TestCase):
    def test_get_elem_left_boarder(self):
        samples = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.assertEqual(3, PdfEstimation.get_elem(samples, 0, 4))

    def test_get_elem_right_boarder(self):
        samples = np.array([-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0])
        self.assertEqual(-3, PdfEstimation.get_elem(samples, 10, 4))

    def test_get_elem_middle_cases(self):
        samples = np.array([-2.5, -1, 0, 1, 2, 3, 4, 5])
        self.assertEqual(2, PdfEstimation.get_elem(samples, 2, 4))


if __name__ == '__main__':
    unittest.main()
