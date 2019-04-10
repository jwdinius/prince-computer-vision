import numpy as np
import unittest

def delta_function(x):
	try:
		if isinstance(x, (int, float)) or np.isscalar(x):
			if np.isclose(x, 0):
				return 1.
			else:
				return 0.
		if all([np.isclose(_x, 0) for _x in x]):
			return 1.
		else:
			return 0.
	except TypeError:
		raise TypeError('The input must be a single number or an iterable container of numbers.')

class TestCommonUtilities(unittest.TestCase):
	def test_delta_function(self):
		# TEST 1: TEST BAD DATA
		x = None
		self.assertRaises(TypeError, delta_function, x)
		x = [0, None]
		self.assertRaises(TypeError, delta_function, x)
		# TEST 2: SINGLE-VARIATE
		x = 0
		self.assertTrue(np.isclose(1, delta_function(x)))
		x = 1
		self.assertTrue(np.isclose(0, delta_function(x)))
		# TEST 3: MULTI-VARIATE
		x = [0, 0, 0, 0]
		self.assertTrue(np.isclose(1, delta_function(x)))
		x = [1, 0, 0]
		self.assertTrue(np.isclose(0, delta_function(x)))


if __name__ == '__main__':
	unittest.main()
