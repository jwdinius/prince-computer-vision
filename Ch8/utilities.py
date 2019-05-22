import numpy as np
import scipy.special as sci_spec
import scipy.stats as stats
from scipy.optimize import minimize, fminbound
import sys
sys.path.append("..")
from importlib import reload
import common.utilities as common
reload(common)

import unittest


def linear_regression_mle(training_data, training_targets):
	# Algorithm 8.1
	I = training_data.shape[0]
	# TODO(jwd) - add exception handling
	training_targets = training_targets.reshape((I,))

	# append ones to the training_data
	_ones = np.ones((I, 1))
	_data = np.hstack((_ones, training_data))
	XXT = _data.T@_data
	phi = np.linalg.inv(XXT)@_data.T@training_targets
	dw = training_targets - _data@phi
	sigma_sq = dw.T@dw / I

	return phi, sigma_sq


class TestLinRegMLE(unittest.TestCase):
	def test_nominal(self):
		# just check to make sure the call is functioning for nominal input
		rng = np.random.RandomState(seed=0)
		X = np.linspace(-5, 5, 100).reshape((100, 1))
		# test parameters
		_sigma_sq = 0.2
		_phi = (5., 1.5)
		w = _phi[1] * X + _phi[0] * np.ones((100, 1)) + np.sqrt(_sigma_sq) * rng.randn(100,1)
		phi, sigma_sq = linear_regression_mle(X, w)
		self.assertAlmostEqual(_phi[0], phi[0], places=1)
		self.assertAlmostEqual(_phi[1], phi[1], places=1)
		self.assertAlmostEqual(_sigma_sq, sigma_sq, places=1)

if __name__ == '__main__':
	unittest.main()
