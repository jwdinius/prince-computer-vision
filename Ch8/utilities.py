import numpy as np
import scipy.special as sci_spec
import scipy.stats as stats
from scipy.optimize import minimize
import sys
sys.path.append("..")
from importlib import reload
import common.utilities as common
reload(common)

import unittest
import warnings

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


def linear_regression_bayes(training_data, training_targets, sigma_sq_p):
	# Algorithm 8.2
	I, D = training_data.shape
	# TODO(jwd) - add exception handling
	training_targets = training_targets.reshape((I,))

	# append ones to the training_data
	_ones = np.ones((I, 1))
	_data = np.hstack((_ones, training_data))
	XXT = _data.T@_data
	XTX = _data@_data.T
	# compute product of X and w for computing mean of conditional distribution
	Xw = _data.T@training_targets
	
	def obj(_sigma_sq, _targets, _XTX, _sigma_sq_p):
		Sigma = _sigma_sq_p * _XTX + _sigma_sq * np.eye(_targets.size)
		return -np.log(stats.multivariate_normal.pdf(_targets.flatten(), mean=np.zeros((_targets.size,)).flatten(), cov=Sigma))

	sol = minimize(obj, 1000., args=(training_targets, XTX, sigma_sq_p), bounds=[(1e-2, 1e6)])
	
	if sol.success:
		sigma_sq = float(sol.x)
	else:
		raise RuntimeError('minimize failed')

	if D < I:
		A = XXT / sigma_sq + np.eye(D) / sigma_sq_p
		A_inv = np.linalg.inv(A)
	else:
		B = _data@_data.T + (sigma_sq / sigma_sq_p)*np.eye(D)
		A_inv = sigma_sq_p * (np.eye(D) - _data.T@np.linalg.inv(B)@_data)	
	return sigma_sq, A_inv, Xw

def gauss_proc_reg(training_data, training_targets, sigma_sq_p, f):
	# Algorithm 8.3
	I, D = training_data.shape
	# TODO(jwd) - add exception handling
	training_targets = training_targets.reshape((I,))

	# append ones to the training_data
	_ones = np.ones((I, 1))
	_data = np.hstack((_ones, training_data))
	XXT = _data.T@_data
	XTX = _data@_data.T

	# TODO(jwd) - vectorize
	K = np.zeros((I, I))
	for i in range(K.shape[0]):
		for j in range(K.shape[1]):
			K[i, j] = np.inner(f(_data[i]), f(_data[j]))

	def obj(_sigma_sq, _targets, _K, _sigma_sq_p):
		Sigma = _sigma_sq_p * _K + _sigma_sq * np.eye(_targets.size)
		return -np.log(stats.multivariate_normal.pdf(_targets.flatten(), mean=np.zeros((_targets.size,)).flatten(), cov=Sigma))

	sol = minimize(obj, 1000., args=(training_targets, K, sigma_sq_p), bounds=[(1e-2, 1e6)])
	
	if sol.success:
		sigma_sq = float(sol.x)
	else:
		raise RuntimeError('minimize failed')

	A = K + (sigma_sq / sigma_sq_p) * np.eye(I)
	return sigma_sq, np.linalg.inv(A), K, _data


def sparse_linear_regression(training_data, training_targets, nu):
	# Algorithm 8.4
	# TODO(jwd)
	raise NotImplementedError('Algorithm 8.4 is unavailable')


def dual_linear_regression_bayes(training_data, training_targets, sigma_sq_p):
	# Algorithm 8.5
	# TODO(jwd)
	raise NotImplementedError('Algorithm 8.5 is unavailable')


def dual_gauss_proc_reg(training_data, training_targets, sigma_sq_p, f):
	# Algorithm 8.6
	# TODO(jwd)
	raise NotImplementedError('Algorithm 8.6 is unavailable')


def relevance_vector_regression(training_data, training_targets, f, nu):
	# Algorithm 8.7
	# TODO(jwd)
	raise NotImplementedError('Algorithm 8.7 is unavailable')


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

class TestLinRegBayes(unittest.TestCase):
	def test_nominal(self):
		# just check to make sure the call is functioning for nominal input
		rng = np.random.RandomState(seed=0)
		X = np.linspace(-5, 5, 100).reshape((100, 1))
		# test parameters
		_sigma_sq = 0.2
		_phi = (5., 1.5)
		w = _phi[1] * X + _phi[0] * np.ones((100, 1)) + np.sqrt(_sigma_sq) * rng.randn(100,1)
		sigma_sq_p = 1.0
		with warnings.catch_warnings():
			warnings.filterwarnings('ignore', category=RuntimeWarning)
			sigma_sq, A_inv, Xw = linear_regression_bayes(X, w, sigma_sq_p)
			# compute the predictive distribution now
			_x = rng.randn(20, 1)
			x_star = np.hstack((np.ones((20, 1)), _x))
			for x in x_star:
				xT_Ainv = x@A_inv
				cond_mu = xT_Ainv@Xw / sigma_sq
				cond_sigma_sq = xT_Ainv@x.T + sigma_sq

class TestGaussianProcReg(unittest.TestCase):
	def test_simple(self):
		# just check to make sure the call is functioning for nominal input
		rng = np.random.RandomState(seed=0)
		I = 100
		X = np.linspace(-5, 5, I).reshape((I, 1))
		# test parameters
		_sigma_sq = 0.2
		_phi = (5., 1.5)
		w = _phi[1] * X + _phi[0] * np.ones((I, 1)) + np.sqrt(_sigma_sq) * rng.randn(I,1)
		sigma_sq_p = 1.0
		
		# test with simple identity function
		def f(x):
			return x

		with warnings.catch_warnings():
			warnings.filterwarnings('ignore', category=RuntimeWarning)
			sigma_sq, A_inv, K, _X = gauss_proc_reg(X, w, sigma_sq_p, f)
			# compute the predictive distribution now
			_x = rng.randn(20, 1)
			x_star = np.hstack((np.ones((20, 1)), _x))
			for x in x_star:
				k = np.zeros((1, I))
				for j in range(I):
					k[0, j] = np.inner(f(x), f(_X[j]))
				cond_mu = float((sigma_sq_p / sigma_sq) * k@(np.eye(I) - A_inv@K)@w)
				cond_sigma_sq = float(sigma_sq_p * (np.inner(f(x), f(x)) - k@A_inv@k.T) + sigma_sq)

# TODO(jwd) - add unit tests for Algorithms 8.4-8.7 after implementing

if __name__ == '__main__':
	unittest.main()
