import numpy as np
import scipy.special as sci_spec
import sys
sys.path.append("..")
from importlib import reload
import common.utilities as common
reload(common)

import unittest


def learned_univariate_normal(data, method, **kwargs):
	try:
		data = np.float_(data.ravel())
	except AttributeError as e:
		raise AttributeError('\'data\' input must be a numpy array/matrix type.')
	I = float(data.size)
	if method.lower() == 'ml':
		# algorithm 4.1
		mu = np.sum(data) / I
		var = np.sum([(x - mu)**2 for x in data]) / I
		return mu, var
	elif method.lower() == 'map':
		# algorithm 4.2
		try:
			for k,v in kwargs.items():
				if not isinstance(v, (int, float)):
					raise ValueError()
			alpha, beta, gamma, delta = kwargs['alpha'], kwargs['beta'], kwargs['gamma'], kwargs['delta']
			assert(min(alpha, beta, gamma) > 0)
			mu = (np.sum(data) + float(gamma) * float(delta)) / (I + float(gamma))
			var = (np.sum([(x - mu)**2 for x in data]) + 2 * float(beta) + float(gamma) * (float(delta) - mu)**2) / (I + 3 + 2 * float(alpha))
			return mu, var
		except AssertionError:
			raise AssertionError('Make sure that \'alpha\', \'beta\', and \'gamma\' are all positive.')
		except KeyError:
			raise KeyError('Make sure to set \'alpha\', \'beta\', \'gamma\', and \'delta\'.')
		except ValueError:
			raise ValueError('Make sure that all inputs are numeric.')
	elif method.lower() == 'bayes':
		# algorithm 4.3
		try:
			for k,v in kwargs.items():
				if not isinstance(v, (int, float)):
					raise ValueError()
			alpha, beta, gamma, delta, x_star = kwargs['alpha'], kwargs['beta'], kwargs['gamma'], kwargs['delta'], kwargs['x_star']
			assert(min(alpha, beta, gamma) > 0)
			x_star = float(x_star)
			# Compute normal inverse gamma posterior over normal parameters
			alpha_tilde = alpha + 0.5 * I
			beta_tilde = 0.5 * np.sum([xi**2 for xi in data]) + float(beta) + 0.5 * float(gamma) * delta**2 - (float(gamma) *delta + np.sum(data))**2 / (2 * float(gamma) + 2 * I)
			gamma_tilde = gamma + I
			delta_tilde = (float(gamma) * delta + np.sum(data)) / gamma_tilde
			# Compute intermediate parameters
			alpha_scoop = alpha_tilde + 0.5
			beta_scoop = 0.5 * x_star**2 + beta_tilde + 0.5 * gamma_tilde * delta_tilde**2 - (delta_tilde * gamma_tilde + x_star)**2 / (2 * gamma_tilde + 2)
			gamma_scoop = gamma_tilde + 1
			# Evaluate new datapoint under predictive distribution
			pred_prob_xstar_given_data = (np.sqrt(gamma_tilde) * beta_tilde**alpha_tilde * sci_spec.gamma(alpha_scoop)) / (np.sqrt(2 * np.pi) * np.sqrt(gamma_scoop) * beta_scoop**alpha_scoop * sci_spec.gamma(alpha_tilde))
			return alpha_tilde, beta_tilde, gamma_tilde, delta_tilde, pred_prob_xstar_given_data
		except AssertionError:
			raise AssertionError('Make sure that \'alpha\', \'beta\', and \'gamma\' are all positive.')
		except KeyError:
			raise KeyError('Make sure to set \'alpha\', \'beta\', \'gamma\', \'delta\', and \'x_star\'.')
		except ValueError:
			raise ValueError('Make sure that inputs have the correct type.')
	else:
		raise ValueError('Only maximum likelihood (\'ML\'), ' +
			'maximum a posteriori (\'MAP\'), and Bayes\' (\'(Bayes\') methods are supported.')


def learned_categorical(data, num_values, method, **kwargs):
	''' count is 1-based '''
	try:
		data = np.int_(data.ravel())
	except AttributeError as e:
		raise AttributeError('\'data\' input must be a numpy array/matrix type.')
	I = float(data.size)
	if not isinstance(num_values, (int, float)):
		raise ValueError('\'num_values\' must be numeric.')
	_lambda = np.zeros((num_values,))
	if method.lower() == 'ml':
		# algorithm 4.4
		for k in range(1, num_values+1):
			_lambda[k-1] = float(np.sum([common.delta_function(d-k) for d in data])) / I
		return _lambda
	elif method.lower() == 'map':
		# algorithm 4.5
		try:
			alpha = kwargs['alpha']
			assert(all([a > 0 for a in alpha]))
			sum_alpha = np.sum(alpha)
			K = float(num_values)
			for k in range(1, num_values+1):
				Nk = float(np.sum([common.delta_function(d-k) for d in data]))
				_lambda[k-1] = (Nk - 1 + alpha[k-1]) / (I - K + sum_alpha)
			return _lambda
		except AssertionError:
			raise AssertionError('Make sure that \'alpha\' entries are all positive numbers.')
		except KeyError:
			raise KeyError('Make sure to set \'alpha\'.')
		except ValueError:
			raise ValueError('Make sure that all inputs are numeric.')
	elif method.lower() == 'bayes':
		# algorithm 4.6
		try:
			alpha = kwargs['alpha']
			assert(all([a > 0 for a in alpha]))
			x_star = int(kwargs['x_star'])
			# Compute categorical posterior over lambda
			alpha_tilde = [alpha[k-1] + np.sum([common.delta_function(d - k) for d in data]) for k in range(1, num_values+1)]
			sum_alpha_tilde = float(np.sum(alpha_tilde))
			# Evaluate new datapoint under predictive distribution
			pred_prob_xstar_given_data = float(alpha_tilde[x_star-1]) / sum_alpha_tilde
			return alpha_tilde, pred_prob_xstar_given_data
		except AssertionError:
			raise AssertionError('Make sure that \'alpha\' entries are all positive numbers.')
		except TypeError:
			raise TypeError('\'x_star\' must be convertable to an integer.')
		except KeyError:
			raise KeyError('Make sure to set \'alpha\' and x_star.')
		except ValueError:
			raise ValueError('Make sure that all inputs are numeric.')
	else:
		raise ValueError('Only maximum likelihood (\'ML\'), ' +
			'maximum a posteriori (\'MAP\'), and Bayes\' (\'(Bayes\') methods are supported.')


class TestUnivariateNormalFit(unittest.TestCase):
	mu, var = 0., 1.
	n_samples = 50
	np.random.seed(0)
	data = var * np.random.randn(n_samples) + mu
	alpha, beta, gamma, delta, x_star = 1, 1, 1, 0, -0.1

	def test_bogus_input(self):
		# TEST 1: TEST BAD DATA
		data = None
		method = 'bogus method'
		self.assertRaises(AttributeError, learned_univariate_normal, data, method)
		# TEST 2: TEST BAD METHOD
		data = np.copy(self.data)
		self.assertRaises(ValueError, learned_univariate_normal, data, method)
		# TEST 3: TEST BAYES INPUT VALUES MISSING 
		method = 'bayes'
		a, b, c, d = -1., 1., 2., 3.
		self.assertRaises(KeyError, learned_univariate_normal, data, method, alpha=a, beta=b, gamma=c, delta=d)
		# TEST 4: TEST BAYES INVALID INPUT VALUES
		a, b, c, d, e = 1., 1., 2., 3., None
		self.assertRaises(ValueError, learned_univariate_normal, data, method, alpha=a, beta=b, gamma=c, delta=d, x_star=e)
		
	def test_univariate_normal_ml(self):
		# TEST 1: check that sample mean and standard deviation of data is returned
		method = 'ml'
		mu_hat, var_hat = learned_univariate_normal(self.data, method)
		self.assertAlmostEqual(mu_hat, np.mean(self.data), places=6)
		self.assertAlmostEqual(var_hat, np.var(self.data), places=6)

	def test_univariate_normal_map(self):
		# TEST 1: test equality with result equations 4.18, 4.19
		method = 'map'
		a, b, c, d = -1., 1., 2., 3.
		self.assertRaises(AssertionError, learned_univariate_normal, self.data, method, alpha=a, beta=b, gamma=c, delta=d)
		# TEST 2: TEST INPUT VALUES MISSING
		a, b, c, d = 1., 1., 2., 3.
		self.assertRaises(KeyError, learned_univariate_normal, self.data, method, alpha=a, beta=b, gamma=c)
		# TEST 3: TEST INVALID INPUT VALUES
		a, b, c, d = None, 1., 2., 3.
		self.assertRaises(ValueError, learned_univariate_normal, self.data, method, alpha=a, beta=b, gamma=c, delta=d)
		# TEST 4: VALID VALUES
		a, b, c, d = self.alpha, self.beta, self.gamma, self.delta
		mu_hat, var_hat = learned_univariate_normal(self.data, method, alpha=a, beta=b, gamma=c, delta=d)
		I = float(self.data.size)
		mu_map = (I * np.mean(self.data) + float(c) * d) / (I + c)
		var_map = (np.sum([(xi-mu_map)**2 for xi in self.data]) + 2. * b + c * (d - mu_map)**2) / (I + 3 + 2. * a)
		self.assertAlmostEqual(mu_hat, mu_map, places=6)
		self.assertAlmostEqual(var_hat, var_map, places=6)

	def test_univariate_normal_bayes(self):
		method = 'bayes'
		a, b, c, d, xs = -1., 1., 2., 3., -0.1
		self.assertRaises(AssertionError, learned_univariate_normal, self.data, method, alpha=a, beta=b, gamma=c, delta=d, x_star=xs)
		# TEST 2: TEST INPUT VALUES MISSING
		a, b, c, d = 1., 1., 2., 3.
		self.assertRaises(KeyError, learned_univariate_normal, self.data, method, alpha=a, beta=b, gamma=c)
		# TEST 3: TEST INVALID INPUT VALUES
		a, b, c, d = None, 1., 2., 3.
		self.assertRaises(ValueError, learned_univariate_normal, self.data, method, alpha=a, beta=b, gamma=c, delta=d, x_star=xs)


class TestCategoricalFit(unittest.TestCase):
	mu, var = 0., 1.
	n_samples = 50
	n_outcomes = 10
	np.random.seed(0)
	_lambda = np.array([a+1 for a in range(n_outcomes)], dtype=np.float32)
	_lambda *= 1./np.sum(_lambda)
	data = np.random.multinomial(n_samples, _lambda, size=1)[0]
	alpha = np.ones((n_outcomes,))

	def test_bogus_input(self):
		# TEST 1: TEST BAD DATA
		data = None
		method = 'bogus method'
		self.assertRaises(AttributeError, learned_categorical, data, self.n_outcomes, method)
		# TEST 2: TEST BAD NUM OUTCOMES
		bad_outcomes = None
		self.assertRaises(ValueError, learned_categorical, self.data, bad_outcomes, method)
		# TEST 3: TEST BAD ALPHAS
		method = 'map'
		self.assertRaises(AssertionError, learned_categorical, self.data, self.n_outcomes, method, alpha=np.zeros((self.n_outcomes,)))
		# TEST 4: TEST BAD METHOD
		data = np.copy(self.data)
		method = 'bogus_method'
		self.assertRaises(ValueError, learned_categorical, data, self.n_outcomes, method)
		# TEST 5: TEST BAYES INPUT VALUES MISSING 
		method = 'bayes'
		self.assertRaises(KeyError, learned_categorical, data, self.n_outcomes, method, alpha=np.ones((self.n_outcomes,)))
		# TEST 6: TEST BAYES INVALID INPUT VALUES
		self.assertRaises(TypeError, learned_categorical, data, self.n_outcomes, method, alpha=np.ones((self.n_outcomes,)), x_star=None)

	def test_categorical_ml(self):
		method = 'ml'
		_lambda = learned_categorical(self.data, self.n_outcomes, method)
		# TEST 1: TEST PREDICTED DISTRIBUTION IS VALID (sums to 1)
		self.assertTrue(np.isclose(np.sum(_lambda), 1))
		
	def test_categorical_map(self):
		method = 'map'
		_lambda = learned_categorical(self.data, self.n_outcomes, method, alpha=self.alpha)
		# TEST 1: TEST PREDICTED DISTRIBUTION IS VALID (sums to 1)
		self.assertTrue(np.isclose(np.sum(_lambda), 1))

	def test_categorical_bayes(self):
		method = 'bayes'
		pr = np.zeros((self.n_outcomes,))
		for i in range(1, self.n_outcomes+1):
			_, pr[i-1] = learned_categorical(self.data, self.n_outcomes, method, alpha=self.alpha, x_star=i)
		# TEST 1: TEST PREDICTED DISTRIBUTION OVER SET OF x_star IS VALID (sums to 1)
		self.assertTrue(np.isclose(np.sum(pr), 1))

if __name__ == '__main__':
	unittest.main()
