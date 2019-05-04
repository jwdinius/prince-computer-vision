import numpy as np
import scipy.special as sci_spec
import scipy.stats as stats
import sys
sys.path.append("..")
from importlib import reload
import common.utilities as common
reload(common)

import unittest

def evaluate_posterior_probabilities(mus, Sigmas, _lambdas, new_datapoint):
	if mus.shape[0] != _lambdas.size or mus.shape[0] != Sigmas.shape[0]:
		raise AssertionError('\'mus\', \'Sigmas\', and/or \'_lambdas\' sizes don\'t match.')
	if mus[0].size != new_datapoint.size:
		raise AssertionError('\'mus\' and \'new_datapoint\' sizes don\'t match.')
	if mus[0].size != Sigmas[0].shape[0] or Sigmas[0].shape[0] != Sigmas[0].shape[1]:
		raise AssertionError('\'mus\' and \'Sigmas\' sizes don\'t match.')
	likelihoods = np.zeros((mus.shape[0],))
	posteriors = np.zeros_like(likelihoods)
	for l in range(mus.shape[0]):
		likelihoods[l] = stats.multivariate_normal.pdf(new_datapoint, mean=mus[l, :], cov=Sigmas[l, :, :])
	denom = np.sum([p*lam for p, lam in zip(likelihoods, _lambdas)])
	# Classify new datapoint using Bayes' rule
	# (NOTE(jwd) - this approach is slow, but follows the pseudocode in the algorithm manual)
	for i in range(mus.shape[0]):
		posteriors[i] = likelihoods[i] * _lambdas[i] / denom
	return posteriors
	
def basic_generative_classifier(num_labels, training_data, training_labels):
	# Algorithm 6.1
	training_labels = np.copy(training_labels).ravel()
	valid_labels = np.array([k for k in range(num_labels)])
	if not all([l in valid_labels for l in training_labels]):
		raise ValueError('\'training_labels\' is invalid given \'num_labels\'.')

	if not training_data.shape[1] == training_labels.size:
		raise ValueError('\'training_data\' and \'training_labels\' sizes don\'t match.')

	I = float(training_labels.size)
	
	# For each training class
	# initialize array for lambda prior
	_lambdas = np.zeros((num_labels,))
	mus = np.zeros((num_labels, training_data.shape[0]))
	Sigmas = np.zeros((num_labels, training_data.shape[0], training_data.shape[0]))

	for k in valid_labels:
		_labels = training_labels[training_labels==k]
		_data = training_data[:, training_labels==k]
		denom = np.sum([common.delta_function(wi-k) for wi in training_labels])
		# Set mean
		# (NOTE(jwd): use the transpose of training data because of how numpy iterates over containers)
		mu_k = np.sum([xi*common.delta_function(wi-k) for xi,wi in zip(training_data.T, training_labels)]) / denom
		mus[k, :] = mu_k
		# Set variance
		# (NOTE(jwd): use the transpose of training data because of how numpy iterates over containers)
		Sigma_k = np.sum([(xi-mu_k)@(xi-mu_k).T * common.delta_function(wi-k) for xi,wi in zip(training_data.T, training_labels)]) / denom
		Sigmas[k, :, :] = Sigma_k
		# Set prior
		_lambdas[k] = denom / I
	return _lambdas, mus, Sigmas


class TestBasicGenerativeClassifier(unittest.TestCase):
	mu_0, var_0 = -2., 0.5
	mu_1, var_1 = 2., 0.5
	num_labels = 2
	n_samples = 100
	distro = np.random.RandomState(seed=0)
	data = np.empty((1, 2*n_samples))
	data[0, :n_samples] = np.sqrt(var_0) * distro.randn(n_samples) + mu_0
	data[0, n_samples:] = np.sqrt(var_1) * distro.randn(n_samples) + mu_1
	labels = np.empty((2*n_samples,))
	labels[:n_samples] = 0
	labels[n_samples:] = 1

	def test_invalid_labels(self):
		num_labels = 1
		self.assertRaises(ValueError, basic_generative_classifier, num_labels, self.data, self.labels)

	def test_invalid_sizes(self):
		# vary size of input data by 1
		self.assertRaises(ValueError, basic_generative_classifier, self.num_labels, self.data[0, 1:].reshape((1, 199)), self.labels)

	def test_output_no_datapoint(self):
		_lambdas, mus, Sigmas = basic_generative_classifier(self.num_labels, self.data, self.labels)
		self.assertTrue(np.isclose(_lambdas[0], _lambdas[1]))
		self.assertAlmostEqual(mus[0, 0], self.mu_0, places=0)
		self.assertAlmostEqual(mus[1, 0], self.mu_1, places=0)
		self.assertAlmostEqual(Sigmas[0, 0, 0], self.var_0, places=0)
		self.assertAlmostEqual(Sigmas[1, 0, 0], self.var_1, places=0)

	def test_output_datapoint(self):
		# make sure that it is more likely to have one label when datapoint is near the mean of one world distribution
		_lambdas, mus, Sigmas = basic_generative_classifier(self.num_labels, self.data, self.labels)
		posterior_probs = evaluate_posterior_probabilities(mus, Sigmas, _lambdas, np.array([[-3.]]))
		self.assertTrue(posterior_probs[0] > 0.5)
		posterior_probs = evaluate_posterior_probabilities(mus, Sigmas, _lambdas, np.array([[3.]]))
		self.assertTrue(posterior_probs[1] > 0.5)
		

if __name__ == '__main__':
	unittest.main()
