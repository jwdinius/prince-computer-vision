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

'''
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
'''

def fit_gaussian_mixture(training_data, num_clusters, stopping_thresh=1e-2):
	# Algorithm 7.1
	I = float(training_data.shape[0])
	num_clusters = int(num_clusters)
	if num_clusters < 1:
		raise AssertionError('`num_clusters` must be >= 1')

	# initialization
	_lambdas = np.array([1./float(num_clusters) for _ in range(num_clusters)])
	rng = np.random.RandomState(seed=0)
	mus = rng.randn(num_clusters, training_data.shape[1])
	Sigmas = np.zeros((num_clusters, training_data.shape[1], training_data.shape[1]))
	for k in range(num_clusters):
		sqrt_Sigmas = rng.randn(training_data.shape[1], training_data.shape[1])
		# setup a diagonal covariance matrix
		Sigmas[k, :, :] = sqrt_Sigmas@sqrt_Sigmas.T

	# initialize log-likelihood
	L_prev = None
	its = 0
	while True:
		its += 1
		# Expectation Step
		l = np.zeros((int(I), num_clusters))
		r = np.zeros((int(I), num_clusters))
		for i,_data in enumerate(training_data):
			for k in range(num_clusters):
				# numerator of Bayes' rule
				l[i, k] = stats.multivariate_normal.pdf(_data, mean=mus[k, :], cov=Sigmas[k, :, :])
			total_prob = np.sum(l[i, :])
			for k in range(num_clusters):
				# Compute posterior (responsibilities) by normalizing
				r[i, k] = l[i, k] / total_prob
		# Maximization Step
		nrmlzr = np.sum(np.sum(r, axis=0))
		for k in range(num_clusters):
			sum_r = np.sum(r[:, k])
			_lambdas[k] = np.sum(r[:, k]) / nrmlzr
			weighted_data = np.zeros((1, training_data.shape[1]))
			for i in range(int(I)):
				weighted_data += r[i, k] * training_data[i, :]
			mus[k, :] = weighted_data / sum_r
			new_Sigma = np.zeros((training_data.shape[1], training_data.shape[1]))
			for i in range(int(I)):
				delta = (training_data[i, :] - mus[k, :]).reshape((1, 2))
				resp_weighted_delta = r[i, k] * delta.T@delta
				new_Sigma += resp_weighted_delta
			#Sigmas[k, :, :] = np.diag(np.diag(new_Sigma /sum_r))
			Sigmas[k, :, :] = new_Sigma /sum_r
		# Compute Data Log Likelihood and EM Bound
		tmp = np.zeros((int(I), num_clusters))
		for i in range(int(I)):
			for k in range(num_clusters):
				tmp[i, k] = _lambdas[k] * stats.multivariate_normal.pdf(training_data[i, :], mean=mus[k, :], cov=Sigmas[k, :, :])
		tmp = np.sum(tmp, axis=1)
		L = np.sum(np.log(tmp))
		# if uninitialized, set L_prev and move on to next loop iteration
		if L_prev is None:
			L_prev = L
			continue
		if np.abs(L-L_prev) < stopping_thresh:
			print('stopping criteria met after {its} iterations.'.format(its=its))
			break
		L_prev = L
	return _lambdas, mus, Sigmas


def fit_student_distribution(training_data, stopping_thresh=1e-2):
	# Algorithm 7.2
	I, D = [float(t) for t in training_data.shape]
	
	# initialization: follows footnote a) of algorithm guide
	rng = np.random.RandomState(seed=0)
	mu = np.mean(training_data, axis=0)
	x_minus_mu = np.array([d - mu for d in training_data])
	Sigma = np.zeros((int(D), int(D)))
	for d in x_minus_mu:
		Sigma += np.outer(d, d) / I
	nu = 10.
	
	# define objective function for maximizing likelihood wrt nu
	def t_fit_cost(nu, _E_h, _E_logh):
		return -(float(_E_h.size) * (0.5 * nu * np.log(0.5 * nu) + sci_spec.gammaln(0.5 * nu)) - (0.5 * nu - 1.) * np.sum(_E_logh) + 0.5 * nu * np.sum(_E_h))

	# initialize log-likelihood
	L_prev = None
	its = 0
	while True:
		its += 1
		# Expectation step
		x_minus_mu = np.array([d - mu for d in training_data])
		inv_sigma = np.linalg.inv(Sigma)
		delta = np.array([d.reshape((1, int(D)))@inv_sigma@d.reshape((int(D), 1)) for d in x_minus_mu]).flatten()
		E_h = np.array([(nu + D) / (nu + d) for d in delta]).flatten()
		E_logh = np.array([sci_spec.digamma((nu+D)/2) - np.log((nu+float(d))/2.) for d in delta]).flatten()
		
		# Maximization step
		sum_E_h = np.sum(E_h)
		sum_E_h_times_x = np.zeros((1, int(D)))
		for i in range(int(I)):
			sum_E_h_times_x += E_h[i] * training_data[i]
		mu = sum_E_h_times_x / sum_E_h
		x_minus_mu = np.array([d - mu for d in training_data])
		Sigma = np.zeros((int(D), int(D)))
		for i in range(int(I)):
			d = x_minus_mu[i].reshape((1, int(D)))
			Sigma += E_h[i] * np.outer(d, d) / sum_E_h
		inv_sigma = np.linalg.inv(Sigma)
		delta = np.array([d.reshape((1, int(D)))@inv_sigma@d.reshape((int(D), 1)) for d in x_minus_mu]).flatten()
		x0 = np.array([[nu]])
		bnds = [(1e-15, 1e3)]
		sol = minimize(t_fit_cost, x0, args=(E_h, E_logh), bounds=bnds)
		nu = sol.x

		# Compute the log likelihood
		L = I * (sci_spec.gammaln((nu + D)/2) - (D/2) * np.log(nu * np.pi)- np.log(np.linalg.det(Sigma)) / 2 - sci_spec.gammaln(nu/2))
		s = np.sum([np.log(1. + d/nu) for d in delta])
		L -= (nu + D) * s
		if L_prev is None:
			L_prev = L
			continue
		if np.abs(L-L_prev) < stopping_thresh:
			print('stopping criteria met after {its} iterations.'.format(its=its))
			break
		L_prev = L
	return mu, Sigma, nu


class TestFMOG(unittest.TestCase):
	def test_basic_call(self):
		# just check to make sure the call is functioning for nominal input
		rng = np.random.RandomState(seed=0)
		mu1 = np.array([1., 2.])
		cov1 = np.array([[2., 0.], [0., .5]])
		mu2 = np.array([3., 5.])
		cov2 = np.array([[1., 0.], [0., .1]])
		X = np.vstack((rng.multivariate_normal(mu1, cov1, 500), rng.multivariate_normal(mu2, cov2, 500)))
		_lambdas, mus, Sigmas = fit_gaussian_mixture(X, 2)

class TestStudentFit(unittest.TestCase):
	# just check to make sure the call is functioning for nominal input
	rng = np.random.RandomState(seed=0)
	mu1 = np.array([1., 2.])
	cov1 = np.array([[2., 0.], [0., .5]])
	num_data_pts = np.array([200])
	X = rng.multivariate_normal(mu1, cov1, num_data_pts[0])
	num_outliers = 15
	outliers_mu = np.array([5., 7.])
	outliers_Sigma = np.array([[.2, 0.], [0., .2]])
	X_with_outliers = np.vstack((X, rng.multivariate_normal(outliers_mu, outliers_Sigma, num_outliers)))
	t_mu, t_sig, t_nu = fit_student_distribution(X_with_outliers, stopping_thresh=1e-6)

if __name__ == '__main__':
	unittest.main()
