import numpy as np
import unittest
import scipy.special as sci_spec

def pdf(X, mu, Sigma, nu):
	I, D = [float(t) for t in X.shape]
	if X.shape[1] != mu.size:
		raise AssertionError('`X` and `mu` have incompatible sizes')
	if mu.size != Sigma.shape[0] or Sigma.shape[0] != Sigma.shape[1]:
		raise AssertionError('`mu` and `Sigma` have incompatible sizes')
	if not np.all(np.all([[np.isclose(Sigma[i, j], Sigma[j, i]) for j in range(Sigma.shape[0])] for i in range(Sigma.shape[0])])):
		raise AssertionError('`Sigma` is not symmetric!')
	# TODO: check that matrix is pos-def
	if np.isclose(nu, 0) or nu < 0:
		raise ValueError('`nu` must be > 0')
	normalizer = np.exp(sci_spec.gammaln(0.5 * (nu+D)) - sci_spec.gammaln(0.5*nu)) / ((nu * np.pi)**(0.5*D) * np.sqrt(np.linalg.det(Sigma)))
	x_minus_mu = np.array([d - mu for d in X])
	inv_sigma = np.linalg.inv(Sigma)
	delta = np.array([d.reshape((1, int(D)))@inv_sigma@d.reshape((int(D), 1)) for d in x_minus_mu]).flatten()
	return normalizer * np.power((1. + delta / nu), -(0.5 * (nu+D)))
	

class TestMultivariateStudent(unittest.TestCase):
	def test_pdf(self):
		pass


if __name__ == '__main__':
	unittest.main()
