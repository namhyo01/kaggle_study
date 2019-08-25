__author__ = 'Miroslaw Horbal'
__email__ = 'miroslaw@gmail.com'
__date__ = '2013-03-09'
import numpy as np
from scipy import optimize

OPTIMIZATION_FUNCTIONS = { 'bfgs': optimize.fmin_l_bfgs_b,
		'nc':optimize.fmin_tnc }

class LinearMAE(object):
	"""Linear model with Mean Absolute Error"""
	def __init__(self, l1=0.0, l2=0.0, opt='bfgs', maxfun=1000, 
			tol=1e-4, verbose=False):
		"""
		Parameters:
		  l1 - magnitude of l1 penalty (default 0.0)
		  l2 - magnitude of l2 penalty (default 0.0)
		  opt - optimization algorithm to use for gardient decent 
				options are 'cg', 'bfgs' (default 'bfgs')
		  maxiter - maximum number of iterations (default 1000)
		  tol - terminate optimization if gradient l2 is smaller than tol (default 1e-4)
		  verbose - display convergence information at each iteration (default False)
		"""
		self.opt = opt
		self.maxfun = maxfun
		self.tol = tol
		self.l1 = l1
		self.l2 = l2
		self.verbose = verbose

	@property
	def opt(self):
		"""Optimization algorithm to use for gradient decent"""
		return self._opt

	@opt.setter
	def opt(self, o):
		"""
		Set the optimization algorithm for gradient decent

		Parameters:
		  o - 'cg' for conjugate gradient decent
			  'bfgs' for BFGS algorithm
		"""
		if o not in OPTIMIZATION_FUNCTIONS:
			raise Error('Unknown optimization routine %s' % o)
		self._opt = o
		self._optimize = OPTIMIZATION_FUNCTIONS[o]

	def score(self, X, y):
		"""
		Compute the MAE of the linear model prediction on X against y

		Must only be run after calling fit

		Parameters:
		  X - data array for the linear model. Has shape (m,n)
		  y - output target array for the linear model. Has shape (m,o)
		"""
		#y = _2d(y)
		pred = self.predict(X)
		return np.mean(np.abs(pred - y))

	def predict(self, X):
		"""
		Compute the linear model prediction on X

		Must only be run after calling fit

		Parameters:
		  X - data array for the linear model. Has shape (m,n)
		"""
		return (X.dot(self.coef_[1:]) + self.coef_[0]) / len(self.coef_ - 1)
		#return X.dot(self.coef_[1:]) + self.coef_[0] 

	def fit(self, X, y, coef=None):
		"""
		Fit the linear model using gradient decent methods

		Parameters:
		  X - data array for the linear model. Has shape (m,n)
		  y - output target array for the linear model. Has shape (m,o)
		  coef - None or array of size (n+1) * o

		Sets attributes:
		  coef_ - the weights of the linear model
		"""
		#y = _2d(y)
		#m, n = X.shape
		#m, o = y.shape
		#if coef is None:
		#	coef = np.zeros((n+1, o))
		#elif coef.shape != (n+1, o):
		#	raise Error('coef must be None or be shape %s' % (str((n+1, o))))
		#self._coef_shape = coef.shape
		m, n = X.shape
		if coef is None:
			coef = np.zeros( n+1 )
		coef = self._optimize(func=cost_grad,
				x0 = coef, #coef.flatten(),
				args=( X, y, self.l1, self.l2),
			  	#gtol=self.tol,
				maxfun=self.maxfun,
			 	disp=0 )[0]
			  	#callback=self._callback(X,y))[0]
		#self.coef_ = np.reshape(coef, self._coef_shape)
		self.coef_ = coef
		return self

	def _callback(self, X, y):
		"""
		Helper method that generates a callback function for the optimization
		algorithm opt if verbose is set to True
		"""
		def callback(coef):
			self.i += 1
			#self.coef_ = np.reshape(coef, self._coef_shape)
			score = self.score(X, y)
			print 'iter %i | Score: %f\r' % (self.i, score)
		self.i = 0
		return callback if self.verbose else None

def cost_grad(coef, X, y, l1=0, l2=0):
	"""
	Compute the cost of a linear model with mean absolute error:

	  cost = X.dot(coef) + l1*mean(abs(coef[1:])) + 0.5*l2*mean(coef[1:] ** 2) 

	Parameters:
	  coef - the weights of the linear model must have size (n + 1)*o
	  X - data array for the linear model. Has shape (m,n)
	  y - output target array for the linear model. Has shape (m,o)
	  l1 - magnitude of the l1 penalty
	  l2 - magnitude of the l2 penalty
	"""
	m, n = X.shape
	Xb = np.hstack((np.ones((m,1)), X))
	pred = Xb.dot(coef)
	pred = pred / (len(coef) - 1)
	err = pred - y

	c = np.mean(np.abs(err))
	c_l1 = np.mean(np.abs(coef[1:])) if l1 > 0 else 0
	c_l2 = np.mean(np.square(coef[1:])) if l2 > 0 else 0

	derr = Xb.T.dot(np.sign(err)) / m
	#dl1 = np.vstack((np.zeros(1), np.sign(coef[1:]))) if l1 > 0 else 0 
	#dl2 = np.vstack((np.zeros(1), np.copy(coef[1:]))) if l2 > 0 else 0
	dl1 = np.sign( coef ) if l1 > 0 else 0
	dl1[0] = 0
	dl2 = np.copy( coef ) if l2 > 0 else 0
	dl2[0] = 0

	cost = c + l1 * c_l1 + 0.5 * l2 * c_l2
	grad = (derr + l1 * dl1 + l2 * dl2).flatten()
	return cost, grad

def _2d(a):
	"""Returns a 2d array of a if rank a <= 2"""
	if len(a.shape) == 1:
		a.shape = (len(a), 1)
	return a
