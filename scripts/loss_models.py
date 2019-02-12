import numpy as np
import sklearn.linear_model
from scipy.optimize import minimize, newton

class LossModel:
    def theta(self, X, y, w):
        """
        Returns the optimal parameters for a given weighting of the dataset
        X -- (N, D) array of the N points
        y -- (N,) array of the labels
        w -- (N,) array of the sample weights
        returns theta as an (N,) matrix
        """
        raise NotImplementedError()
    
    def L(self, X, y, w, theta):
        """
        Computes the weighted loss on a set of points
        
        """
        raise NotImplementedError()
        
    def G(self, X, y, w, theta):
        """
        Computes the individual loss gradients of a set of points
        Returns an (N, D) matrix
        """
        raise NotImplementedError()
    
    def g(self, X, y, w, theta):
        """
        Computes the total loss gradient of a set of points
        Returns a (D,) array
        """
        y, w = np.array(y), np.array(w)
        return np.sum(self.G(X, y, w, theta), axis=0).T
    
    def H(self, X, y, w, theta):
        """
        Computes the total hessian of a set of points
        """
        raise NotImplementedError()
    
    def reshape_scalar(self, X, s):
        s = np.array(s).reshape(-1)
        return np.full(X.shape[0], s)
    
class LogisticRegression(LossModel):
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))
    
    def optimize_direct_bfgs(self, X, y):
        def objective(theta):
            return -np.mean(np.log(sigmoid(y * np.dot(X, theta))))
        res = minimize(objective, np.zeros(X.shape[1]), method='BFGS', tol=1e-8)
        return np.array(res.x.reshape(-1))

    def optimize_sklearn(self, X, y, w=None):
        model = sklearn.linear_model.LogisticRegression(C=1e8, fit_intercept=False, solver='liblinear', tol=1e-8)
        model.fit(X, y, sample_weight=w)
        return np.array(model.coef_.reshape(-1))
    
    def theta(self, X, y, w):
        y, w = self.reshape_scalar(X, y), self.reshape_scalar(X, w) # allow scalar parameters
        return self.optimize_sklearn(X, y, w)
    
    def L(self, X, y, w, theta):
        y, w = self.reshape_scalar(X, y), self.reshape_scalar(X, w) # allow scalar parameters
        sigmoids = self.sigmoid(y * np.dot(X, theta))
        return np.dot(w, -np.log(sigmoids))
    
    def G(self, X, y, w, theta):
        y, w = self.reshape_scalar(X, y), self.reshape_scalar(X, w) # allow scalar parameters
        sigmoids = self.sigmoid(y * np.dot(X, theta))
        return -(w * y * (1 - sigmoids))[:, np.newaxis] * X
    
    def H(self, X, y, w, theta):
        y, w = self.reshape_scalar(X, y), self.reshape_scalar(X, w) # allow scalar parameters
        sigmoids = self.sigmoid(y * np.dot(X, theta))
        # note: no averaging!
        return np.dot(X.T * (w * sigmoids * (1 - sigmoids)), X)
    
class LinearRegression(LossModel):
    def theta(self, X, y, w):
        y, w = self.reshape_scalar(X, y), self.reshape_scalar(X, w) # allow scalar parameters
        model = sklearn.linear_model.LinearRegression(fit_intercept=False)
        model.fit(X, y, sample_weight=w)
        return np.array(model.coef_.reshape(-1))
    
    def L(self, X, y, w, theta):
        y, w = self.reshape_scalar(X, y), self.reshape_scalar(X, w) # allow scalar parameters
        return 0.5 * np.sum(w * (np.dot(X, theta) - y) ** 2)
    
    def G(self, X, y, w, theta):
        y, w = self.reshape_scalar(X, y), self.reshape_scalar(X, w) # allow scalar parameters
        return (w * (np.dot(X, theta) - y))[:, np.newaxis] * X
    
    def H(self, X, y, w, theta):
        y, w = self.reshape_scalar(X, y), self.reshape_scalar(X, w) # allow scalar parameters
        return np.dot(X.T * w, X)