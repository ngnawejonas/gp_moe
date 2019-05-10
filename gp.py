# coding: utf-8

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from scipy.optimize import minimize

###############################################################################
def mat_sqrt(C, cholesky=False):
    if cholesky:
#         print('cholesky decomposition on ', self.cholesky)
        return np.linalg.cholesky(C)
    return sc.linalg.sqrtm(C)

def mat_square(L, cholesky=False):
    if cholesky:
        return L @ L.T
    return L @ L

def is_hermitian(A, epsilon=1e-7):
    if np.linalg.norm(A.H - A) < epsilon:
        return True
    return False

def is_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def is_psd(A, epsilon=1e-7, pd=True, cholesky_test=False):
    if cholesky_test:
        try:
            np.linalg.cholesky(A)
            return True
        except:
            return False
            
    else:
        if is_symmetric(A):
            eigvals = np.linalg.eigvalsh(A)
            real_part_of_eigvals = np.real(eigvals)
        else:
            eigvals = np.linalg.eigvals((A+A.T)/2)
            real_part_of_eigvals = np.real(eigvals)

        if pd is True:
#             print("real part of eigvals = ", real_part_of_eigvals)
#             print("eigvals = ", eigvals)
            return np.all(real_part_of_eigvals>0)
        else:
            return np.all(real_part_of_eigvals>=0)
###############################################################################
def partitionXX(X, Y, n):
    # assuming X is NxD, y is Nx1
    N, D = X.shape
    X_p = np.zeros((n, N//n, D))
    Y_p = np.zeros((n, N//n, 1))
    for i in range(n):
        X_p[i] = X[i*(N//n): (i+1)*(N//n)]
        Y_p[i] = Y[i*(N//n): (i+1)*(N//n)]

    return X_p, Y_p
################################################################################


def plot_gp(X, m, C, training_points=None):
    """ Plotting utility to plot a GP fit with 95% confidence interval """
    # Plot 95% confidence interval
    plt.fill_between(X[:, 0],
                     m[:, 0] - 1.96*np.sqrt(np.diag(C)),
                     m[:, 0] + 1.96*np.sqrt(np.diag(C)),
                     alpha=0.5)
    # Plot GP mean and initial training points
    plt.plot(X, m, "-")
    plt.legend(labels=["GP fit"])

    plt.xlabel("x"), plt.ylabel("f")

    # Plot training points if included
    if training_points is not None:
        X_, Y_ = training_points
        plt.plot(X_, Y_, "kx", mew=2)
        plt.legend(labels=["GP fit", "sample points"])
################################################################################


def plot_dgp(X, m, C, fm, fC, training_points=None, full=True):
    """ Plotting utility to plot a GP fit with 95% confidence interval """
    # Plot 95% confidence interval
    plt.fill_between(X[:, 0],
                     m[:, 0] - 1.96*np.sqrt(np.diag(C)),
                     m[:, 0] + 1.96*np.sqrt(np.diag(C)),
                     alpha=0.5)
    # Plot GP mean and initial training points
    plt.plot(X, m, "-")
    if full:
        # Plot 95% confidence interval
        plt.fill_between(X[:, 0],
                         fm[:, 0] - 1.96*np.sqrt(np.diag(fC)),
                         fm[:, 0] + 1.96*np.sqrt(np.diag(fC)),
                         alpha=0.5)
        # Plot GP mean and initial training points
        plt.plot(X, fm, "--")
        plt.legend(labels=["Distributed GP fit", "Full GP fit"])
        plt.xlabel("x"), plt.ylabel("f")

    # Plot training points if included
    if training_points is not None:
        X_, Y_ = training_points
        plt.plot(X_, Y_, "kx", mew=2)
        plt.legend(labels=["Distributed GP fit",
                   "Full GP fit", "sample points"])

# ##############################################################################
# Returns a single sample from a multivariate Gaussian with mean and cov.
# ##############################################################################


def multivariateGaussianDraw(mean, cov, ndraws = 1):
    
    #samples = np.zeros((mean.shape[0], ))  # This is only a placeholder
    # Task 2:
    # TODO: Implement a draw from a multivariate Gaussian here
    samples = np.random.multivariate_normal(mean, cov, ndraws)
    # Return drawn sample
    if ndraws == 1:
        return samples.reshape(-1,)
    else:
        return samples

# ##############################################################################
# RadialBasisFunction for the kernel function
# k(x,x') = s2_f*exp(-norm(x,x')^2/(2l^2)). If s2_n is provided, then s2_n is
# added to the elements along the main diagonal, and the kernel function is for
# the distribution of y,y* not f, f*.
# ##############################################################################


class RadialBasisFunction():
    def __init__(self, params):
        self.ln_sigma_f = params[0]
        self.ln_length_scale = params[1]
        self.ln_sigma_n = params[2]

        self.sigma2_f = np.exp(2*self.ln_sigma_f)
        self.sigma2_n = np.exp(2*self.ln_sigma_n)
        self.length_scale = np.exp(self.ln_length_scale)

    def setParams(self, params):
        # params =  params.flatten()
        self.ln_sigma_f = params[0]
        self.ln_length_scale = params[1]
        self.ln_sigma_n = params[2]

        self.sigma2_f = np.exp(2*self.ln_sigma_f)
        self.sigma2_n = np.exp(2*self.ln_sigma_n)
        self.length_scale = np.exp(self.ln_length_scale)

    def getParams(self):
        return np.array([self.ln_sigma_f, self.ln_length_scale, self.ln_sigma_n])

    def getParamsExp(self):
        return np.array([self.sigma2_f, self.length_scale, self.sigma2_n])

    # ##########################################################################
    # covMatrix computes the covariance matrix for the provided matrix X using
    # the RBF. If two matrices are provided, for a training set and a test set,
    # then covMatrix computes the covariance matrix between all inputs in the
    # training and test set.
    # ##########################################################################
    def covMatrix(self, X, Xa=None):
        if Xa is not None:
            X_aug = np.zeros((X.shape[0]+Xa.shape[0], X.shape[1]))
            X_aug[:X.shape[0], :X.shape[1]] = X
            X_aug[X.shape[0]:, :X.shape[1]] = Xa
            X = X_aug

        n = X.shape[0]
        covMat = np.zeros((n, n))

        # Task 1:
        # TODO: Implement the covariance matrix here
        for i in range(n):
            for j in range(i, n):
                covMat[i, j] = self.sigma2_f * \
                    np.exp(-(np.linalg.norm(X[i] - X[j])
                             ** 2)/(2*self.length_scale**2))
        #
        covMat += np.triu(covMat, 1).T
        # If additive Gaussian noise is provided, this adds the sigma2_n along
        # the main diagonal. So the covariance matrix will be for [y y*]. If
        # you want [y f*], simply subtract the noise from the lower right
        # quadrant.
        if self.sigma2_n is not None:
            covMat += self.sigma2_n*np.identity(n)

        # Return computed covariance matrix
        return covMat


class GaussianProcessRegression():
    def __init__(self, X, y, k):
        self.X = X
        self.n = X.shape[0]
        self.y = y
        self.k = k
        self.K = self.KMat(self.X)
        self.L = np.linalg.cholesky(self.K)

    # ##########################################################################
    # Recomputes the covariance matrix and the inverse covariance
    # matrix when new hyperparameters are provided.
    # ##########################################################################
    def KMat(self, X, params=None):
        if params is not None:
            self.k.setParams(params)
        K = self.k.covMatrix(X)
        self.K = K
        self.L = np.linalg.cholesky(self.K)
        return K

    # ##########################################################################
    # Computes the posterior mean of the Gaussian process regression and the
    # covariance for a set of test points.
    # NOTE: This should return predictions using the 'clean' (not noisy) covariance
    # ##########################################################################
    def predict(self, Xa):
        mean_fa = np.zeros((Xa.shape[0], 1))
        cov_fa = np.zeros((Xa.shape[0], Xa.shape[0]))
        # Task 3:
        # TODO: compute the mean and covariance of the prediction
        # Covariance between training sample points (- Gaussian noise)
#         Kxx = self.K # - self.k.sigma2_f* np.identity(self.n)
        # covmatrix of X, Xa
        covxxa = self.k.covMatrix(self.X, Xa)
        Kxx = covxxa[:self.n, :self.n]
        # Covariance between training and test points
        # extracted from covxxa
        Ksx = covxxa[self.n:, :self.n]
        # Covariance between test points
        # extracted from covxxa
        Kss = covxxa[self.n:, self.n:] - \
            self.k.sigma2_n * np.identity(Xa.shape[0])
#         Kss = self.k.covMatrix(Xa)
        # The mean of the GP fit (note that @ is matrix multiplcation: A @ B is equivalent to np.matmul(A,B))
        mean_fa = Ksx @ np.linalg.solve(
            self.L.T, np.linalg.solve(self.L, self.y))
        # The covariance matrix of the GP fit
        cov_fa = Kss - \
            Ksx @ np.linalg.solve(self.L.T, np.linalg.solve(self.L, Ksx.T))
        # Return the mean and covariance
        return mean_fa.flatten(), cov_fa + self.k.sigma2_n * np.identity(cov_fa.shape[0])

    # ##########################################################################
    # Return negative log marginal likelihood of training set. Needs to be
    # negative since the optimiser only minimises.
    # ##########################################################################
    def logMarginalLikelihood(self, params=None):
        if params is not None:
            self.KMat(self.X, params)
        # Task 4:
        # TODO: Calculate the log marginal likelihood ( mll ) of self.y
        logdetK = 2*np.sum(np.log(np.diag(self.L)))
        term1 = 0.5 * (self.y.T @ np.linalg.solve(self.L.T,
                                                  np.linalg.solve(self.L, self.y)))
        term2 = 0.5 * logdetK
        t1 = 0
        try:
            t1 = term1[0][0]
        except:
            t1 = term1
        mll = t1 + term2 + (self.n/2.)*np.log(2*np.pi)
        # Return mll
        return mll

    # ##########################################################################
    # Computes the gradients of the negative log marginal likelihood wrt each
    # hyperparameter.
    # ##########################################################################
    def gradLogMarginalLikelihood(self, params=None):
        if params is not None:
            K = self.KMat(self.X, params)
        grad_ln_sigma_f = grad_ln_length_scale = grad_ln_sigma_n = 0
        # Task 5:
        # TODO: calculate the gradients of the negative log marginal likelihood
        # wrt. the hyperparameters
        a = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y))
        dK1 = np.zeros((self.n, self.n))
        dK2 = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                norm_squared = np.linalg.norm(self.X[i] - self.X[j])**2
                expfactor = np.exp(-norm_squared/(2*self.k.length_scale**2))
                dK1[i, j] = 2 * self.k.sigma2_f * expfactor
                dK2[i, j] = self.k.sigma2_f * \
                    (norm_squared/(self.k.length_scale**2)) * expfactor
        # gradlnsigmaf
        term1 = a.T @ (dK1 @ a)
        term2 = np.linalg.solve(self.L.T, np.linalg.solve(self.L, dK1))
        t1 = 0
        try:
            t1 = term1[0][0]
        except:
            t1 = term1
        grad_ln_sigma_f = -0.5 * t1 + 0.5 * np.trace(term2)

        # gradlnlengthscale
        term1 = a.T @ (dK2 @ a)
        term2 = np.linalg.solve(self.K, dK2)
        grad_ln_length_scale = -0.5 * \
            term1.flatten()[0] + 0.5 * np.trace(term2)

        # gradlnsigman
        dK3 = 2 * self.k.sigma2_n * np.identity(self.n)
        term1 = a.T @ (dK3 @ a)
        term2 = np.linalg.solve(self.K, dK3)
        grad_ln_sigma_n = -0.5 * term1.flatten()[0] + 0.5 * np.trace(term2)
        # Combine gradients
        gradients = np.array(
            [grad_ln_sigma_f, grad_ln_length_scale, grad_ln_sigma_n])
        # Return the gradients
        return gradients

    # ##########################################################################
    # Computes the mean squared error between two input vectors.
    # ##########################################################################
    def mse(self, ya, fbar):
        mse = 0
        # Task 7:
        # TODO: Implement the MSE between ya and fbar
        mse = np.sum((ya.flatten() - fbar)**2)/ya.shape[0]
        # Return mse()
        return mse

    # ##########################################################################
    # Computes the mean standardised log loss.
    # ##########################################################################
    def msll(self, ya, fbar, cov):
        msll = 0
        # Task 7:
        # TODO: Implement MSLL of the prediction fbar, cov given the target ya
        n = ya.shape[0]
        ya = ya.flatten()
        sigma2xs = np.diag(cov) + self.k.sigma2_n
        for i in range(n):
            msll += 0.5 * \
                np.log(
                    2*np.pi*sigma2xs[i]) + np.linalg.norm((ya[i] - fbar[i]))**2/(2*sigma2xs[i])
        msll = msll/n
        return msll

    # ##########################################################################
    # Minimises the negative log marginal likelihood on the training set to find
    # the optimal hyperparameters using BFGS.
    # ##########################################################################
    def optimize(self, params, disp=True):
        res = minimize(self.logMarginalLikelihood, params, method='BFGS',
                       jac=self.gradLogMarginalLikelihood, options={'disp': disp})
        return res.x
