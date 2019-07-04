# coding: utf-8

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm import tqdm
###############################################################################
from sklearn.cluster import KMeans
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def gp_plot2D(Xtest_grid, X2, m2, C2=None):
    [Xi, Xj] = Xtest_grid
    # Setup plot environment
    fig = plt.figure(figsize=(20, 10))

    # Left plot shows mean of GP fit
    fig.add_subplot(221)

    # Plot mean surface
    plt.contour(Xi, Xj, m2.reshape(Xi.shape))
    # Show sample locations
    plt.plot(X2[:,0],X2[:,1],'o'), plt.axis("square")
    # Annotate plot
    plt.xlabel("$x_1$"), plt.ylabel("$x_2$")
    plt.title("Mean of GP fit"), plt.colorbar()

    # Right plot shows the variance of the GP
    fig.add_subplot(222)    
    # Plot variance surface
    if C2 is not None:
        plt.pcolor(Xi, Xj, np.diag(C2).reshape(Xi.shape))
    # Show sample locations
    plt.plot(X2[:,0],X2[:,1],'o'), plt.axis("square")
    # Annotate plot
    plt.xlabel("$x_1$"), plt.ylabel("$x_2$")
    plt.title("Variance of GP fit")
    if C2 is not None:
        plt.colorbar()
        
    fig.add_subplot(223)    
    im = plt.imshow(m2.reshape(Xi.shape), cmap=cm.RdBu)  # drawing the function
    # adding the Contour lines with labels
    cset = plt.contour(m2.reshape(Xi.shape), np.arange(-1, 1.5, 0.2), linewidths=2, cmap=cm.Set2)
    plt.clabel(cset, inline=True, fmt='%1.1f', fontsize=10)
    plt.colorbar(im)  # adding the colobar on the right
    # latex fashion title
    plt.title('colorplot')
    
    ax = fig.add_subplot(224, projection='3d')
       
    surf = ax.plot_surface(Xi, Xj, m2.reshape(Xi.shape), rstride=1, cstride=1, 
                          cmap=cm.RdBu,linewidth=0, antialiased=False)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)

def partition_data(X, y, n, random=False):
    # assuming X is NxD, y is Nx1
    N, D = X.shape
    assert len(y) == N

    indexes = dict()

    if not random:
        kmeans_idx = KMeans(n_clusters=n).fit_predict(X)
        for i in range(n):
            indexes[i] = (kmeans_idx == i)
    else:  # random partition
        list_ = np.arange(N)
        i = 0
        k = N//n
        r = N%n
        seed = np.random.randint(0, 1000)
        np.random.np.random.seed(seed)
        while i < n:
            if i < r:
                partition = np.random.choice(list_, k+1, replace=False)
                indexes[i] = partition
                list_ = list(set(list_) - set(partition))
            else:
                partition = np.random.choice(list_, k, replace=False)
                indexes[i] = partition
                list_ = list(set(list_) - set(partition))
            i += 1
    return indexes

def create_full_gp(X,y):
    params = np.random.randn(3,)
    kernel = RadialBasisFunction(params)
    fgp = GaussianProcessRegression(X, y, kernel)
    opt_params = fgp.optimize(params, True);
    fgp.KMat(X, opt_params)
    return fgp

def kl_divergence(mu1, s1, mu2, s2):
    
    kl = np.log(s2) - np.log(s1) - 1 + s1/s2
    kl += (mu1-mu2)**2/(s2**2)
    return 0.5 * kl
    
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

def plot_gp(X, m, C, fm, fC, sample_points=None, full=True):
    """ Plotting utility to plot a GP fit with 95% confidence interval """
    # Plot 95% confidence interval
    plt.fill_between(X[:, 0],
                     m[:, 0] - 1.96*np.sqrt(np.diag(C)),
                     m[:, 0] + 1.96*np.sqrt(np.diag(C)),
                     alpha=0.5)
    # Plot GP mean and initial training points
    plt.plot(X, m, "-")
    legend = ["Distributed GP fit"]           
    if full:
        # Plot 95% confidence interval
        plt.fill_between(X[:, 0],
                         fm[:, 0] - 1.96*np.sqrt(np.diag(fC)),
                         fm[:, 0] + 1.96*np.sqrt(np.diag(fC)),
                         alpha=0.5)
        # Plot GP mean and initial training points
        plt.plot(X, fm, "--")
        legend.append("Full GP fit")
        plt.xlabel("x"), plt.ylabel("f")

    # Plot training points if included
    if sample_points is not None:
        X_, Y_ = sample_points
        plt.plot(X_, Y_, "kx", mew=2)
        legend.append("Sample points")
    plt.legend(labels=legend)

###############################################################################
# Returns a single sample from a multivariate Gaussian with mean and cov.
# #############################################################################


def multivariateGaussianDraw(mean, cov, ndraws = 1):
    """Returns a single sample from a multivariate Gaussian with mean and cov."""
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
    """RadialBasisFunction for the kernel function
    # k(x,x') = s2_f*exp(-norm(x,x')^2/(2l^2)). If s2_n is provided, then s2_n is
    # added to the elements along the main diagonal, and the kernel function is for
    # the distribution of y,y* not f, f*."""
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
#         covMat = np.zeros((n, n))

        # Task 1:
        # TODO: Implement the covariance matrix here
        norm_squared = np.linalg.norm(X[:, None]-X, axis=2)**2
        covMat = self.sigma2_f * \
                np.exp(-norm_squared/(2*self.length_scale**2))

        # If additive Gaussian noise is provided, this adds the sigma2_n along
        # the main diagonal. So the covariance matrix will be for [y y*]. If
        # you want [y f*], simply subtract the noise from the lower right
        # quadrant.
        if self.sigma2_n is not None:
            covMat += self.sigma2_n*np.identity(n)

        # Return computed covariance matrix
        return covMat

    def crossCovMatrix(self, X, Xa):
        # TODO: Implement the covariance matrix here
        norm_squared = np.linalg.norm(X[:, None]-Xa, axis=2)**2
        covMat = self.sigma2_f * \
                np.exp(-norm_squared/(2*self.length_scale**2))
        # Return computed covariance matrix
        return covMat

class GaussianProcessRegression():
    def __init__(self, X, y, k):
        self.X = X
        self.n = X.shape[0]
        self.y = y.reshape(-1,1)
        self.k = k
        self.K = self.KMat(self.X)
        self.L = np.linalg.cholesky(self.K)

    # ##########################################################################
    # Recomputes the covariance matrix and the inverse covariance
    # matrix when new hyperparameters are provided.
    # ##########################################################################
    def KMat(self, X=None, params=None, donotreturn=False):
        if X is None:
            X = self.X
        if params is not None:
            self.k.setParams(params)
        K = self.k.covMatrix(X)
        self.K = K
        self.L = np.linalg.cholesky(self.K)
        if not donotreturn:
            return K
        pass

    # ##########################################################################
    # Computes the posterior mean of the Gaussian process regression and the
    # covariance for a set of test points.
    # NOTE: This should return predictions using the 'clean' (not noisy) covariance
    # ##########################################################################
    def predict(self, Xa, fullcov=True):
        mean_fa = np.zeros((Xa.shape[0], 1))
        cov_fa = np.zeros((Xa.shape[0], Xa.shape[0]))
        # Task 3:
        # TODO: compute the mean and covariance of the prediction
        # Covariance between training sample points (- Gaussian noise)
        Kxx = self.K 
        # Covariance between training and test points
        Ksx = self.k.crossCovMatrix(Xa, self.X)
        # Covariance between test points
        Kss = self.k.covMatrix(Xa)
        
        # The mean of the GP fit (note that @ is matrix multiplcation: A @ B is equivalent to np.matmul(A,B))
        mean_fa = Ksx @ np.linalg.solve(
            self.L.T, np.linalg.solve(self.L, self.y))
        # The covariance matrix of the GP fit
        cov_fa = Kss - \
            Ksx @ np.linalg.solve(self.L.T, np.linalg.solve(self.L, Ksx.T))
        # Return the mean and covariance
        out_cov = np.diag(cov_fa) + self.k.sigma2_n
        if fullcov:
            out_cov = cov_fa + self.k.sigma2_n * np.identity(cov_fa.shape[0])
        return mean_fa.flatten(), out_cov
    
    def predict2(self, Xa, fullcov=True):
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
        out_cov = np.diag(cov_fa) + self.k.sigma2_n
        if fullcov:
            out_cov = cov_fa + self.k.sigma2_n * np.identity(cov_fa.shape[0])
        return mean_fa.flatten(), out_cov
    
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
        #
        norm_squared = np.linalg.norm(self.X[:, None] - self.X, axis=2)**2
        expfactor = np.exp(-norm_squared/(2*self.k.length_scale**2))
        dK1 = 2 * self.k.sigma2_f * expfactor
        dK2 = self.k.sigma2_f * \
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
        sigma2xs = cov + self.k.sigma2_n
        msll = 0.5 * (np.log(2*np.pi*sigma2xs) + np.linalg.norm((ya.reshape(-1,1)- fbar.reshape(-1,1)), axis=1)**2/sigma2xs)
        msll = msll.mean()
        return msll

    # ##########################################################################
    # Minimises the negative log marginal likelihood on the training set to find
    # the optimal hyperparameters using BFGS.
    # ##########################################################################
    def optimize(self, params, disp=True):
        res = minimize(self.logMarginalLikelihood, params, method='BFGS',
                       jac=self.gradLogMarginalLikelihood, options={'disp': disp})
        return res.x
#########################################################################################################
import multiprocessing as mp
class DistributedGPRegression():
    def __init__(self, X, y, n, random_partition=False):
        self.n = n  # number of experts
        self.X = X
        self.y = y
        self.ksi = None
        self.random_partition = random_partition
        self.indexes = None
        self.Experts = self.create_experts()

    def new_expert(self, i):
        params = np.array([0, np.log(0.1 + 0.01*i), 0])
        kernel = RadialBasisFunction(params)
        Xi = self.X[self.indexes[i]]
        yi = self.y[self.indexes[i]]
        gp = GaussianProcessRegression(Xi, yi, kernel)
        return gp

    def create_experts(self):
        experts = []
        N, D = self.X.shape
        self.indexes = partition_data(
            self.X, self.y, self.n, self.random_partition)
        pool = mp.Pool(mp.cpu_count())

        experts = pool.map_async(self.new_expert, np.arange(self.n)).get()
        pool.close()
        pool.join()
        return experts

    def set_random_partition(self, val=True):
        self.random_partition = val
        self.Experts = self.create_experts()

    def logMl_i(self, i, params):
        return self.Experts[i].logMarginalLikelihood(params)

    def logMl(self, params=None):
        f = 0.
        for i in range(self.n):
            f += self.logMl_i(i, params)
        return f

    def gradLogMl_i(self, i, params):
        return self.Experts[i].gradLogMarginalLikelihood(params)

    def gradLogMl(self, params=None):
        gradf = 0
        for i in range(self.n):
            gradf += self.gradLogMl_i(i, params)
        return gradf

    # Optimize the sum of log-marginal likelihood
    def train(self, display=False, pretrained=None):
        if pretrained is None:
            params = np.random.randn(3,)
            res = minimize(self.logMl, params, method='BFGS',
                           jac=self.gradLogMl, options={'disp': display})
            opt_params = res.x
        else:
            opt_params = pretrained
        # update the experts
        for j in range(self.n):
            self.update_expert(j, opt_params)
    ###end train###

    def update_expert(self, i, opt_params):
        self.Experts[i].KMat(params=opt_params, donotreturn=True)
        return i

    def pred_expert_i(self, i, Xa, sqrt=False, fullcov=False):
        m_k, cov_k = self.Experts[i].predict(Xa,fullcov)
#         print('expert {} pred: m{},cov{}'.format(i, m_k.shape, cov_k.shape))
        if fullcov:
            assert is_symmetric(cov_k)
            assert is_psd(cov_k, cholesky_test=True)
        if sqrt:
            L_k = mat_sqrt(cov_k)
#             print('expert ',i,' computed!')
            return (i, m_k, L_k)
        else:
            #             print('expert ',i,' computed!')
            return (i, m_k, cov_k)
        # end pred expert i

    def experts_preds(self, Xa, pool, sqrt=False, fullcov=False):
        """returns the predictions m, Cov of the experts on data Xa
           for the Cov the square root will be returned if sqrt is True
        """
        preds = pool.starmap_async(
            self.pred_expert_i, [(i, Xa, sqrt, fullcov) for i in range(self.n)]).get()
        preds.sort(key=lambda x: x[0])
        pool.close()
        pool.join()
        return preds

    def gbarycenter(self, Xa, ksi='default'):
        N, D = Xa.shape
        pool = mp.Pool(mp.cpu_count())
        preds = self.experts_preds(Xa, pool)
#         print('all experts computed!')
        # computing the weights ksi
        if ksi is 'default':
            ksi = np.ones((self.n, N))/self.n

        elif ksi is 'opt':
            ksi = np.zeros((self.n, N))
            # prior covariance of matrix of Xa
            sigma2_ss = self.Experts[0].k.sigma2_f + self.Experts[0].k.sigma2_n
            for j, m_k, cov_k in preds:
                sigma2_k_s = cov_k #np.diag(Cov_k)
                ksi[j] = 0.5 * (np.log(sigma2_ss) - np.log(sigma2_k_s)) + 1e-8
            ksi /= ksi.sum(0)
            assert ((ksi.sum(0)-1) < 1e-7).all()
        elif ksi is 'kld':
            ksi = np.zeros((self.n, N))
            # prior covariance of matrix of Xa
            sigma2_ss = self.Experts[0].k.sigma2_f + self.Experts[0].k.sigma2_n
            for j, mu_k_s, cov_k in preds:
                sigma2_k_s = cov_k #np.diag(Cov_k)
                ksi[j] = kl_divergence(mu_k_s, sigma2_k_s, 0, sigma2_ss) + 1e-8
            ksi /= ksi.sum(0)
            assert ((ksi.sum(0)-1) < 1e-7).all()
        self.ksi = ksi
        # starting computing the wasserstein barycenter
        # mean m
        m = np.zeros(N)
        for j, m_k, cov_k in preds:
            m += ksi[j] * m_k
        # variance sigma2
        sigma2 = self.fixed_point_K(preds, N, ksi)
#         C = np.diag(sigma2)
        return m, sigma2

    def fixed_point_K(self, preds, N, ksi, eps=1e-6, verbose=False):
        # initialisations
        # sigma^2 squared
        sigma2_current = np.random.random(N)
        precision = eps + 1
        # number of iterations
        n_iter = 0
        if verbose:
            print('starting iterations...')
        while precision > eps and n_iter < 100:
            sigma = np.sqrt(sigma2_current)
            Sum = np.zeros(N)  # , dtype='complex128')
            for j, m_k, Cov_k in preds:
                sigma2_j = Cov_k# np.diag(Cov_k)
                sKs = sigma * sigma2_j * sigma
                Sum += ksi[j] * np.sqrt(sKs)
            # endfor
#             sigma2_new =  Sum
            sigma2_new = (1/sigma) * Sum**2 * (1/sigma)
            precision = np.linalg.norm(sigma2_new - sigma2_current)
            sigma2_current = sigma2_new
            n_iter += 1
            if verbose:
                print('n_iter: ', n_iter, ', precision: ', precision)
        # end while
        if verbose:
            print('final n_iter: ', n_iter, ', precision: ', precision)
        return sigma2_current

    def predict_poe(self, Xa):
        N, D = Xa.shape
        sigma_star = np.zeros(N)
        sum_star = np.zeros(N)
        pool = mp.Pool(mp.cpu_count())
        preds = self.experts_preds(Xa, pool)
        pool.close()
        pool.join()
        for k, m_k, cov_k in preds:
            # ?invert the diagonal of cov
            inv_diag = 1/cov_k #np.diag(cov_k)
            sigma_star = sigma_star + inv_diag
            sum_star = sum_star + inv_diag * m_k

        sigma_star = 1 / sigma_star
#         print("mean_sigma* = ", sigma_star.mean(),
#               ", std_sigma* = ", sigma_star.std())
#         cov_star = np.diag(sigma_star)
        m_star = sigma_star * sum_star
        return m_star.flatten(), sigma_star #cov_star

    def predict_gpoe(self, Xa, betas='default'):
        if betas is 'default':
            betas = np.ones(self.n)
        N, D = Xa.shape
        sigma_star = np.zeros(N)
        sum_star = np.zeros((N))
        pool = mp.Pool(mp.cpu_count())
        preds = self.experts_preds(Xa, pool)
#         print('all experts computed!')
        for k, m_k, cov_k in preds:
            inv_diag = 1/cov_k #np.diag(cov_k)
            sigma_star = sigma_star + betas[k]*inv_diag
            sum_star = sum_star + betas[k]*m_k * inv_diag

        sigma_star = 1 / sigma_star
#         print("mean_sigma* = ", sigma_star.mean(),
#               ", std_sigma* = ", sigma_star.std())
        m_star = sigma_star * sum_star
#         cov_star = np.diag(sigma_star)
        return m_star.flatten(), sigma_star #cov_star

    def predict_bcm(self, Xa):
        N, D = Xa.shape
        sigma_star = np.zeros(N)
        sum_star = np.zeros(N)
        pool = mp.Pool(mp.cpu_count())
        preds = self.experts_preds(Xa, pool)
#         print('all experts computed!')
        for k, m_k, cov_k in preds:
            # invert the diagonal of cov
            inv_diag = 1/cov_k#np.diag(cov_k)
            sigma_star = sigma_star + inv_diag
            sum_star = sum_star + m_k * inv_diag
            #
        sigma_prior = np.diag(self.Experts[0].k.covMatrix(Xa))
        sigma_star = sigma_star + (1-self.n)/sigma_prior
        sigma_star = 1/sigma_star
#         print("mean_sigma* = ", sigma_star.mean(),
#               ", std_sigma* = ", sigma_star.std())
        m_star = sigma_star * sum_star
#         cov_star = np.diag(sigma_star)
        return m_star, sigma_star#cov_star

    def predict_rbcm(self, Xa):
        N, D = Xa.shape
        sigma_star = np.zeros(N)
        sum_star = np.zeros(N)
        sigma_prior = np.diag(self.Experts[0].k.covMatrix(Xa))
        bconst = 0.5 * np.log(sigma_prior)
        sum_betas = 0
        pool = mp.Pool(mp.cpu_count())
        preds = self.experts_preds(Xa, pool)
#         print('all experts computed!')
        for k, m_k, cov_k in preds:
            # invert the diagonal of cov
            inv_diag = 1/cov_k #np.diag(cov_k)
            beta = bconst - 0.5 * np.log(cov_k) #np.diag(cov_k))
            sum_betas = sum_betas + beta
            sigma_star = sigma_star + beta * inv_diag
            sum_star = sum_star + beta * m_k * inv_diag
            #

        sigma_star = sigma_star + (1-sum_betas) * (1/sigma_prior)
        sigma_star = 1/sigma_star
#         print("mean_sigma* = ", sigma_star.mean(),
#               ", std_sigma* = ", sigma_star.std())
        m_star = sigma_star * sum_star
#         cov_star = np.diag(sigma_star)
        return m_star, sigma_star #cov_star

    # mean squared error
    def mse(self, ya, fbar):
        return self.Experts[0].mse(ya, fbar)
    # mean standardised log loss or  mean (negative) log predictive density

    def msll(self, ya, fbar, cov):
        return self.Experts[0].msll(ya, fbar, cov)
#################END CLASS###################################################