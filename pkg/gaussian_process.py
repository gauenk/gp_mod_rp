import numpy as np
import numpy.linalg as npl
import numpy.random as npr

import scipy as sc
import scipy.linalg as scl


class GP():

    def __init__(self,x_test,x_data,y_data,kernel_type="rbf",alpha=1e-5,nsim=1):
        self.x_test = x_test.reshape(-1,1)
        self.x_data = x_data.reshape(-1,1)
        self.y_data = y_data.reshape(-1,1)
        self.ntest = len(x_test)
        self.ndata = len(x_data)
        self.n = 0
        self.update_n()
        self.alpha = alpha
        self.nsim = nsim
        self.kernel_type = kernel_type
        self.K_ss = self.kernel(self.x_test,self.x_test)
        self.K_xx = self.kernel(self.x_data,self.x_data)
        self.K_sx = self.kernel(self.x_test,self.x_data)
        self._posterior_means = None
        self._posterior_sigma = None
        print(x_data.shape,y_data.shape)

    def update_test(self,x_test):
        self.x_test = x_test.reshape(-1,1)
        self.K_ss = self.kernel(x_test,x_test)
        self.K_sx = self.kernel(x_test,self.x_data)
        self.ntest = len(x_test)
        self.update_n()

    def update_data(self,x_data,y_data):
        self.x_data = x_data.reshape(-1,1)
        self.y_data = y_data.reshape(-1,1)
        print(x_data.shape,y_data.shape)
        self.K_xx = self.kernel(x_data,x_data)
        self.K_sx = self.kernel(self.x_test,x_data)
        self.ndata = len(x_data)
        self.update_n()

    def update_n(self):
        self.n = self.ndata + self.ntest
        
    def sample_posterior(self,nsim=None,update=False):
        if nsim is None:
           nsim = self.nsim 
        # unpack
        n = self.n
        ntest = self.ntest
        ndata = self.ndata
        alpha = self.alpha
        x_test = self.x_test
        x_data = self.x_data
        K_sx = self.K_sx
        K_ss = self.K_ss
        K_xx = self.K_xx
        y_data = self.y_data

        # compute posterior info
        if self._posterior_sigma is None or update is True:
            K_xx_hat = K_xx + alpha * np.eye(ndata)
            eigs = npl.eig(K_xx_hat)[0]
            solved = scl.solve(K_xx_hat,K_sx,assume_a='pos').T
            means = np.squeeze(solved @ y_data)
            self._posterior_means = means
            sigma = K_ss + alpha * np.eye(ntest)  - solved @ K_sx
            self._posterior_sigma = sigma
        means = self._posterior_means
        sigma = self._posterior_sigma

        # lets sample!
        f_post = npr.multivariate_normal(mean=means,
                                         cov=sigma,
                                         size=nsim).T
        return f_post
        
    def sample_prior(self,nsim=None):
        if nsim is None:
           nsim = self.nsim 
        ntest = self.ntest
        K_ss = self.K_ss

        # let's sample!
        means = np.zeros(ntest)
        sigma = K_ss
        f_prior = npr.multivariate_normal(mean=np.zeros(ntest),cov=sigma,size=nsim).T
        return f_prior
    
    def sprior(self,nsim=None):
        return self.sample_prior(nsim=nsim)

    def spost(self,nsim=None):
        return self.sample_posterior(nsim=nsim)

    @property
    def posterior_stddev(self):
        return np.sqrt(np.diag(self._posterior_sigma))

    @property
    def posterior_sigma(self):
        return self._posterior_sigma

    @property
    def psigma(self):
        return self._posterior_sigma

    @property
    def posterior_means(self):
        return self._posterior_means

    @property
    def pmeans(self):
        return self._posterior_sigma

    def kernel(self,a,b,param=0.1,sigma=1,gamma=1,l=2):
        # efficient computation of squared distance
        # (n,1)                         (n,m)              (1,m)
        #sqdist = np.sum(a**2,1).reshape(-1,1) - 2*np.dot(a,b.T) + np.sum(b**2,1)
        # cosd = np.abs(np.cos(sqdist)+4*np.eye(n,m))
        # these are row matrices; hence the weird tranposes

        a = a.reshape(-1,1)
        b = b.reshape(-1,1)
        sqdist = np.sum(b**2,1).reshape(-1,1) - 2* b @ a.T + np.sum(a.T**2,1)
        n,m = sqdist.shape
        if self.kernel_type == 'rbf':
            rbf = param * np.exp(-.5 * (1/param) * sqdist)
            return rbf
        elif self.kernel_type == 'rqk':
            rqk = sigma * (1  + 1/(2 * gamma * l**2) * sqdist)**(-gamma)
            return rqk
        else:
            raise ValueError("Kernel Type [{}] is unknown".format(self.kernel_type))
        

"""
This class is a wrapper for sampled grid times and values,
improving code readablility.
"""

class SampledGrid():
    def __init__(self,times=None,vals=None):
        self.times = []
        self.vals = []
        if times is not None:
            self.times = times
        if vals is not None:
            self.vals = vals
    @property
    def v(self):
        return np.array(self.vals).reshape(-1,1)
    @property
    def t(self):
        return np.array(self.times).reshape(-1,1)

