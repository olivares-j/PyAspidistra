import sys
import numpy as np
from numba import jit
from scipy.stats import dirichlet,halfcauchy,uniform,norm
from scipy.special import erf
from time import time

D2R   = np.pi/180.

@jit
def Support(params):
    if params[3] <= 0.0 : return False
    return True

@jit
def LikePA(cdf,theta,sg):
    return norm.logpdf(cdf,loc=theta/(2.*np.pi),scale=sg)

def Number(r,params,Rmax):
    return np.zeros_like(r)

def Density(r,params,Rmax):
    return np.zeros_like(r)

@jit
def LogLikeStar(p,cdf,theta,sg):
    res   = np.log(p)+LikePA(cdf,theta,sg) #+ (1.-p)*(1.0/(2.*np.pi))
    return res

class Module:
    """
    Chain for computing the likelihood 
    """
    def __init__(self,cdts,pro,Rmax,trans_lim,Dist):
        """
        Constructor of the logposteriorModule
        """
        self.cdts       = cdts
        self.pro        = pro
        self.Rmax       = Rmax
        self.t          = trans_lim
        self.Dist       = Dist
        self.sg         = 0.01
        print("Module Initialized")

    def Priors(self,params, ndim, nparams):
        #------- Uniform Priors ------
        for i in range(ndim):
            params[i] = (params[i])*(self.t[i,1]-self.t[i,0])+self.t[i,0]

    def LogLike(self,params,ndim,nparams):
        cntr  = params[:2]
        phi   = params[2]

        #----- Checks if parameters' values are in the ranges
        if not Support(params):
            return -1e50
        #============== Obtains radii and pa ===================
        theta = np.arctan2(np.sin((self.cdts[:,0]-cntr[0])*D2R),
                         np.cos(cntr[1]*D2R)*np.tan(self.cdts[:,1]*D2R)-
                         np.sin(cntr[1]*D2R)*np.cos((self.cdts[:,0]-cntr[0])*D2R))

        # print(min(theta),max(theta))
        idx   = np.where(theta  <= 0)[0]
        theta[idx] = theta[idx] + 2*np.pi
        theta = theta + phi
        idx   = np.where(theta  <= 0)[0]
        theta[idx] = theta[idx] + 2*np.pi
        idx   = np.where(theta  > 2*np.pi)[0]
        theta[idx] = theta[idx] - 2*np.pi


        idx   = np.argsort(theta)
        cdf   = (1./len(theta))*np.arange(len(theta))
        #=======================================================
        #------ Computes Likelihood -----------
        llike  = np.sum(map(lambda w,x,y:LogLikeStar(w,x,y,self.sg),self.pro[idx],cdf,theta[idx]))
        # print(llike)
        return llike



