import sys
import numpy as np
from numba import jit
from scipy.stats import dirichlet,halfcauchy,uniform,norm
from scipy.special import erf
from time import time

D2R   = np.pi/180.

@jit
def Support(params):
    rc = params[2]
    if rc <= 0.0 : return False
    return True

@jit
def Number(r,params,Rmax):
    rc = params[2]
    return len(r)*(CDF(r,rc)/CDF(Rmax,rc))

@jit
def Density(r,params,Rmax):
    rc = params[2]
    x = (r/rc)**2
    z = np.exp(-x/2)/(rc**2)
    return z/CDF(Rmax,rc)

@jit
def CDF(r,rc):
    x  = (r/rc)**2
    frac = (1-np.exp(-x/2))
    return frac

@jit
def LLikeRadius(r,rc,Rmax):
    x = (r/rc)**2
    z = np.exp(-x/2)/(rc**2)
    return np.log(z/CDF(Rmax,rc))

@jit
def LLikePA(cdf,theta,sg):
    # print(norm.logpdf(cdf,loc=theta/np.pi,scale=sg))
    return norm.logpdf(theta/(2.*np.pi),loc=cdf,scale=sg)

class Module:
    """
    Chain for computing the likelihood 
    """
    def __init__(self,cdts,pro,Rmax,hyper,trans_lim,Dist):
        """
        Constructor of the logposteriorModule
        """
        self.cdts       = cdts
        self.pro        = pro
        self.Rmax       = Rmax
        self.hyper      = hyper
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
        sg    = params[3]

        #----- Checks if parameters' values are in the ranges
        if not Support(params):
            return -1e50
        #============== Obtains radii and pa ===================

        radii = np.arccos(np.sin(cntr[1]*D2R)*np.sin(self.cdts[:,1]*D2R)+
                np.cos(cntr[1]*D2R)*np.cos(self.cdts[:,1]*D2R)*
                np.cos((cntr[0]-self.cdts[:,0])*D2R))*self.Dist

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
        theta = theta[idx]

        #=======================================================
        #------ Computes Likelihood -----------
        llike_pro = np.sum(np.log(self.pro))
        llike_the = np.sum(map(lambda x,y:LLikePA(x,y,self.Rmax,sg),cdf,theta))
        # llike_r   = np.sum(map(lambda x:LLikeRadius(x,params[3],self.Rmax),radii))
        llike     = llike_pro + llike_the # +llike_r
        # print(llike)
        return llike



