import sys
import numpy as np
from numba import jit
from scipy.stats import dirichlet,halfcauchy,uniform
from scipy.special import erf

@jit
def Support(rc):
    if rc <= 0 : return -np.inf

@jit
def Number(r,rc,Rmax):
    return (CDF(r,rc)/CDF(Rmax,rc))

@jit
def CDF(r,rc):
    x  = (r/rc)**2
    frac = (1-np.exp(-x/2))
    return frac

@jit
def logLikeStar(r,p,rc,Rmax):
    return np.log(p*LikeRadious(r,rc,Rmax) + (1.-p)*LikeField(r,Rmax))

@jit
def LikeRadious(r,rc,Rmax):
    x = (r/rc)**2
    z = r*(np.exp(-x/2)/(rc**2))
    return z/CDF(Rmax,rc)

@jit
def LikeField(r,rm):
    return 2.*r/rm**2

@jit
def Density(r,rc,Rmax):
    x = (r/rc)**2
    z = (np.exp(-x/2)/(rc**2))
    return z/CDF(Rmax,rc)



class Module:
    """
    Chain for computing the likelihood 
    """
    def __init__(self,radii,pro,Rmax,hyper,trans_lim):
        """
        Constructor of the logposteriorModule
        """
        self.radii      = radii
        self.pro        = pro
        self.Rmax       = Rmax
        self.hyper      = hyper
        self.t          = trans_lim
        print("Gaussian Initialized")

    def Priors(self,params, ndim, nparams):
        #-------- Uniform Priors -------------
        for i in range(ndim):
            params[i] = (params[i])*(self.t[i,1]-self.t[i,0])+self.t[i,0]

    def LogLike(self,params,ndim,nparams):
        rc  = params[0]
        #----- Checks if parameters' values are in the ranges
        supp = Support(rc)
        if supp == -np.inf : 
            return -1e50

        # ----- Computes Likelihoods ---------
        llike  = map(lambda x,y:logLikeStar(x,y,rc,self.Rmax),self.radii,self.pro)
        lpos   = np.sum(llike)
        return lpos



