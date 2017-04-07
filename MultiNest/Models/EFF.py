import sys
import numpy as np
from numba import jit
from scipy.stats import dirichlet
from scipy.stats import halfcauchy

@jit
def Support(params):
    rc = params[0]
    g  = params[1] 
    if rc <= 0 : return False
    if g <= 2 : return False
    return True

@jit
def Number(r,params,Rmax):
    return (CDF(r,params)/CDF(Rmax,params))

@jit
def CDF(r,params):
    rc = params[0]
    g  = params[1] 
    return 1 - ((rc**(g-2))*(rc**2 + r**2)**((2-g)/2))

@jit
def logLikeStar(r,p,params,Rmax):
    return np.log(p*LikeRadious(r,params,Rmax) + (1.-p)*LikeField(r,Rmax))

@jit
def LikeRadious(r,params,Rmax):
    rc = params[0]
    g  = params[1]
    x  = 1+(r/rc)**2
    res= r*(((g-2)*(x**(-g/2)))/(rc**2))
    return res/CDF(Rmax,params)

@jit
def LikeField(r,rm):
    return 2.*r/(rm**2)

@jit
def Density(r,params,Rmax):
    rc = params[0]
    g  = params[1]
    x  = 1+(r/rc)**2
    res= ((g-2)*(x**(-g/2)))/(rc**2)
    return res/CDF(Rmax,params)

class Module:
    """
    Chain for computing the likelihood 
    """
    def __init__(self,radii,pro,Rmax,trans_lim):
        """
        Constructor of the logposteriorModule
        """
        self.radii      = radii
        self.pro        = pro
        self.Rmax       = Rmax
        self.t          = trans_lim
        print("Module Initialized")

    def Priors(self,params, ndim, nparams):
        #-------- Uniform Priors -------
        for i in range(ndim):
            params[i] = (params[i])*(self.t[i,1]-self.t[i,0])+self.t[i,0]


    def LogLike(self,params,ndim,nparams):
        #----- Checks if parameters' values are in the ranges
        if not Support(params):
            return -1e50
        # ----- Computes Likelihoods ---------
        llike  = np.sum(map(lambda x,y:logLikeStar(x,y,params,self.Rmax),self.radii,self.pro))
        # print(llike)
        return llike



