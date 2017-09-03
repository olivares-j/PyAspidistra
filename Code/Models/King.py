import sys
import numpy as np
from numba import jit
import scipy.stats as st

@jit
def Support(rc,rt):
    if rc <= 0  : return False
    if rt <= rc : return False
    return True

@jit
def cdf(r,params,Rm):
    rc = params[0]
    rt = params[1]
    w = rc**2 +  r**2 
    y = rc**2 + Rm**2
    z = rc**2 + rt**2
    a = (r**2)/z  +  4*(rc-np.sqrt(w))/np.sqrt(z) + np.log(w) - 2*np.log(rc)
    b = (Rm**2)/z +  4*(rc-np.sqrt(y))/np.sqrt(z) + np.log(y) - 2*np.log(rc)
    return a/b

@jit
def Number(r,params,Rm,Nstr):
    # Rm must be less or equal to rt
    return Nstr*cdf(r,params,Rm)

@jit
def Kernel(r,rc,rt):
    x = 1 + (r/rc)**2
    y = 1 + (rt/rc)**2
    k = ((x**(-0.5))-(y**-0.5))**2
    return k

@jit
def LikeField(r,rm):
    return 2.*r/rm**2

def Density(r,params,Rmax):
    rc = params[0]
    rt = params[1]
    ker = Kernel(r,rc,rt)
    # In king's profile no objects is larger than tidal radius
    idBad = np.where(r > rt)[0]
    ker[idBad] = 0.0

    # Normalisation constant
    # w = rc**2 +  r**2 
    y = rc**2 + Rmax**2
    z = rc**2 + rt**2
    a = (Rmax**2)/z +  4*(rc-np.sqrt(y))/np.sqrt(z) + np.log(y) - 2*np.log(rc) # Truncated at Rm
    # a = (rt**2)/z - 4.0 +  4.0*(rc/np.sqrt(z)) + np.log(z) - 2*np.log(rc)  # Truncated at Rt (i.e. No truncated)
    k  = 2.0/(a*(rc**2.0))

    return k*ker

class Module:
    """
    Chain for computing the likelihood 
    """
    def __init__(self,cdts,Rmax,hyp,Dist):
        """
        Constructor of the logposteriorModule
        """
        self.pro        = cdts[:,2]
        self.rad        = cdts[:,3]
        self.Rmax       = Rmax
        # -------- Priors --------
        self.Prior_0    = st.halfcauchy(loc=0,scale=hyp[0])
        self.Prior_1    = st.halfcauchy(loc=0,scale=hyp[1])
        print("Module Initialized")

    def Priors(self,params, ndim, nparams):
        params[0]  = self.Prior_0.ppf(params[0])
        params[1]  = self.Prior_1.ppf(params[1])

    def LogLike(self,params,ndim,nparams):
        rc = params[0]
        rt = params[1]
        #----- Checks if parameters' values are in the ranges
        if not Support(rc,rt):
            return -1e50

        # Computes likelihood
        lf = (1.0-self.pro)*LikeField(self.rad,self.Rmax)
        lk = self.rad*(self.pro)*Kernel(self.rad,rc,rt)

        # In king's profile no objects is larger than tidal radius
        idBad = np.where(self.rad > rt)[0]
        lk[idBad] = 0.0

        # Normalisation constant
        # w = rc**2 +  r**2 
        y = rc**2 + self.Rmax**2
        z = rc**2 + rt**2
        a = (self.Rmax**2)/z +  4*(rc-np.sqrt(y))/np.sqrt(z) + np.log(y) - 2*np.log(rc) # Truncated at Rm
        # a = (rt**2)/z - 4.0 +  4.0*(rc/np.sqrt(z)) + np.log(z) - 2*np.log(rc)  # Truncated at Rt (i.e. No truncated)
        k  = 2.0/(a*(rc**2.0))

        llike  = np.sum(np.log((k*lk + lf)))
        # print(llike)
        return llike







