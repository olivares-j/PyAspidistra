import sys
import numpy as np
from numba import jit
import scipy.stats as st
from Functions import Deg2pc,TruncSort

print "EFF NoCentre imported!"

@jit
def Support(rc,g):
    if rc <= 0.0 : return False
    if  g <= 2.0 : return False
    return True

@jit
def Number(r,params,Rm,Nstr):
    return Nstr*cdf(r,params,Rm)

@jit
def cdf(r,params,Rm):
    rc = params[0]
    g  = params[1]
    w  = r**2  + rc**2
    y  = Rm**2 + rc**2
    a  = rc**2 - (rc**g)*(w**(1.0-0.5*g))
    b  = (rc**2)*(1.0-(rc**(g-2.0))*(y**(1.0-0.5*g)))
    return a/b

@jit
def Kernel(r,rc,g):
    x  = 1.0+(r/rc)**2
    return x**(-0.5*g)

@jit
def LikeField(r,rm):
    return 2.0*r/(rm**2)

@jit
def Density(r,params,Rm):
    rc = params[0]
    g  = params[1]
    y = rc**2 + Rm**2
    a = (1.0/(g-2.0))*(1.0-(rc**(g-2.0))*(y**(1.0-0.5*g))) # Truncated at Rm
    # a = 1.0/(g-2.0)                                      #  No truncated
    k  = 1.0/(a*(rc**2.0))
    return k*Kernel(r,rc,g)

class Module:
    """
    Chain for computing the likelihood 
    """
    def __init__(self,cdts,Rcut,hyp,Dist,centre_init):
        """
        Constructor of the logposteriorModule
        """
        rad,thet        = Deg2pc(cdts,centre_init,Dist)
        c,rad,_,Rmax    = TruncSort(cdts,rad,thet,Rcut)
        self.pro        = c[:,2]
        self.rad        = rad
        self.Rmax       = Rmax
        self.Prior_0    = st.halfcauchy(loc=0,scale=hyp[0])
        self.Prior_1    = st.uniform(loc=0,scale=hyp[1])
        print "Module Initialized"

    def Priors(self,params, ndim, nparams):
        #-------- Uniform Priors -------
        params[0]  = self.Prior_0.ppf(params[0])
        params[1]  = self.Prior_1.ppf(params[1])


    def LogLike(self,params,ndim,nparams):
        rc = params[0]
        g  = params[1]
        #----- Checks if parameters' values are in the ranges
        if not Support(rc,g):
            return -1e50
        # Computes likelihood
        lf = (1.0-self.pro)*LikeField(self.rad,self.Rmax)
        lk = self.rad*(self.pro)*Kernel(self.rad,rc,g)


        # Normalisation constant
        # w = rc**2 +  r**2 
        y = rc**2 + self.Rmax**2
        a = (1.0/(g-2.0))*(1.0-(rc**(g-2.0))*(y**(1.0-0.5*g))) # Truncated at Rm
        # a = 1.0/(g-2.0)                                      #  No truncated
        k  = 1.0/(a*(rc**2.0))

        llike  = np.sum(np.log((k*lk + lf)))
        # print(llike)
        return llike



