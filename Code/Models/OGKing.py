import sys
import numpy as np
from numba import jit
import scipy.stats as st
import scipy.integrate as integrate
from Functions import Deg2pc,TruncSort


lo     = 1e-5

@jit
def Support(params):
    rc = params[0]
    rt = params[1]
    if rc <= 0 : return False
    if rt <= rc : return False
    return True

@jit
def Kernel(r,params):
    rc = params[0]
    rt = params[1]
    a  = 0.38
    b  = 1.15
    x  = (1.0 +  (r/rc)**(1./a))**-a
    y  = (1.0 + (rt/rc)**(1./a))**-a
    return (x-y)**b

def cdf(r,params,Rm):
    return NormCte(params,r)/NormCte(params,Rm)


def Number(r,params,Rm,Nstr):
    cte = NormCte(params,Rm)
    Num = np.vectorize(lambda y: integrate.quad(lambda x:x*Kernel(x,params)/cte,lo,y,
                epsabs=1.49e-03, epsrel=1.49e-03,limit=1000)[0])
    return Nstr*Num(r)


def Density(r,params,Rm):
    cte = NormCte(params,Rm)
    Den = np.vectorize(lambda x:Kernel(x,params)/cte)
    return Den(r)

def NormCte(params,Rm):
    if params[1] < Rm :
        up = params[1]
    else :
        up = Rm
    cte = integrate.quad(lambda x:x*Kernel(x,params),lo,up,epsabs=1.49e-03, epsrel=1.49e-03,limit=1000)[0]
    return cte

@jit
def LikeField(r,rm):
    return 2.*r/rm**2

class Module:
    """
    Chain for computing the likelihood 
    """
    def __init__(self,cdts,Rcut,hyp,Dist,centre):
        """
        Constructor of the logposteriorModule
        """
        rad,thet        = Deg2pc(cdts,centre,Dist)
        c,r,t,self.Rmax = TruncSort(cdts,rad,thet,Rcut)
        self.pro        = c[:,2]
        self.rad        = r
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
        if not Support(params) : 
            return -1e50

        # Computes likelihood
        lf = (1.0-self.pro)*LikeField(self.rad,self.Rmax)
        lk = self.rad*(self.pro)*Kernel(self.rad,params)

        # In king's profile no objects is larger than tidal radius
        idBad = np.where(self.rad > rt)[0]
        lk[idBad] = 0.0

        # Normalisation constant
        cte = NormCte(params,self.Rmax)

        k = 1.0/cte

        llike  = np.sum(np.log((k*lk + lf)))
        # print(llike)
        return llike



