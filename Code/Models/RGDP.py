import sys
import numpy as np
from numba import jit
import scipy.stats as st
from Functions import Deg2pc,TruncSort

from scipy.special import hyp2f1

@jit
def Support(rc,a,b):
    if rc <= 0 : return False
    if a  <= 0 : return False
    if b  <  0 : return False
    return True

@jit
def cdf(r,params,Rm):
    rc = params[0]
    a  = params[1]
    b  = params[2]

    # Normalisation constant
    x = -((rc/Rm)**(-1.0/a) + 0.0j) 
    y = -((r/rc)**(1.0/a)   + 0.0j)
    u = 2.0*a

    c = ((x**u)/u)*hyp2f1(u,a*b,1.0 + u,x)
    d = ((y**u)/u)*hyp2f1(u,a*b,1.0 + u,y)

    return np.abs(d)/np.abs(c)

@jit
def Number(r,params,Rm,Nstr):
    res = np.zeros_like(r)
    for i,s in enumerate(r):
        res[i] = cdf(r[i],params,Rm)
    return Nstr*res

@jit
def Kernel(r,rc,a,b):
    y = (1.0 + (r/rc)**(1.0/a))**(a*b)
    return 1.0/y

@jit
def Density(r,params,Rm):
    rc = params[0]
    a  = params[1]
    b  = params[2]

    # Normalisation constant
    x = -1.0*((rc/Rm)**(-1.0/a))
    c = np.abs((Rm**2)*hyp2f1(2.0*a,a*b,1 + 2.0*a,x))
    k  = 2.0/c

    return k*Kernel(r,rc,a,b)

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
        self.Prior_1    = st.halfcauchy(loc=0.01,scale=hyp[1])
        self.Prior_2    = st.halfcauchy(loc=0.01,scale=hyp[2])
        print("Module Initialized")

    def Priors(self,params, ndim, nparams):
        params[0]  = self.Prior_0.ppf(params[0])
        params[1]  = self.Prior_1.ppf(params[1])
        params[2]  = self.Prior_2.ppf(params[2])

    def LogLike(self,params,ndim,nparams):
        rc = params[0]
        a  = params[1]
        b  = params[2]
        g  = params[3]
         #----- Checks if parameters' values are in the ranges
        if not Support(rc,a,b) : 
            return -1e50

        # Computes likelihood
        lf = (1.0-self.pro)*LikeField(self.rad,self.Rmax)
        lk = self.rad*(self.pro)*Kernel(self.rad,rc,a,b)

        # Normalisation constant
        x = -1.0*((rc/self.Rmax)**(-1.0/a))
        c = np.abs((self.Rmax**2)*hyp2f1(2.0*a,a*b,1 + 2.0*a,x))
        k  = 2.0/c

        llike  = np.sum(np.log((k*lk + lf)))
        # print(llike)
        return llike


