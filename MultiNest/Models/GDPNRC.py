import sys
import numpy as np
from numba import jit

from scipy.special import hyp2f1
import scipy.integrate as integrate

@jit
def Support(params):
    # if params[0] <= 0 : return False
    if params[0] <= 0 : return False
    if params[2] >= 2 : return False
    return True

@jit
def Density(r,params,Rmax):
    a  = params[0]
    b  = params[1]
    g  = params[2]
    v1 = (r**(-g))*(1.0 + r**(1./a))**(-a*(b - g)) 
    w  = a*(b-g)
    x  = -a*(g-2.)
    y  = 1. - (a*(g-2.))
    z  = -((1.0/Rmax)**(-1./a))
    v2 = -(((Rmax**(-1./a))**(a*(g-2.)))/(g-2.))*hyp2f1(w,x,y,z)
    return v1/v2


def Number(r,params,Rmax):
    Num = np.vectorize(lambda y: integrate.quad(lambda x:Density(x,params,Rmax)*x,1e-5,y,
                epsabs=1.49e-03, epsrel=1.49e-03,limit=1000)[0])
    return Num(r)

@jit
def logLikeStar(p,r,params,Rmax):
    return np.log((p*r*Density(r,params,Rmax)) + (1.-p)*LikeField(r,Rmax))

@jit
def LikeField(r,rm):
    return 2.*r/rm**2

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
        #------- Uniform Priors -------
        for i in range(ndim):
            params[i] = (params[i])*(self.t[i,1]-self.t[i,0])+self.t[i,0]

    def LogLike(self,params,ndim,nparams):
         #----- Checks if parameters' values are in the ranges
        if not Support(params) : 
            return -1e50

        # ----- Computes Likelihoods ---------
        llike  = np.sum(map(lambda w,x:logLikeStar(w,x,params,self.Rmax),self.pro,self.radii))
        # print(llike)
        return llike



