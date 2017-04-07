import sys
import numpy as np
from numba import jit
from scipy.stats import dirichlet
from scipy.stats import halfcauchy
from scipy.special import gamma
import scipy.integrate as integrate
from scipy.interpolate import interp1d

a     = 1e-5

@jit
def Support(params):
    if params[0] <= 0 : return False
    if params[1] <= params[0] : return False
    if params[2] <= 0 : return False
    return True

def LikeMKing(r,params):
    rc = params[0]
    rt = params[1]
    a  = params[2]
    x  = (1.0 +  (r/rc)**(1./a))**-a
    y  = (1.0 + (rt/rc)**(1./a))**-a
    return r*(x-y)**2


def Number(r,params,Rmax):
    cte = integrate.quad(lambda x:LikeRadious(x,params),a,Rmax,epsabs=1.49e-03, epsrel=1.49e-03,limit=1000)[0]
    Num = np.vectorize(lambda y: integrate.quad(lambda x:LikeRadious(x,params)/cte,a,y,
                epsabs=1.49e-03, epsrel=1.49e-03,limit=1000)[0])
    return Num(r)


def Density(r,params,Rmax):
    cte = integrate.quad(lambda x:LikeRadious(x,params),a,Rmax,epsabs=1.49e-03, epsrel=1.49e-03,limit=1000)[0]
    Den = np.vectorize(lambda x:LikeRadious(x,params)/(x*cte))
    return Den(r)

@jit
def logLikeStar(p,r,params,Rmax,cte):
    return np.log(p*(LikeRadious(r,params)/cte) + (1.-p)*LikeField(r,Rmax))

def LikeRadious(r,params):
    return np.piecewise(r,[r>params[1],r<=params[1]],
        [0.0,lambda x: LikeMKing(x,params)])

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

        cte = integrate.quad(lambda x:LikeRadious(x,params),a,self.Rmax,epsabs=1.49e-03, epsrel=1.49e-03,limit=1000)[0]
        # ----- Computes Likelihoods ---------
        llike  = np.sum(map(lambda w,x:logLikeStar(w,x,params,self.Rmax,cte),self.pro,self.radii))
        return llike



