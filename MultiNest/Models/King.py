import sys
import numpy as np
from numba import jit
import scipy.integrate as integrate

@jit
def Support(params):
    rc = params[0]
    rt = params[1]
    if rc <= 0 : return False
    if rt <= rc : return False
    return True

@jit
def CDF(r,params):
    rc = params[0]
    rt = params[1]
    if r > rt:
        return 1
    w = r**2 + rc**2
    x = 1 + (rt/rc)**2
    y = 1 + (r/rc)**2
    z = rc**2 + rt**2
    a = (r**2)/z +  4*(rc-np.sqrt(w))/np.sqrt(z) + np.log(w) -2*np.log(rc)
    b = -4 + (rt**2)/z + 4*rc/np.sqrt(z)         + np.log(z) -2*np.log(rc)
    NK  = a/b
    return NK


def Number(r,params,Rmax):
    Num = np.vectorize(lambda y: integrate.quad(lambda x:Density(x,params,Rmax)*x,1e-5,y,
                epsabs=1.49e-03, epsrel=1.49e-03,limit=1000)[0])
    return Num(r)

@jit
def logLikeStar(p,r,params,Rmax):
    return np.log(p*LikeRadious(r,params,Rmax)+ (1.-p)*LikeField(r,Rmax))

@jit
def LikeKing(r,rc,rt):
    x = 1 + (r/rc)**2
    y = 1 + (rt/rc)**2
    z = rc**2 + rt**2
    k   = 2.*((x**(-0.5))-(y**-0.5))**2
    norm= (rc**2)*(-4 + (rt**2)/z + 4*rc/np.sqrt(z) + np.log(z) -2*np.log(rc))
    lik = (r*k)/norm
    return lik

@jit
def LikeField(r,rm):
    return 2.*r/rm**2

def LikeRadious(r,params,Rmax):
    return np.piecewise(r,[r>params[1],r<=params[1]],
        [0.000001,lambda x: LikeKing(x,params[0],params[1])/CDF(Rmax,params)])

def Density(r,params,Rmax):
    return np.piecewise(r,[r>params[1],r<=params[1]],
        [0.000001,lambda x: LikeKing(x,params[0],params[1])/(x*CDF(Rmax,params))])

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
        #--------- Uniform Priors ---------
        for i in range(ndim):
            params[i] = (params[i])*(self.t[i,1]-self.t[i,0])+self.t[i,0]

    def LogLike(self,params,ndim,nparams):
        #----- Checks if parameters' values are in the ranges
        if not Support(params):
            return -1e50

        # ----- Computes Likelihoods ---------
        llike  = np.sum(map(lambda w,x:logLikeStar(w,x,params,self.Rmax),self.pro,self.radii))
        # print(llike)
        return llike







