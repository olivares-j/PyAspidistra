import sys
import numpy as np
from numba import jit
from scipy.stats import norm

D2R   = np.pi/180.

@jit
def Support(params):
    rc = params[3]
    g  = params[4]
    if rc <= 0.0 : return False
    if g  <= 2.0 : return False
    return True

@jit
def Number(r,params,Rmax):
    return (CDF(r,params[3],params[4])/CDF(Rmax,params[3],params[4]))

@jit
def Density(r,params,Rmax):
    rc = params[3]
    g  = params[4]
    x  = 1+(r/rc)**2
    res= ((g-2)*(x**(-g/2)))/(rc**2)
    return res/CDF(Rmax,rc,g)

@jit
def CDF(r,rc,g): 
    return 1 - (rc**(g-2))*(rc**2 + r**2)**((2-g)/2)

@jit
def logLikeStar(x,params,Rmax,sg):
    return np.log(x[0]*(x[1]*Density(x[1],params,Rmax))*LikePA(x[2],x[3],sg) + (1.-x[0])*LikeField(x[1],Rmax))

@jit
def LikePA(cdf,theta,sg):
    return norm.pdf(cdf,loc=theta/(2.*np.pi),scale=sg)+ 1e-100

@jit
def LikeField(r,rm):
    return 2.*r/rm**2

class Module:
    """
    Chain for computing the likelihood 
    """
    def __init__(self,cdts,pro,Rmax,trans_lim,Dist):
        """
        Constructor of the logposteriorModule
        """
        self.cdts       = cdts
        self.pro        = pro
        self.Rmax       = Rmax
        self.t          = trans_lim
        self.Dist       = Dist
        self.sg         = 0.01
        print("Module Initialized")

    def Priors(self,params, ndim, nparams):
        #---------- Uniform Priors ------
        for i in range(ndim):
            params[i] = (params[i])*(self.t[i,1]-self.t[i,0])+self.t[i,0]

    def LogLike(self,params,ndim,nparams):
        cntr  = params[1:3]
        phi   = params[0]

        #----- Checks if parameters' values are in the ranges
        if not Support(params):
            return -1e50
        #============== Obtains radii and pa ===================

        radii = np.arccos(np.sin(cntr[1]*D2R)*np.sin(self.cdts[:,1]*D2R)+
                np.cos(cntr[1]*D2R)*np.cos(self.cdts[:,1]*D2R)*
                np.cos((cntr[0]-self.cdts[:,0])*D2R))*self.Dist + 1e-20
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
        data  = np.vstack([self.pro[idx],radii[idx],cdf,theta[idx]]).T

        #------ Computes Likelihood -----------
        llike  = np.sum(map(lambda x:logLikeStar(x,params,self.Rmax,self.sg),data))
        # print(llike)
        return llike



