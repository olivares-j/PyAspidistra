import sys
import numpy as np
from numba import jit
from scipy.stats import norm

from scipy.special import gamma
import scipy.integrate as integrate
from scipy.interpolate import interp1d

D2R   = np.pi/180.

a     = 1e-5

@jit
def Support(params):
    rc  = params[3]
    rt  = params[4]
    a   = params[5]
    b   = params[6]
    if rc <= 0 : return False
    if rt <= rc : return False
    if a <= 0 : return False
    if b <= 0 : return False
    return True

def rho(r,params):
    rc = params[3]
    rt = params[4]
    a  = params[5]
    b  = params[6]
    x  = (1.0 +  (r/rc)**(1./a))**-a
    y  = (1.0 + (rt/rc)**(1./a))**-a
    return (x-y)**b


def Number(r,params,Rmax):
    Num = np.array(map(lambda y: integrate.quad(lambda x:Density(x,params,Rmax)*x,1e-5,y,
                epsabs=1.49e-03, epsrel=1.49e-03,limit=1000),r))

    return Num


def Density(r,params,Rmax):
    cte = integrate.quad(lambda x:rho(x,params)*x,a,Rmax,epsabs=1.49e-03, epsrel=1.49e-03,limit=1000)[0]
    return LikeRadious(r,params,cte)

def LikeRadious(r,params,cte):
    return np.piecewise(r,[r>params[4],r<=params[4]],[1e-10,lambda x: rho(x,params)/cte])

@jit
def LikePA(cdf,theta,sg):
    return norm.pdf(cdf,loc=theta/(2.*np.pi),scale=sg)+ 1e-100

@jit
def LogLikeStar(x,params,Rmax,sg,cte):
    return np.log(x[0]*(x[1]*LikeRadious(x[1],params,cte))*LikePA(x[2],x[3],sg) + (1.-x[0])*LikeField(x[1],Rmax))

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
        self.llike_p    = np.sum(np.log(pro))
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
                np.cos((cntr[0]-self.cdts[:,0])*D2R))*self.Dist + 1e-20 # avoids zeros
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

        # Normalisation constant
        cte = integrate.quad(lambda x:rho(x,params)*x,a,self.Rmax, epsabs=1.49e-03, epsrel=1.49e-03,limit=1000)[0]

        #------ Computes Likelihood -----------
        llike = np.sum(map(lambda x:LogLikeStar(x,params,self.Rmax,self.sg,cte),data))
        # print(llike)
        return llike



