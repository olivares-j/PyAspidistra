import sys
import numpy as np
from numba import jit
from scipy.stats import norm

D2R   = np.pi/180.
cntr  = [56.65,24.13]

@jit
def Support(params):
    e  = params[2]
    rc = params[4]
    g  = params[5]
    if rc <= 0.0 : return False
    if g  <= 2.0 : return False
    if e < 0.0 or e >= 1.0 : return False
    return True

@jit
def Number(r,params,Rmax):
    return (CDF(r,params[4],params[5])/CDF(Rmax,params[4],params[5]))

@jit
def Density(r,params,Rmax):
    rc = params[4]
    g  = params[5]
    x  = 1+(r/rc)**2
    res= ((g-2)*(x**(-g/2)))/(rc**2)
    return res/CDF(Rmax,rc,g)

@jit
def CDF(r,rc,g): 
    return 1 - (rc**(g-2))*(rc**2 + r**2)**((2-g)/2)

# @jit
# def logLikeStar(x,params,Rmax,sg):
#     return np.log(x[0]*(x[1]*Density(x[1],params,Rmax))*LikePA(x[2],x[3],sg) + (1.-x[0])*LikeField(x[1],Rmax))

@jit
def logLikeStar(x,params,Rmax):
    return np.log(x[0]*(x[1]*Density(x[1],params,Rmax)*LikePA(x[2],x[3],0.01)) + (1.-x[0])*LikeField(x[1],Rmax))


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
    def __init__(self,cdts,Rmax,trans_lim,Dist):
        """
        Constructor of the logposteriorModule
        """
        self.cdts       = cdts
        self.Rmax       = Rmax
        self.t          = trans_lim
        self.Dist       = Dist
        print("Module Initialized")

    def Priors(self,params, ndim, nparams):
        #---------- Uniform Priors ------
        for i in range(ndim):
            params[i] = (params[i])*(self.t[i,1]-self.t[i,0])+self.t[i,0]

    def LogLike(self,params,ndim,nparams):
        #----- Checks if parameters' values are in the ranges
        if not Support(params):
            return -1e50

        a0,d0,epsi,delta   = params[0],params[1],params[2],params[3]
        #============== Obtains radii and pa ===================
        # Equation 2 from Kacharov 2014. Since we also infer the centre positon x0=y0=0
        x     = np.sin((self.cdts[:,0]-(cntr[0]-a0))*D2R)*np.cos(self.cdts[:,1]*D2R)
        y     = (np.cos((cntr[1]-d0)*D2R)*np.sin(self.cdts[:,1]*D2R)-
                np.sin((cntr[1]-d0)*D2R)*np.cos(self.cdts[:,1]*D2R)*np.cos((self.cdts[:,0]-(cntr[0]-a0))*D2R))

        x_new = x*np.sin(delta) + y*np.cos(delta)
        y_new = x*np.cos(delta) - y*np.sin(delta)

        radii = (np.sqrt((x_new*(1.0-epsi))**2 + y_new**2)/(1.0-epsi))*self.Dist

        # radii = np.arccos(np.sin(cntr[1]*D2R)*np.sin(self.cdts[:,1]*D2R)+
        #         np.cos(cntr[1]*D2R)*np.cos(self.cdts[:,1]*D2R)*
        #         np.cos((cntr[0]-self.cdts[:,0])*D2R))*self.Dist + 1e-20
        # theta = np.arctan2(np.sin((self.cdts[:,0]-cntr[0])*D2R),
        #                  np.cos(cntr[1]*D2R)*np.tan(self.cdts[:,1]*D2R)-
        #                  np.sin(cntr[1]*D2R)*np.cos((self.cdts[:,0]-cntr[0])*D2R))
        theta   = np.arctan2(x_new,y_new)
        theta   = (theta + 2*np.pi)%(2*np.pi)
        # print(min(theta),max(theta))
        # idx   = np.where(theta  <= 0)[0]
        # theta[idx] = theta[idx] + 2*np.pi
        # theta = theta + phi
        # idx   = np.where(theta  <= 0)[0]
        # theta[idx] = theta[idx] + 2*np.pi
        # idx   = np.where(theta  > 2*np.pi)[0]
        # theta[idx] = theta[idx] - 2*np.pi

        idx   = np.argsort(theta)
        cdf   = (1./len(theta))*np.arange(len(theta))
        data  = np.vstack([self.cdts[:,2][idx],radii[idx],cdf,theta[idx]]).T

        #------ Computes Likelihood -----------
        llike  = np.sum(map(lambda x:logLikeStar(x,params,self.Rmax),data))
        # print(llike)
        return llike



