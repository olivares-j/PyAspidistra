import sys
import numpy as np
from numba import jit
import scipy.stats as st
from Functions import Deg2pc,TruncSort
from pandas import cut, value_counts

print "EFF Centre imported!"

@jit
def Support(rc,g):
    if rc <= 0.0 : return False
    # if  g <= 2.0 : return False
    return True

@jit
def Number(r,params,Rm,Nstr):
    return Nstr*cdf(r,params,Rm)

@jit
def cdf(r,params,Rm):
    rc = params[2]
    g  = params[3]
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
    rc = params[2]
    g  = params[3]
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
        c,r,t,self.Rmax = TruncSort(cdts,rad,thet,Rcut)
        self.pro        = c[:,2]
        self.cdts       = c[:,:2]
        self.Dist       = Dist
        print "There are ", len(self.cdts)," observations."
        # sys.exit()
        #------------- poisson ----------------
        self.quadrants  = [0,np.pi/2.0,np.pi,3.0*np.pi/2.0,2.0*np.pi]
        self.poisson    = st.poisson(len(r)/4.0)
        #-------------- priors ----------------
        self.Prior_0    = st.norm(loc=centre_init[0],scale=hyp[0])
        self.Prior_1    = st.norm(loc=centre_init[1],scale=hyp[1])
        self.Prior_2    = st.halfcauchy(loc=0.01,scale=hyp[2])
        self.Prior_3    = st.truncexpon(b=hyp[3],loc=2.01,scale=hyp[4])
        print "Module Initialized"

    def Priors(self,params, ndim, nparams):
        params[0]  = self.Prior_0.ppf(params[0])
        params[1]  = self.Prior_1.ppf(params[1])
        params[2]  = self.Prior_2.ppf(params[2])
        params[3]  = self.Prior_3.ppf(params[3])


    def LogLike(self,params,ndim,nparams):
        ctr= params[:2]
        rc = params[2]
        g  = params[3]
        #----- Checks if parameters' values are in the ranges
        #----- only needed if prior is not properly set
        # if not Support(rc,g):
        #     return -1e50

        #------- Obtains radii and angles ---------
        radii,theta    = Deg2pc(self.cdts,ctr,self.Dist)

        ############### Radial likelihood ###################
        lf = (1.0-self.pro)*LikeField(radii,self.Rmax)
        lk = radii*(self.pro)*Kernel(radii,rc,g)

        # Normalisation constant 
        y = rc**2 + self.Rmax**2
        # print rc,g
        a = (1.0/(g-2.0))*(1.0-(rc**(g-2.0))*(y**(1.0-0.5*g))) # Truncated at Rm
        # a = 1.0/(g-2.0)                                      #  No truncated
        k  = 1.0/(a*(rc**2.0))

        llike_r  = np.sum(np.log((k*lk + lf)))
        ##################### POISSON ###################################
        quarter  = cut(theta,bins=self.quadrants,include_lowest=True)
        counts   = value_counts(quarter)
        llike_t  = self.poisson.logpmf(counts).sum()
        ##################################################################

        llike = llike_t + llike_r
        # print(llike)
        return llike



