import sys
import numpy as np
from numba import jit
import scipy.stats as st
from Functions import Deg2pc,TruncSort
from pandas import cut, value_counts

from scipy.special import hyp2f1

print "GDP Centre imported!"

@jit
def Support(rc,a,b,g):
    if rc <= 0 : return False
    if a  <= 0 : return False
    if b  <  0 : return False
    if g  <  0 : return False
    if g  >= 2 : return False
    return True

@jit
def cdf(r,params,Rm):
    rc = params[2]
    a  = params[3]
    b  = params[4]
    g  = params[5]

    # Normalisation constant
    x = - ((rc/Rm)**(-1.0/a))
    y = - ((r/rc)**(1.0/a))
    z = (r/Rm)**(2.0-g)
    u =  a*(b-g)
    v = -a*(g-2.0)
    w = 1.0 + v

    c = hyp2f1(u,v,w,x)
    d = hyp2f1(u,v,w,y)

    return z*d.real/c.real

@jit
def Number(r,params,Rm,Nstr):
    return Nstr*cdf(r,params,Rm)

@jit
def Kernel(r,rc,a,b,g):
    y = ((r/rc)**g)*((1.0 + (r/rc)**(1.0/a))**(a*(b-g)))
    return 1.0/y

@jit
def Density(r,params,Rm):
    rc = params[2]
    a  = params[3]
    b  = params[4]
    g  = params[5]

     # Normalisation constant
    x = -1.0*((rc/Rm)**(-1.0/a))
    u = -a*(g-2.0)
    v = 1.0+ a*(g-b)
    z = ((-1.0+0j)**(a*(g-2.0)))*a*(rc**2)
    betainc = (((x+0j)**u)/u)*hyp2f1(u,1.0-v,u+1.0,x)
    k  = 1.0/np.abs(z*betainc)

    return k*Kernel(r,rc,a,b,g)

@jit
def LikeField(r,rm):
    return 2.*r/rm**2

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
        #------------- poisson ----------------
        self.quadrants  = [0,np.pi/2.0,np.pi,3.0*np.pi/2.0,2.0*np.pi]
        self.poisson    = st.poisson(len(r)/4.0)
        #-------------- priors ----------------
        self.Prior_0    = st.norm(loc=centre_init[0],scale=hyp[0])
        self.Prior_1    = st.norm(loc=centre_init[1],scale=hyp[1])
        self.Prior_2    = st.halfcauchy(loc=0.01,scale=hyp[2])
        self.Prior_3    = st.truncexpon(b=hyp[3],loc=0.01,scale=hyp[4])
        self.Prior_4    = st.truncexpon(b=hyp[3],loc=0.01,scale=hyp[4])
        self.Prior_5    = st.truncexpon(b=2.0,loc=0.01,scale=hyp[4])
        print("Module Initialized")

    def Priors(self,params, ndim, nparams):
        #------- Uniform Priors -------
        params[0]  = self.Prior_0.ppf(params[0])
        params[1]  = self.Prior_1.ppf(params[1])
        params[2]  = self.Prior_2.ppf(params[2])
        params[3]  = self.Prior_3.ppf(params[3])
        params[4]  = self.Prior_4.ppf(params[4])
        params[5]  = self.Prior_5.ppf(params[5])

    def LogLike(self,params,ndim,nparams):
        ctr= params[:2]
        rc = params[2]
        a  = params[3]
        b  = params[4]
        g  = params[5]
         #----- Checks if parameters' values are in the ranges
        # if not Support(rc,a,b,g) : 
        #     return -1e50

        #------- Obtains radii and angles ---------
        radii,theta    = Deg2pc(self.cdts,ctr,self.Dist)

        ############### Radial likelihood ###################
        # Computes likelihood
        lf = (1.0-self.pro)*LikeField(radii,self.Rmax)
        lk = radii*(self.pro)*Kernel(radii,rc,a,b,g)

        # Normalisation constant
        x = -1.0*((rc/self.Rmax)**(-1.0/a))
        u = -a*(g-2.0)
        v = 1.0+ a*(g-b)
        z = ((-1.0+0j)**(a*(g-2.0)))*a*(rc**2)
        betainc = (((x+0j)**u)/u)*hyp2f1(u,1.0-v,u+1.0,x)
        k  = 1.0/np.abs(z*betainc)

        llike_r  = np.sum(np.log((k*lk + lf)))
        ##################### POISSON ###################################
        quarter  = cut(theta,bins=self.quadrants,include_lowest=True)
        counts   = value_counts(quarter)
        llike_t  = self.poisson.logpmf(counts).sum()
        ##################################################################

        llike = llike_t + llike_r
        # print(llike)
        return llike


