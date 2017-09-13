import sys
import numpy as np
from numba import jit
import scipy.stats as st
from Functions import Deg2pc,TruncSort,RotRadii
from pandas import cut, value_counts

print "EFF Segregated imported!"

@jit
def Support(rca,rcb):
    if rcb > rca: return False
    return True

@jit
def Number(r,params,Rm,Nstr):
    return Nstr*cdf(r,params,Rm)

@jit
def cdf(r,params,Rm):
    rc = params[3]
    g  = params[5]
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
def Kernel1(r,rc,g):
    x  = 1.0+(r/rc)**2
    return x**(-0.5*g)

@jit
def LikeField(r,rm):
    return 2.0*r/(rm**2)

@jit
def Density(r,params,Rm):
    rc = params[3]
    g  = params[5]
    y = rc**2 + Rm**2
    a = (1.0/(g-2.0))*(1.0-(rc**(g-2.0))*(y**(1.0-0.5*g))) # Truncated at Rm
    # a = 1.0/(g-2.0)                                      #  No truncated
    k  = 1.0/(a*(rc**2.0))
    return k*Kernel1(r,rc,g)

def DenSeg(r,params,Rm,delta):
    params[3] = params[3] + (delta*params[6])
    return Density(r,params,Rm)

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
        #-------------- Finds the mode of the band -------
        band_all        = c[:,3]
        idv             = np.where(np.isfinite(c[:,3]))[0]
        band            = c[idv,3]
        kde             = st.gaussian_kde(band)
        x               = np.linspace(np.min(band),np.max(band),num=1000)
        self.mode       = x[kde(x).argmax()]
        print "Mode of band at ",self.mode

        #---- repleace NANs by mode -----
        idnv            = np.setdiff1d(np.arange(len(band_all)),idv)
        band_all[idnv]  = self.mode
        self.delta_band = band_all - self.mode
        #------------- poisson ----------------
        self.quadrants  = [0,np.pi/2.0,np.pi,3.0*np.pi/2.0,2.0*np.pi]
        self.poisson    = st.poisson(len(r)/4.0)
        #-------------- priors ----------------
        self.Prior_0    = st.norm(loc=centre_init[0],scale=hyp[0])
        self.Prior_1    = st.norm(loc=centre_init[1],scale=hyp[1])
        self.Prior_2    = st.uniform(loc=-0.5*np.pi,scale=np.pi)
        self.Prior_3    = st.halfcauchy(loc=0.01,scale=hyp[2])
        self.Prior_4    = st.halfcauchy(loc=0.01,scale=hyp[2])
        self.Prior_5    = st.truncexpon(b=hyp[3],loc=2.01,scale=hyp[4])
        self.Prior_6    = st.norm(loc=hyp[5],scale=hyp[6])
        print "Module Initialized"

    def Priors(self,params, ndim, nparams):
        params[0]  = self.Prior_0.ppf(params[0])
        params[1]  = self.Prior_1.ppf(params[1])
        params[2]  = self.Prior_2.ppf(params[2])
        params[3]  = self.Prior_3.ppf(params[3])
        params[4]  = self.Prior_4.ppf(params[4])
        params[5]  = self.Prior_5.ppf(params[5])
        params[6]  = self.Prior_6.ppf(params[6])


    def LogLike(self,params,ndim,nparams):
        ctr = params[:2]
        dlt = params[2]
        rca = params[3]
        rcb = params[4]
        g   = params[5]
        slp = params[6]
        #----- Checks if parameters' values are in the ranges
        if not Support(rca,rcb):
            return -1e50

        #------- Obtains radii and angles ---------
        radii,theta    = RotRadii(self.cdts,ctr,self.Dist,dlt)

        rc = (rca*rcb)/np.sqrt((rcb*np.cos(theta))**2+(rca*np.sin(theta))**2)

        # Include mass segregation with linear relation
        rcs = rc + (slp*self.delta_band)

        #----- Checks if parameters' values are in the ranges
        if np.min(rcs) <= 0 : 
            return -1e50

        ############### Radial likelihood ###################
        lf = (1.0-self.pro)*LikeField(radii,self.Rmax)
        lk = radii*(self.pro)*Kernel(radii,rcs,g)

        # Normalisation constant 
        y = rcs**2 + self.Rmax**2
        a = (1.0/(g-2.0))*(1.0-(rcs**(g-2.0))*(y**(1.0-0.5*g))) # Truncated at Rm
        # a = 1.0/(g-2.0)                                      #  No truncated
        k  = 1.0/(a*(rcs**2.0))

        llike_r  = np.sum(np.log((k*lk + lf)))
        ##################### POISSON ###################################
        quarter  = cut(theta,bins=self.quadrants,include_lowest=True)
        counts   = value_counts(quarter)
        llike_t  = self.poisson.logpmf(counts).sum()
        ##################################################################

        llike =  llike_r + llike_t
        # print(llike)
        return llike



