import sys
import numpy as np
from numba import jit
import scipy.stats as st
from Functions import Deg2pc,TruncSort,RotRadii
from pandas import cut, value_counts

print "King Segregated imported!"

@jit
def Support(rca,rta,rcb,rtb):
    if rca <= 0 : return False
    if rcb <= 0 : return False
    if rcb > rca: return False
    if rta <= rca : return False
    if rtb <= rcb : return False
    if rtb > rta: return False
    return True

@jit
def cdf(r,params,Rm):
    rc = params[3]
    rt = params[4]
    w = rc**2 +  r**2 
    y = rc**2 + Rm**2
    z = rc**2 + rt**2
    a = (r**2)/z  +  4*(rc-np.sqrt(w))/np.sqrt(z) + np.log(w) - 2*np.log(rc)
    b = (Rm**2)/z +  4*(rc-np.sqrt(y))/np.sqrt(z) + np.log(y) - 2*np.log(rc)
    return a/b

@jit
def Number(r,params,Rm,Nstr):
    # Rm must be less or equal to rt
    return Nstr*cdf(r,params,Rm)

@jit
def Kernel(r,rc,rt): # Receives vectors
    x = 1 + (r/rc)**2
    y = 1 + (rt/rc)**2
    k = ((x**(-0.5))-(y**-0.5))**2
    return k

@jit
def Kernel1(r,rc,rt): # Receive vector (r) and rc and rt as scalars
    x = 1 + (r/rc)**2
    y = 1 + (rt/rc)**2
    k = ((x**(-0.5))-(y**-0.5))**2
    return k

@jit
def LikeField(r,rm):
    return 2.*r/rm**2

def Density(r,params,Rm):
    rc = params[3]
    rt = params[4]
    # print rc,rt

    ker = Kernel1(r,rc,rt)
    # In king's profile no objects is larger than tidal radius
    idBad = np.where(r > rt)[0]
    ker[idBad] = 0.0

    if rt < Rm:
        up = rt
    else:
        up = Rm

    cte = NormCte(np.array([rc,rt,up]))

    k  = 2.0/cte

    return k*ker

def DenSeg(r,params,Rm,delta):
    params[3] = params[3] + (delta*params[7])
    return Density(r,params,Rm)

@jit
def NormCte(z):
    rc = z[0]
    rt = z[1]
    Rm = z[2]

    y = rc**2 + Rm**2
    z = rc**2 + rt**2
    a = (Rm**2)/z +  4*(rc-np.sqrt(y))/np.sqrt(z) + np.log(y) - 2*np.log(rc) # Truncated at Rm
    k  = a*(rc**2.0)
    return k

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
        self.Prior_3    = st.halfcauchy(loc=0,scale=hyp[2])
        self.Prior_4    = st.halfcauchy(loc=0,scale=hyp[3])
        self.Prior_5    = st.halfcauchy(loc=0,scale=hyp[2])
        self.Prior_6    = st.halfcauchy(loc=0,scale=hyp[3])
        self.Prior_7    = st.uniform(loc=hyp[4],scale=hyp[5])
        print "Module Initialized"

    def Priors(self,params, ndim, nparams):
        params[0]  = self.Prior_0.ppf(params[0])
        params[1]  = self.Prior_1.ppf(params[1])
        params[2]  = self.Prior_2.ppf(params[2])
        params[3]  = self.Prior_3.ppf(params[3])
        params[4]  = self.Prior_4.ppf(params[4])
        params[5]  = self.Prior_5.ppf(params[5])
        params[6]  = self.Prior_6.ppf(params[6])
        params[7]  = self.Prior_7.ppf(params[7])

    def LogLike(self,params,ndim,nparams):
        ctr = params[:2]
        dlt = params[2]
        rca = params[3]
        rta = params[4]
        rcb = params[5]
        rtb = params[6]
        slp = params[7]
        #----- Checks if parameters' values are in the ranges
        if not Support(rca,rta,rcb,rtb):
            return -1e50

        #------- Obtains radii and angles ---------
        radii,theta    = RotRadii(self.cdts,ctr,self.Dist,dlt)

        rc  = (rca*rcb)/np.sqrt((rcb*np.cos(theta))**2+(rca*np.sin(theta))**2)
        rts = (rta*rtb)/np.sqrt((rtb*np.cos(theta))**2+(rta*np.sin(theta))**2)

        # Include mass segregation with linear relation
        rcs = rc + (slp*self.delta_band)

        #----- Checks if parameters' values are in the ranges
        if np.min(rcs) <= 0 or np.greater(rcs,rts).any():
            return -1e50

        ############### Radial likelihood ###################

        # Computes likelihood
        lf = (1.0-self.pro)*LikeField(radii,self.Rmax)
        lk = radii*(self.pro)*Kernel(radii,rcs,rts)

        # In king's profile no objects is larger than tidal radius
        idBad = np.where(radii > rts)[0]
        lk[idBad] = 0.0

        # Normalisation constant
        ups      = self.Rmax*np.ones_like(rts)
        ids      = np.where(rts < self.Rmax)[0]
        ups[ids] = rts[ids]

        cte = np.array(map(NormCte,np.c_[rcs,rts,ups]))

        k        = 2.0/cte

        llike_r  = np.sum(np.log((k*lk + lf)))
        ##################### POISSON ###################################
        quarter  = cut(theta,bins=self.quadrants,include_lowest=True)
        counts   = value_counts(quarter)
        llike_t  = self.poisson.logpmf(counts).sum()
        ##################################################################

        llike = llike_t + llike_r
        # print(llike)
        return llike







