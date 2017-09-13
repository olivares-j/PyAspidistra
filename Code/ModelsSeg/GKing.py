import sys
import numpy as np
from numba import jit
import scipy.stats as st
from Functions import Deg2pc,TruncSort,RotRadii
from pandas import cut, value_counts

import scipy.integrate as integrate
print "GKing Segregated imported!"

lo     = 1e-5

@jit
def Support(rca,rta,rcb,rtb,a,b):
    # if rca <= 0 : return False
    # if rcb <= 0 : return False
    if rcb > rca: return False
    if rta <= rca : return False
    if rtb <= rcb : return False
    if rtb > rta: return False
    # if a <= 0 : return False
    # if b <= 0 : return False
    # if a > 10.0 or b > 10.0 : return False   # To avoid overflows
    return True

@jit
def Kernel(r,rc,rt,a,b):
    x  = (1.0 +  (r/rc)**(1./a))**-a
    y  = (1.0 + (rt/rc)**(1./a))**-a
    z  = (x-y + 0j)**b
    return z.real

@jit
def Kernel1(r,rc,rt,a,b):
    x  = (1.0 +  (r/rc)**(1./a))**-a
    y  = (1.0 + (rt/rc)**(1./a))**-a
    z  = (x-y + 0j)**b
    return z.real

def cdf(r,params,Rm):
    return NormCte(np.array([params,r]))/NormCte(np.array([params,Rm]))


def Number(r,params,Rm,Nstr):
    rca = params[3]
    rta = params[4]
    a   = params[7]
    b   = params[8]

    cte = NormCte(np.array([rca,rta,a,b,Rm]))
    Num = np.vectorize(lambda y: integrate.quad(lambda x:x*Kernel1(x,rca,rta,a,b)/cte,lo,y,
                epsabs=1.49e-03, epsrel=1.49e-03,limit=1000)[0])
    return Nstr*Num(r)


def Density(r,params,Rm):
    rca = params[3]
    rta = params[4]
    a   = params[7]
    b   = params[8]

    cte = NormCte(np.array([rca,rta,a,b,Rm]))
    Den = np.vectorize(lambda x:Kernel1(x,rca,rta,a,b)/cte)
    return Den(r)


def DenSeg(r,params,Rm,delta):
    params[3] = params[3] + (delta*params[9])
    return Density(r,params,Rm)


def NormCte(z):
    cte = integrate.quad(lambda x:x*Kernel1(x,z[0],z[1],z[2],z[3]),lo,z[4],
                    epsabs=1.49e-03, epsrel=1.49e-03,limit=1000)[0]
    return cte

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
        self.Prior_4    = st.halfcauchy(loc=0.01,scale=hyp[3])
        self.Prior_5    = st.halfcauchy(loc=0.01,scale=hyp[2])
        self.Prior_6    = st.halfcauchy(loc=0.01,scale=hyp[3])
        self.Prior_7    = st.truncexpon(b=hyp[4],loc=0.01,scale=hyp[5])
        self.Prior_8    = st.truncexpon(b=hyp[4],loc=0.01,scale=hyp[5])
        self.Prior_9    = st.norm(loc=hyp[6],scale=hyp[7])
        print("Module Initialized")

    def Priors(self,params, ndim, nparams):
        params[0]  = self.Prior_0.ppf(params[0])
        params[1]  = self.Prior_1.ppf(params[1])
        params[2]  = self.Prior_2.ppf(params[2])
        params[3]  = self.Prior_3.ppf(params[3])
        params[4]  = self.Prior_4.ppf(params[4])
        params[5]  = self.Prior_5.ppf(params[5])
        params[6]  = self.Prior_6.ppf(params[6])
        params[7]  = self.Prior_7.ppf(params[7])
        params[8]  = self.Prior_8.ppf(params[8])
        params[9]  = self.Prior_9.ppf(params[9])


    def LogLike(self,params,ndim,nparams):
        ctr = params[:2]
        dlt = params[2]
        rca = params[3]
        rta = params[4]
        rcb = params[5]
        rtb = params[6]
        a   = params[7]
        b   = params[8]
        slp = params[9]
         #----- Checks if parameters' values are in the ranges
        if not Support(rca,rta,rcb,rtb,a,b) : 
            return -1e50

        #------- Obtains radii and angles ---------
        radii,theta    = RotRadii(self.cdts,ctr,self.Dist,dlt)

        rc = (rca*rcb)/np.sqrt((rcb*np.cos(theta))**2+(rca*np.sin(theta))**2)
        rts = (rta*rtb)/np.sqrt((rtb*np.cos(theta))**2+(rta*np.sin(theta))**2)

        # Include mass segregation with linear relation
        rcs = rc + (slp*self.delta_band)


        #----- Checks if parameters' values are in the ranges
        if np.min(rcs) <= 0 or np.greater(rcs,rts).any(): 
            return -1e50

        ############### Radial likelihood ###################
        # Computes likelihood
        lf = (1.0-self.pro)*LikeField(radii,self.Rmax)
        lk = radii*(self.pro)*Kernel(radii,rcs,rts,a,b)

        # In king's profile no objects is larger than tidal radius
        idBad = np.where(radii > rts)[0]
        lk[idBad] = 0.0

        # Normalisation constant
        ups      = self.Rmax*np.ones_like(rts)
        ids      = np.where(rts < self.Rmax)[0]
        ups[ids] = rts[ids]

        aes      = a*np.ones_like(rts)
        bes      = b*np.ones_like(rts)

        cte = np.array(map(NormCte,np.c_[rcs,rts,aes,bes,ups]))

        k = 1.0/cte

        # print(np.min(cte),np.min(lk),np.min(lf))

        llike_r  = np.sum(np.log((k*lk + lf)))
        if np.isnan(llike_r):
            return -1e50
        ##################### POISSON ###################################
        quarter  = cut(theta,bins=self.quadrants,include_lowest=True)
        counts   = value_counts(quarter)
        llike_t  = self.poisson.logpmf(counts).sum()
        ##################################################################

        llike = llike_t + llike_r
        # print(llike)
        return llike



