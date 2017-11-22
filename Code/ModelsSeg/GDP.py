'''
Copyright 2017 Javier Olivares Romero

This file is part of PyAspidistra.

    PyAspidistra is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PyAspidistra is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with PyAspidistra.  If not, see <http://www.gnu.org/licenses/>.
'''
import sys
import numpy as np
from numba import jit
import scipy.stats as st
from Functions import Deg2pc,TruncSort,RotRadii
from pandas import cut, value_counts

from scipy.special import hyp2f1

print "GDP Segregated imported!"

@jit
def Support(rca,rcb,a,b,g):
    # if rca <= 0 : return False
    # if rcb <= 0 : return False
    if rcb > rca: return False
    # if a   <= 0 : return False
    # if b   <  0 : return False
    # if g   <  0 : return False
    # if g   >= 2 : return False
    # if a > 100.0 or b > 100.0 : return False   # To avoid overflows
    return True

@jit
def cdf(r,params,Rm):
    rca = params[3]
    rcb = params[4]
    a   = params[5]
    b   = params[6]
    g   = params[7]

    # Normalisation constant
    x = - ((rca/Rm)**(-1.0/a))
    y = - ((r/rca)**(1.0/a))
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
def Kernel1(r,rc,a,b,g):
    y = ((r/rc)**g)*((1.0 + (r/rc)**(1.0/a))**(a*(b-g)))
    return 1.0/y

@jit
def Density(r,params,Rm):
    rca = params[3]
    rcb = params[4]
    a   = params[5]
    b   = params[6]
    g   = params[7]

     # Normalisation constant
    x = -1.0*((rca/Rm)**(-1.0/a))
    u = -a*(g-2.0)
    v = 1.0+ a*(g-b)
    z = ((-1.0+0j)**(a*(g-2.0)))*a*(rca**2)
    betainc = (((x+0j)**u)/u)*hyp2f1(u,1.0-v,u+1.0,x)
    k  = 1.0/np.abs(z*betainc)

    return k*Kernel1(r,rca,a,b,g)

def DenSeg(r,params,Rm,delta):
    params[3] = params[3] + (delta*params[8])
    return Density(r,params,Rm)


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
        self.Prior_4    = st.halfcauchy(loc=0.01,scale=hyp[2])
        self.Prior_5    = st.truncexpon(b=hyp[3],loc=0.01,scale=hyp[4])
        self.Prior_6    = st.truncexpon(b=hyp[3],loc=0.01,scale=hyp[4])
        self.Prior_7    = st.truncexpon(b=2.0,loc=0.01,scale=hyp[4])
        self.Prior_8    = st.norm(loc=hyp[5],scale=hyp[6])
        print("Module Initialized")

    def Priors(self,params, ndim, nparams):
        #------- Uniform Priors -------
        params[0]  = self.Prior_0.ppf(params[0])
        params[1]  = self.Prior_1.ppf(params[1])
        params[2]  = self.Prior_2.ppf(params[2])
        params[3]  = self.Prior_3.ppf(params[3])
        params[4]  = self.Prior_4.ppf(params[4])
        params[5]  = self.Prior_5.ppf(params[5])
        params[6]  = self.Prior_6.ppf(params[6])
        params[7]  = self.Prior_7.ppf(params[7])
        params[8]  = self.Prior_8.ppf(params[8])

    def LogLike(self,params,ndim,nparams):
        ctr = params[:2]
        dlt = params[2]
        rca = params[3]
        rcb = params[4]
        a   = params[5]
        b   = params[6]
        g   = params[7]
        slp = params[8]
         #----- Checks if parameters' values are in the ranges
        if not Support(rca,rcb,a,b,g) : 
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
        # Computes likelihood
        lf = (1.0-self.pro)*LikeField(radii,self.Rmax)
        lk = radii*(self.pro)*Kernel(radii,rcs,a,b,g)

        # Normalisation constant
        x = -1.0*((rcs/self.Rmax)**(-1.0/a))
        u = -a*(g-2.0)
        v = 1.0+ a*(g-b)
        w =  (((x+0j)**u)/u)
        z = ((-1.0+0j)**(a*(g-2.0)))*a*(rcs**2)
        h = hyp2f1(u,1.0-v,u+1.0,x)
        cte = np.abs(z*h*w)
        k   = 1.0/cte

        llike_r  = np.sum(np.log((k*lk + lf)))
        ##################### POISSON ###################################
        quarter  = cut(theta,bins=self.quadrants,include_lowest=True)
        counts   = value_counts(quarter)
        llike_t  = self.poisson.logpmf(counts).sum()
        ##################################################################

        llike = llike_t + llike_r
        # if not np.isfinite(llike):
        #     ids = np.where(np.isnan(k))[0]
        #     print h[ids],u,v,a,b,g
        #     sys.exit()
        #     return -1e50
        # print(llike)
        return llike


