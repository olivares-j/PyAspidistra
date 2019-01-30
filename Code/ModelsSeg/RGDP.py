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
from Functions import RotRadii
from pandas import cut, value_counts

from scipy.special import hyp2f1

print "RGDP Segregated imported!"

@jit
def Support(rca,rcb,a,b):
    # if rca <= 0 : return False
    # if rcb <= 0 : return False
    if rcb > rca: return False
    if a   <= 0 : return False
    if b   <= 0 : return False
    # if a > 100.0 or b > 100.0 : return False
    return True

@jit
def cdf(r,theta,params,Rm):
    rca = params[3]
    rcb = params[4]
    a   = params[5]
    b   = params[6]

    rc = (rca*rcb)/np.sqrt((rcb*np.cos(theta))**2+(rca*np.sin(theta))**2)

    # Normalisation constant
    x = -((rc/Rm)**(-1.0/a) + 0.0j) 
    y = -((r/rc)**(1.0/a)   + 0.0j)
    u = 2.0*a

    c = ((x**u)/u)*hyp2f1(u,a*b,1.0 + u,x)
    d = ((y**u)/u)*hyp2f1(u,a*b,1.0 + u,y)

    return np.abs(d)/np.abs(c)

@jit
def Number(r,theta,params,Rm,Nstr):
    return Nstr*cdf(r,theta,params,Rm)

@jit
def Kernel(r,rc,a,b):
    y = (1.0 + (r/rc)**(1.0/a))**(a*b)
    return 1.0/y

@jit
def Kernel1(r,rc,a,b):
    y = (1.0 + (r/rc)**(1.0/a))**(a*b)
    return 1.0/y

@jit
def Density(r,theta,params,Rm):
    rca = params[3]
    rcb = params[4]
    a   = params[5]
    b   = params[6]

    rc = (rca*rcb)/np.sqrt((rcb*np.cos(theta))**2+(rca*np.sin(theta))**2)

    # Normalisation constant
    x = -1.0*((rc/Rm)**(-1.0/a))
    c = np.abs((Rm**2)*hyp2f1(2.0*a,a*b,1 + 2.0*a,x))
    k  = 2.0/c

    return k*Kernel1(r,rc,a,b)

def DenSeg(r,theta,params,Rm,delta):
    params[3] = params[3] + (delta*params[7])
    return Density(r,theta,params,Rm)

@jit
def LikeField(r,rm):
    return 2.*r/rm**2

class Module:
    """
    Chain for computing the likelihood 
    """
    def __init__(self,cdts,Rmax,hyp,Dist,centre_init):
        """
        Constructor of the logposteriorModule
        """
        self.Rmax       = Rmax
        self.pro        = cdts[:,2]
        self.cdts       = cdts[:,:2]
        self.Dist       = Dist
        #------------- poisson ----------------
        self.quadrants  = [0,np.pi/2.0,np.pi,3.0*np.pi/2.0,2.0*np.pi]
        self.poisson    = st.poisson(len(self.pro)/4.0)
        #-------------- Finds the mode of the band -------
        band_all        = cdts[:,3]
        idv             = np.where(np.isfinite(cdts[:,3]))[0]
        band            = cdts[idv,3]
        kde             = st.gaussian_kde(band)
        x               = np.linspace(np.min(band),np.max(band),num=1000)
        self.mode       = x[kde(x).argmax()]
        print "Mode of band at ",self.mode

        #---- repleace NANs by mode -----
        idnv            = np.setdiff1d(np.arange(len(band_all)),idv)
        band_all[idnv]  = self.mode
        self.delta_band = band_all - self.mode
        #-------------- priors ----------------
        self.Prior_0    = st.norm(loc=centre_init[0],scale=hyp[0])
        self.Prior_1    = st.norm(loc=centre_init[1],scale=hyp[1])
        self.Prior_2    = st.uniform(loc=0,scale=np.pi)
        self.Prior_3    = st.halfcauchy(loc=0.01,scale=hyp[2])
        self.Prior_4    = st.halfcauchy(loc=0.01,scale=hyp[2])
        self.Prior_5    = st.truncexpon(b=hyp[3],loc=0.01,scale=hyp[4])
        self.Prior_6    = st.truncexpon(b=hyp[3],loc=0.01,scale=hyp[4])
        self.Prior_7    = st.norm(loc=hyp[5],scale=hyp[6])
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

    def LogLike(self,params,ndim,nparams):
        ctr = params[:2]
        dlt = params[2]
        rca = params[3]
        rcb = params[4]
        a   = params[5]
        b   = params[6]
        slp = params[7]
         #----- Checks if parameters' values are in the ranges
        if not Support(rca,rcb,a,b) : 
            return -1e50

        #------- Obtains radii and angles ---------
        radii,theta    = RotRadii(self.cdts,ctr,self.Dist,dlt)

        rc = (rca*rcb)/np.sqrt((rcb*np.cos(theta))**2+(rca*np.sin(theta))**2)

        # Include mass segregation with linear relation
        rcs = rc + (slp*self.delta_band)

        if np.min(rcs) <= 0 : 
            return -1e50

        ############### Radial likelihood ###################

        # Computes likelihood
        lf = (1.0-self.pro)*LikeField(radii,self.Rmax)
        lk = radii*(self.pro)*Kernel(radii,rcs,a,b)

        # Normalisation constant
        x = -((rcs/self.Rmax)**(-1.0/a))
        c = (self.Rmax**2)*hyp2f1(2.0*a,a*b,1 + 2.0*a,x).real
        k  = 2.0/c

        if np.isnan(k).any():
            return -1e50

        llike_r  = np.sum(np.log((k*lk + lf)))
        ##################### POISSON ###################################
        quarter  = cut(theta,bins=self.quadrants,include_lowest=True)
        counts   = value_counts(quarter)
        llike_t  = self.poisson.logpmf(counts).sum()
        ##################################################################

        llike = llike_t + llike_r
        
        return llike


