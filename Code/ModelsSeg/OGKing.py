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
from __future__ import absolute_import, unicode_literals, print_function
import sys
import numpy as np
from numba import jit
import scipy.stats as st
from Functions import RotRadii
from pandas import cut, value_counts
import scipy.integrate as integrate

lo = 1e-5
a  = 0.418
b  = 2.022

@jit
def Support(rca,rta,rcb,rtb):
    # if rca <= 0 : return False
    # if rcb <= 0 : return False
    if rcb > rca: return False
    if rta <= rca : return False
    if rtb <= rcb : return False
    if rtb > rta: return False
    return True

@jit
def Kernel(r,rc,rt):
    x  = (1.0 +  (r/rc)**(1.0/a))**(-a)
    y  = (1.0 + (rt/rc)**(1.0/a))**(-a)
    z  = (x-y + 0j)**b
    return z.real

@jit
def Kernel1(r,rc,rt):
    x  = (1.0 +  (r/rc)**(1./a))**-a
    y  = (1.0 + (rt/rc)**(1./a))**-a
    z  = (x-y + 0j)**b
    return z.real

def cdf(r,theta,params,Rm):
    rca = params[3]
    rta = params[4]
    rcb = params[5]
    rtb = params[6]

    rc = (rca*rcb)/np.sqrt((rcb*np.cos(theta))**2+(rca*np.sin(theta))**2)
    rt = (rta*rtb)/np.sqrt((rtb*np.cos(theta))**2+(rta*np.sin(theta))**2)

    return NormCte(np.array([rc,rt,a,b,r]))/NormCte(np.array([rc,rt,a,b,Rm]))


def Number(r,theta,params,Rm,Nstr):
    rca = params[3]
    rta = params[4]
    rcb = params[5]
    rtb = params[6]


    rc = (rca*rcb)/np.sqrt((rcb*np.cos(theta))**2+(rca*np.sin(theta))**2)
    rt = (rta*rtb)/np.sqrt((rtb*np.cos(theta))**2+(rta*np.sin(theta))**2)

    if rt < Rm:
        up = rt
    else:
        up = Rm

    cte = NormCte(np.array([rc,rt,up]))
    Num = np.vectorize(lambda y: integrate.quad(lambda x:x*Kernel1(x,rc,rt)/cte,lo,y,
                epsabs=1.49e-03, epsrel=1.49e-03,limit=1000)[0])
    return Nstr*Num(r)


def Density(r,theta,params,Rm):
    rca = params[3]
    rta = params[4]
    rcb = params[5]
    rtb = params[6]


    rc = (rca*rcb)/np.sqrt((rcb*np.cos(theta))**2+(rca*np.sin(theta))**2)
    rt = (rta*rtb)/np.sqrt((rtb*np.cos(theta))**2+(rta*np.sin(theta))**2)

    if rt < Rm:
        up = rt
    else:
        up = Rm

    cte = NormCte(np.array([rc,rt,up]))
    Den = np.vectorize(lambda x:Kernel1(x,rc,rt)/cte)
    return Den(r)

def DenSeg(r,theta,params,Rm,delta):
    params[3] = params[3] + (delta*params[7])
    return Density(r,theta,params,Rm)

def NormCte(z):
    cte = integrate.quad(lambda x:x*Kernel1(x,z[0],z[1]),lo,z[2],epsabs=1.49e-03, epsrel=1.49e-03,limit=1000)[0]
    return cte

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
        print("Mode of band at ",self.mode)

        #---- repleace NANs by mode -----
        idnv            = np.setdiff1d(np.arange(len(band_all)),idv)
        band_all[idnv]  = self.mode
        self.delta_band = band_all - self.mode
        #-------------- priors ----------------
        self.Prior_0    = st.norm(loc=centre_init[0],scale=hyp[0])
        self.Prior_1    = st.norm(loc=centre_init[1],scale=hyp[1])
        self.Prior_2    = st.uniform(loc=0,scale=np.pi)
        self.Prior_3    = st.halfcauchy(loc=0.01,scale=hyp[2])
        self.Prior_4    = st.halfcauchy(loc=0.01,scale=hyp[3])
        self.Prior_5    = st.halfcauchy(loc=0.01,scale=hyp[2])
        self.Prior_6    = st.halfcauchy(loc=0.01,scale=hyp[3])
        self.Prior_7    = st.norm(loc=hyp[4],scale=hyp[5])

        print("Segregated OGKing module initialized")

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
        if not Support(rca,rta,rcb,rtb) : 
            return -1e50

        #------- Obtains radii and angles ---------
        radii,theta    = RotRadii(self.cdts,ctr,self.Dist,dlt)

        rc  = (rca*rcb)/np.sqrt((rcb*np.cos(theta))**2+(rca*np.sin(theta))**2)
        rts = (rta*rtb)/np.sqrt((rtb*np.cos(theta))**2+(rta*np.sin(theta))**2)

        # Include mass segregation with linear relation
        rcs = rc + (slp*self.delta_band)

        # print np.greater(rcs,rts).any()

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

        cte = np.array(list(map(NormCte,np.c_[rcs,rts,ups])))

        k = 1.0/cte

        llike_r  = np.sum(np.log((k*lk + lf)))
        if np.isnan(llike_r):
            return -1e50
        ##################### POISSON ###################################
        quarter  = cut(theta,bins=self.quadrants,include_lowest=True)
        counts   = value_counts(quarter)
        llike_t  = self.poisson.logpmf(counts).sum()
        ##################################################################

        llike = llike_t + llike_r
        # if np.isfinite(llike):
        #     sys.exit()
        # print(slp)
        # print(llike)
        return llike



