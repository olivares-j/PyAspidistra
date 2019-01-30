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

print "King Elliptic imported!"

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
def cdf(r,theta,params,Rm):
    rca = params[3]
    rta = params[4]
    rcb = params[5]
    rtb = params[6]

    rc = (rca*rcb)/np.sqrt((rcb*np.cos(theta))**2+(rca*np.sin(theta))**2)
    rt = (rta*rtb)/np.sqrt((rtb*np.cos(theta))**2+(rta*np.sin(theta))**2)

    w = rc**2 +  r**2 
    y = rc**2 + Rm**2
    z = rc**2 + rt**2
    a = (r**2)/z  +  4*(rc-np.sqrt(w))/np.sqrt(z) + np.log(w) - 2*np.log(rc)
    b = (Rm**2)/z +  4*(rc-np.sqrt(y))/np.sqrt(z) + np.log(y) - 2*np.log(rc)
    return a/b

@jit
def Number(r,theta,params,Rm,Nstr):
    # Rm must be less or equal to rt
    return Nstr*cdf(r,theta,params,Rm)

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

def Density(r,theta,params,Rm):
    rca = params[3]
    rta = params[4]
    rcb = params[5]
    rtb = params[6]

    rc = (rca*rcb)/np.sqrt((rcb*np.cos(theta))**2+(rca*np.sin(theta))**2)
    rt = (rta*rtb)/np.sqrt((rtb*np.cos(theta))**2+(rta*np.sin(theta))**2)

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
        #-------------- priors ----------------
        self.Prior_0    = st.norm(loc=centre_init[0],scale=hyp[0])
        self.Prior_1    = st.norm(loc=centre_init[1],scale=hyp[1])
        self.Prior_2    = st.uniform(loc=0,scale=np.pi)
        self.Prior_3    = st.halfcauchy(loc=0.01,scale=hyp[2])
        self.Prior_4    = st.halfcauchy(loc=0.01,scale=hyp[3])
        self.Prior_5    = st.halfcauchy(loc=0.01,scale=hyp[2])
        self.Prior_6    = st.halfcauchy(loc=0.01,scale=hyp[3])
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
        rta = params[4]
        rcb = params[5]
        rtb = params[6]
        #----- Checks if parameters' values are in the ranges
        if not Support(rca,rta,rcb,rtb):
            return -1e50

        #------- Obtains radii and angles ---------
        radii,theta    = RotRadii(self.cdts,ctr,self.Dist,dlt)

        rcs = (rca*rcb)/np.sqrt((rcb*np.cos(theta))**2+(rca*np.sin(theta))**2)
        rts = (rta*rtb)/np.sqrt((rtb*np.cos(theta))**2+(rta*np.sin(theta))**2)

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
        if not np.isfinite(llike):
            return -1e50
        
        return llike







