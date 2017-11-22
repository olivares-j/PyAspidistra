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
from Functions import Deg2pc,TruncSort
from pandas import cut, value_counts

print "King Centre imported!"

@jit
def Support(rc,rt):
    # if rc <= 0  : return False
    if rt <= rc : return False
    return True

@jit
def cdf(r,params,Rm):
    rc = params[2]
    rt = params[3]
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
def Kernel(r,rc,rt):
    x = 1 + (r/rc)**2
    y = 1 + (rt/rc)**2
    k = ((x**(-0.5))-(y**-0.5))**2
    return k

@jit
def LikeField(r,rm):
    return 2.*r/rm**2

def Density(r,params,Rmax):
    rc = params[2]
    rt = params[3]
    ker = Kernel(r,rc,rt)
    # In king's profile no objects is larger than tidal radius
    idBad = np.where(r > rt)[0]
    ker[idBad] = 0.0

    # Normalisation constant
    # w = rc**2 +  r**2 
    y = rc**2 + Rmax**2
    z = rc**2 + rt**2
    a = (Rmax**2)/z +  4*(rc-np.sqrt(y))/np.sqrt(z) + np.log(y) - 2*np.log(rc) # Truncated at Rm
    # a = (rt**2)/z - 4.0 +  4.0*(rc/np.sqrt(z)) + np.log(z) - 2*np.log(rc)  # Truncated at Rt (i.e. No truncated)
    k  = 2.0/(a*(rc**2.0))

    return k*ker

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
        self.Prior_3    = st.halfcauchy(loc=0.01,scale=hyp[3])
        print "Module Initialized"

    def Priors(self,params, ndim, nparams):
        params[0]  = self.Prior_0.ppf(params[0])
        params[1]  = self.Prior_1.ppf(params[1])
        params[2]  = self.Prior_2.ppf(params[2])
        params[3]  = self.Prior_3.ppf(params[3])

    def LogLike(self,params,ndim,nparams):
        ctr= params[:2]
        rc = params[2]
        rt = params[3]
        #----- Checks if parameters' values are in the ranges
        if not Support(rc,rt):
            return -1e50

        #------- Obtains radii and angles ---------
        radii,theta    = Deg2pc(self.cdts,ctr,self.Dist)

        ############### Radial likelihood ###################

        # Computes likelihood
        lf = (1.0-self.pro)*LikeField(radii,self.Rmax)
        lk = radii*(self.pro)*Kernel(radii,rc,rt)

        # In king's profile no objects is larger than tidal radius
        idBad = np.where(radii > rt)[0]
        lk[idBad] = 0.0

        # Normalisation constant
        # w = rc**2 +  r**2 
        y = rc**2 + self.Rmax**2
        z = rc**2 + rt**2
        a = (self.Rmax**2)/z +  4*(rc-np.sqrt(y))/np.sqrt(z) + np.log(y) - 2*np.log(rc) # Truncated at Rm
        # a = (rt**2)/z - 4.0 +  4.0*(rc/np.sqrt(z)) + np.log(z) - 2*np.log(rc)  # Truncated at Rt (i.e. No truncated)
        k  = 2.0/(a*(rc**2.0))

        llike_r  = np.sum(np.log((k*lk + lf)))
        ##################### POISSON ###################################
        quarter  = cut(theta,bins=self.quadrants,include_lowest=True)
        counts   = value_counts(quarter)
        llike_t  = self.poisson.logpmf(counts).sum()
        ##################################################################

        llike = llike_t + llike_r
        # print(llike)
        return llike







