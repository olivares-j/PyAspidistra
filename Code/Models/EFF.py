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
from Functions import RotRadii,TruncSort

print("EFF NoCentre imported")

@jit
def Support(rc,g):
    if rc <= 0.0 : return False
    if  g <= 2.0 : return False
    if  g > 100.0 : return False
    return True

@jit
def Number(r,theta,params,Rm,Nstr):
    return Nstr*cdf(r,theta,params,Rm)

@jit
def cdf(r,theta,params,Rm):
    rc = params[0]
    g  = params[1]
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
def Density(r,theta,params,Rm):
    rc = params[0]
    g  = params[1]
    y = rc**2 + Rm**2
    a = (1.0/(g-2.0))*(1.0-(rc**(g-2.0))*(y**(1.0-0.5*g))) # Truncated at Rm
    # a = 1.0/(g-2.0)                                      #  No truncated
    k  = 1.0/(a*(rc**2.0))
    return k*Kernel(r,rc,g)

class Module:
    """
    Chain for computing the likelihood 
    """
    def __init__(self,cdts,Rcut,hyp,Dist,centre):
        """
        Constructor of the logposteriorModule
        """
        rad,thet        = RotRadii(cdts,centre,Dist,0)
        c,r,t,self.Rmax = TruncSort(cdts,rad,thet,Rcut)
        self.pro        = c[:,2]
        self.rad        = r
        
        self.Prior_0    = st.halfcauchy(loc=0.01,scale=hyp[0])
        self.Prior_1    = st.halfcauchy(loc=2.001,scale=hyp[1])
                        
        print("Module Initialized")

    def Priors(self,params, ndim, nparams):
        #-------- Uniform Priors -------
        params[0]  = self.Prior_0.ppf(params[0])
        params[1]  = self.Prior_1.ppf(params[1])


    def LogLike(self,params,ndim,nparams):
        rc = params[0]
        g  = params[1]
        #----- Checks if parameters' values are in the ranges
        if not Support(rc,g):
            return -1e50
        # Computes likelihood
        lf = (1.0-self.pro)*LikeField(self.rad,self.Rmax)
        lk = self.rad*(self.pro)*Kernel(self.rad,rc,g)


        # Normalisation constant
        # w = rc**2 +  r**2 
        y = rc**2 + self.Rmax**2
        a = (1.0/(g-2.0))*(1.0-(rc**(g-2.0))*(y**(1.0-0.5*g))) # Truncated at Rm
        # a = 1.0/(g-2.0)                                      #  No truncated
        k  = 1.0/(a*(rc**2.0))

        llike  = np.sum(np.log((k*lk + lf)))
        # print llike 
        return llike



