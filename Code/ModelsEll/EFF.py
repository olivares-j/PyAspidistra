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
import math
from numba import jit
import scipy.stats as st
from Functions import RotRadii
from pandas import cut, value_counts

@jit
def Support(rca,rcb,g):
    # if rca <= 0 : return False
    # if rcb <= 0 : return False
    if rcb > rca: return False
    # if  g <= 2.0 : return False
    # if  g > 100.0 : return False
    return True

@jit
def Number(r,theta,params,Rm,Nstr):
    return Nstr*cdf(r,theta,params,Rm)

@jit
def cdf(r,theta,params,Rm):
    rca = params[3]
    rcb = params[4]
    g   = params[5]
    rc = (rca*rcb)/np.sqrt((rcb*np.cos(theta))**2+(rca*np.sin(theta))**2)
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
def Density(r,theta,params,Rm):
    rca   = params[3]
    rcb   = params[4]
    g     = params[5]
    rc = (rca*rcb)/np.sqrt((rcb*np.cos(theta))**2+(rca*np.sin(theta))**2)
    y  = rc**2 + Rm**2
    a  = (1.0/(g-2.0))*(1.0-(rc**(g-2.0))*(y**(1.0-0.5*g))) # Truncated at Rm
    # a = 1.0/(g-2.0)                                      #  No truncated
    k  = 1.0/(a*(rc**2.0))
    return k*Kernel1(r,rc,g)

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
        self.Prior_4    = st.halfcauchy(loc=0.01,scale=hyp[2])
        self.Prior_5    = st.truncexpon(b=hyp[3],loc=2.01,scale=hyp[4])
        print("EFF Elliptic module initialized")

    def Priors(self,params, ndim, nparams):
        params[0]  = self.Prior_0.ppf(params[0])
        params[1]  = self.Prior_1.ppf(params[1])
        params[2]  = self.Prior_2.ppf(params[2])
        params[3]  = self.Prior_3.ppf(params[3])
        params[4]  = self.Prior_4.ppf(params[4])
        params[5]  = self.Prior_5.ppf(params[5])


    def LogLike(self,params,ndim,nparams):
        ctr = params[:2]
        dlt = params[2]
        rca = params[3]
        rcb = params[4]
        g   = params[5]
        #----- Checks if parameters' values are in the ranges
        if not Support(rca,rcb,g):
            return -1e50

        #------- Obtains radii and angles ---------
        radii,theta = RotRadii(self.cdts,ctr,self.Dist,dlt)

        rcs = (rca*rcb)/np.sqrt((rcb*np.cos(theta))**2+(rca*np.sin(theta))**2)

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



