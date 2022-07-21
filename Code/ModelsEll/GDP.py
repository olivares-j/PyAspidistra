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

from scipy.special import hyp2f1


@jit
def Support(rca,rcb,a,b,g):
    # if rca <= 0 : return False
    # if rcb <= 0 : return False
    if rcb > rca: return False
    # if a   <= 0 : return False
    # if b   <  0 : return False
    # if g   <  0 : return False
    # if g   >= 2 : return False
    return True

@jit
def cdf(r,theta,params,Rm):
    rca = params[3]
    rcb = params[4]
    a   = params[5]
    b   = params[6]
    g   = params[7]

    rc = (rca*rcb)/np.sqrt((rcb*np.cos(theta))**2+(rca*np.sin(theta))**2)
    # Normalisation constant
    x = - ((rc/Rm)**(-1.0/a))
    y = - ((r/rc)**(1.0/a))
    z = (r/Rm)**(2.0-g)
    u =  a*(b-g)
    v = -a*(g-2.0)
    w = 1.0 + v

    c = hyp2f1(u,v,w,x)
    d = hyp2f1(u,v,w,y)

    return z*d.real/c.real

@jit
def Number(r,theta,params,Rm,Nstr):
    return Nstr*cdf(r,theta,params,Rm)

@jit
def Kernel(r,rc,a,b,g):
    y = ((r/rc)**g)*((1.0 + (r/rc)**(1.0/a))**(a*(b-g)))
    return 1.0/y

@jit
def Kernel1(r,rc,a,b,g):
    y = ((r/rc)**g)*((1.0 + (r/rc)**(1.0/a))**(a*(b-g)))
    return 1.0/y

@jit
def Density(r,theta,params,Rm):
    rca = params[3]
    rcb = params[4]
    a   = params[5]
    b   = params[6]
    g   = params[7]

    rc = (rca*rcb)/np.sqrt((rcb*np.cos(theta))**2+(rca*np.sin(theta))**2)
     # Normalisation constant
    x = -1.0*((rc/Rm)**(-1.0/a))
    u = -a*(g-2.0)
    v = 1.0+ a*(g-b)
    z = ((-1.0+0j)**(a*(g-2.0)))*a*(rc**2)
    betainc = (((x+0j)**u)/u)*hyp2f1(u,1.0-v,u+1.0,x)
    k  = 1.0/np.abs(z*betainc)

    return k*Kernel(r,rc,a,b,g)

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
        #-------------- priors ----------------
        self.Prior_0    = st.norm(loc=centre_init[0],scale=hyp[0])
        self.Prior_1    = st.norm(loc=centre_init[1],scale=hyp[1])
        self.Prior_2    = st.uniform(loc=0,scale=np.pi)
        self.Prior_3    = st.halfcauchy(loc=0.01,scale=hyp[2])
        self.Prior_4    = st.halfcauchy(loc=0.01,scale=hyp[2])
        self.Prior_5    = st.truncexpon(b=hyp[3],loc=0.01,scale=hyp[4])
        self.Prior_6    = st.truncexpon(b=hyp[3],loc=0.01,scale=hyp[4])
        self.Prior_7    = st.truncexpon(b=1.99,loc=0.01,scale=hyp[4])
        print("GDP Elliptic module initialized")

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

    def LogLike(self,params,ndim,nparams):
        ctr = params[:2]
        dlt = params[2]
        rca = params[3]
        rcb = params[4]
        a   = params[5]
        b   = params[6]
        g   = params[7]
         #----- Checks if parameters' values are in the ranges
        if not Support(rca,rcb,a,b,g) : 
            return -1e50

        #------- Obtains radii and angles ---------
        radii,theta = RotRadii(self.cdts,ctr,self.Dist,dlt)

        rcs = (rca*rcb)/np.sqrt((rcb*np.cos(theta))**2+(rca*np.sin(theta))**2)

        ############### Radial likelihood ###################
        # Computes likelihood
        lf = (1.0-self.pro)*LikeField(radii,self.Rmax)
        lk = radii*(self.pro)*Kernel(radii,rcs,a,b,g)

        # Normalisation constant
        x = -1.0*((rcs/self.Rmax)**(-1.0/a))
        u = -a*(g-2.0)
        v = 1.0+ a*(g-b)
        z = ((-1.0+0j)**(a*(g-2.0)))*a*(rcs**2)
        betainc = (((x+0j)**u)/u)*hyp2f1(u,1.0-v,u+1.0,x)
        k  = 1.0/np.abs(z*betainc)

        llike_r  = np.sum(np.log((k*lk + lf)))
        # if not np.isfinite(llike_r):
        #     idn = np.where(np.isnan(llike_r))[0]
        #     print k[idn],betainc[idn],a,b,g
        #     sys.exit()
        #     return -1e50
        ##################### POISSON ###################################
        quarter  = cut(theta,bins=self.quadrants,include_lowest=True)
        counts   = value_counts(quarter)
        llike_t  = self.poisson.logpmf(counts).sum()
        ##################################################################

        llike = llike_t + llike_r
        # print(llike)
        return llike


