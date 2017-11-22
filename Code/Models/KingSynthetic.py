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
from scipy.stats import dirichlet,halfcauchy

"""
All objects should be less than Rm
Rm must be lower or equal to Rt
"""

@jit
def Support(rc,rt):
    if rc <= 0  : return -np.inf
    if rt <= rc : return -np.inf

@jit
def cdfKing(r,rc,rt,Rm):
    w = rc**2 +  r**2 
    y = rc**2 + Rm**2
    z = rc**2 + rt**2
    a = (r**2)/z  +  4*(rc-np.sqrt(w))/np.sqrt(z) + np.log(w) - 2*np.log(rc)
    b = (Rm**2)/z +  4*(rc-np.sqrt(y))/np.sqrt(z) + np.log(y) - 2*np.log(rc)
    return a/b

@jit
def Number(r,rc,rt,Rm,Nstr):
    # Rm must be less or equal to rt
    return Nstr*cdfKing(r,rc,rt,Rm)

@jit
def logPriors(rc,rt,hyper):
    lp_rc = halfcauchy.logpdf(rc,loc=0,scale=hyper[0])
    lp_rt = halfcauchy.logpdf(rt,loc=0,scale=hyper[1])
    return lp_rc+lp_rt


@jit
def King(r,rc,rt):
    # King kernel
    a = 1.0 + (r/rc)**2.0
    b = 1.0 + (rt/rc)**2.0
    krnl = ((a**(-0.5))-(b**(-0.5)))**2.0
    return krnl

def LogPosterior(params,r,Rm,hyper):
    rc  = params[0]
    rt  = params[1]
    #----- Checks if parameters' values are in the ranges
    supp = Support(rc,rt)
    if supp == -np.inf : 
        return -np.inf

    # ----- Computes Priors ---------
    lprior = logPriors(rc,rt,hyper)
    # ----- Computes Likelihoods ---------
    # BNormalisation constant
    w = rc**2 +  r**2 
    y = rc**2 + Rm**2
    z = rc**2 + rt**2
    a = (Rm**2)/z +  4*(rc-np.sqrt(y))/np.sqrt(z) + np.log(y) - 2*np.log(rc) # Truncated at Rm
    # a = (rt**2)/z - 4.0 +  4.0*(rc/np.sqrt(z)) + np.log(z) - 2*np.log(rc)  # Truncated at Rt (i.e. No truncated)
    k  = 2.0/(a*(rc**2.0))

    # Obtain log likelihood
    llike  = np.sum(np.log(k*r*King(r,rc,rt)))

    lpos   = llike+lprior
    # print lpos
    return lpos



