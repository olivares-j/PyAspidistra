import sys
import numpy as np
from numba import jit
from scipy.stats import dirichlet
from scipy.stats import halfcauchy

@jit
def Support(rc,g):
    if rc <= 0 : return -np.inf
    if g <= 2 : return -np.inf

@jit
def Number(r,rc,g,Nstr,Rmax):
    return Nstr*(CDF(r,rc,g)/CDF(Rmax,rc,g))

@jit
def CDF(r,rc,g): 
    return 1 - (rc**(g-2))*(rc**2 + r**2)**((2-g)/2)

@jit
def logPriors(rc,g,hyper):
    lp_rc = halfcauchy.logpdf(rc,loc=0,scale=hyper[0])
    lp_g  = halfcauchy.logpdf(g,loc=2,scale=hyper[1])
    return lp_rc+lp_g


@jit
def logLikeStar(r,p,rc,g,Rmax):
    return np.log(p*LikeCluster(r,rc,g,Rmax))


@jit
def LikeCluster(r,rc,g,Rmax):
    x  = 1+(r/rc)**2
    res= ((g-2)*(x**(-g/2)))/(rc**2)
    return res/CDF(Rmax,rc,g)

@jit
def LikeField(r,rm):
    return 2/rm**2


def LogPosterior(params,r,pro,Rmax,hyper):
    rc  = params[0]
    g   = params[1]
    #----- Checks if parameters' values are in the ranges
    supp = Support(rc,g)
    if supp == -np.inf : 
        return -np.inf

    # ----- Computes Priors ---------
    lprior = logPriors(rc,g,hyper)
    # ----- Computes Likelihoods ---------
    llike  = map(lambda x,y:logLikeStar(x,y,rc,g,Rmax),r,pro)

    lpos   = np.sum(llike)+lprior
    # print lpos
    return lpos



