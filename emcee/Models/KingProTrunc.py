import sys
import numpy as np
from numba import jit
from scipy.stats import dirichlet,halfcauchy

@jit
def Support(rc,rt):
    if rc <= 0 : return -np.inf
    if rt <= 0 or rt <= rc : return -np.inf

@jit
def CDFv(r,rc,rt):
    w = r**2 + rc**2
    x = 1 + (rt/rc)**2
    y = 1 + (r/rc)**2
    z = rc**2 + rt**2
    a = (r**2)/z +  4*(rc-np.sqrt(w))/np.sqrt(z) + np.log(w) -2*np.log(rc)
    b = -4 + (rt**2)/z + 4*rc/np.sqrt(z)         + np.log(z) -2*np.log(rc)
    NK  = a/b
    result = np.zeros(len(r))
    idBad  = np.where(r>rt)[0]
    idOK   = np.where(r<=rt)[0]
    result[idOK] = NK[idOK]
    result[idBad] = 1
    return result

@jit
def CDFs(r,rc,rt):
    if r > rt:
        return 1
    w = r**2 + rc**2
    x = 1 + (rt/rc)**2
    y = 1 + (r/rc)**2
    z = rc**2 + rt**2
    a = (r**2)/z +  4*(rc-np.sqrt(w))/np.sqrt(z) + np.log(w) -2*np.log(rc)
    b = -4 + (rt**2)/z + 4*rc/np.sqrt(z)         + np.log(z) -2*np.log(rc)
    NK  = a/b
    return NK

@jit
def Number(r,rc,rt,Nstr,Rmax):
    return Nstr*(CDFv(r,rc,rt)/CDFs(Rmax,rc,rt))


@jit
def logPriors(rc,rt,hyper):
    lp_rc = halfcauchy.logpdf(rc,loc=0,scale=hyper[0])
    lp_rt = halfcauchy.logpdf(rt,loc=0,scale=hyper[1])
    return lp_rc+lp_rt


@jit
def logLikeStar(r,p,rc,rt,Rmax):
    return np.log(p*LikeCluster(r,rc,rt,Rmax))


@jit
def LikeKing(r,rc,rt):
    x = 1 + (r/rc)**2
    y = 1 + (rt/rc)**2
    z = rc**2 + rt**2
    k   = 2*((x**(-0.5))-(y**-0.5))**2
    norm= (rc**2)*(-4 + (rt**2)/z + 4*rc/np.sqrt(z) + np.log(z) -2*np.log(rc))
    lik = k/norm
    return lik

#@jit
def LikeCluster(r,rc,rt,Rmax):
    return np.piecewise(r,[r>rt,r<=rt],[0.000001,lambda x: LikeKing(x,rc,rt)/CDFs(Rmax,rc,rt)])

@jit
def LikeField(r,rm):
    return 2/rm**2


def LogPosterior(params,r,pro,Rmax,hyper):
    rc  = params[0]
    rt  = params[1]
    #----- Checks if parameters' values are in the ranges
    supp = Support(rc,rt)
    if supp == -np.inf : 
        return -np.inf

    # ----- Computes Priors ---------
    lprior = logPriors(rc,rt,hyper)
    # ----- Computes Likelihoods ---------
    llike  = map(lambda x,y:logLikeStar(x,y,rc,rt,Rmax),r,pro)

    lpos   = np.sum(llike)+lprior
    # print lpos
    return lpos



