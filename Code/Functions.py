import numpy as np
from numba import jit
from sklearn.neighbors import NearestNeighbors as kNN
from scipy.spatial.distance import euclidean
D2R   = np.pi/180.
R2D   = 180./np.pi

@jit
def Deg2pc(cdts,cntr,Dist):

	radii = np.arccos(np.sin(cntr[1]*D2R)*np.sin(cdts[:,1]*D2R)+
	            np.cos(cntr[1]*D2R)*np.cos(cdts[:,1]*D2R)*
	            np.cos((cntr[0]-cdts[:,0])*D2R))*Dist
	theta = np.arctan2(np.sin((cdts[:,0]-cntr[0])*D2R),
	                     np.cos(cntr[1]*D2R)*np.tan(cdts[:,1]*D2R)-
	                     np.sin(cntr[1]*D2R)*np.cos((cdts[:,0]-cntr[0])*D2R))
	theta   = (theta + 2*np.pi)%(2*np.pi)
	return radii,theta

def TruncSort(cdts,r,t,Rcut):
	idx    = np.where(r <  float(Rcut))[0]
	r      = r[idx]
	t      = t[idx] 
	cdts   = cdts[idx]
	idx    = np.argsort(r)
	r      = r[idx]
	t      = t[idx] 
	cdts   = cdts[idx]
	Rmax   = np.max(r)
	print "Maximum radius: ",Rmax
	return cdts,r,t,Rmax

def DenNum(r,Rmax,nbins=41):
	bins       = np.linspace(0,Rmax+0.1,nbins)
	hist       = np.histogram(r,bins=bins)[0]
	bins       = bins[1:]
	dr         = np.hstack([bins[0]/2,np.diff(bins)])
	da         = 2*np.pi*bins*dr
	dens       = np.empty((len(hist),2))
	dens[:,0]  = hist/da
	dens[:,1]  = np.sqrt(hist)/da
	dens       = dens/np.sum(dens[:,0]*bins*dr)
	Nr         = np.arange(len(r))
	# bins       = bins - bins[0]

	# print np.sum(dens[:,0]*bins*dr)
	return Nr,bins,dens

@jit
def RotRadii(cdts,cntr,Dist,delta):
	#============== Obtains radii and pa ===================
    # Equation 2 from Kacharov 2014. Since we also infer the centre positon x0=y0=0
    x     = np.sin((cdts[:,0]-cntr[0])*D2R)*np.cos(cdts[:,1]*D2R)*Dist
    y     = (np.cos(cntr[1]*D2R)*np.sin(cdts[:,1]*D2R)-
           np.sin(cntr[1]*D2R)*np.cos(cdts[:,1]*D2R)*np.cos((cdts[:,0]-cntr[0])*D2R))*Dist

    #------- uncorrected --------------
    # x     = (cdts[:,0]-cntr[0])*D2R*self.Dist 
    # y     = (cdts[:,1]-cntr[1])*D2R*self.Dist

    xn = x*np.sin(delta) + y*np.cos(delta)
    yn = x*np.cos(delta) - y*np.sin(delta)
    r  = np.sqrt(xn**2 + yn**2)
    t  = np.arctan2(xn,yn)
    t  = (t + 2*np.pi)%(2*np.pi)

    return r,t

def fMAP(samples,nnn=25):
	nbrs   = kNN(n_neighbors=nnn).fit(samples)
	distances, indices = nbrs.kneighbors(samples)
	idx    = np.argmin(distances[:,-1])
	return samples[idx]

def fCovar(samples,MAP,sigma=68.27):
	n,ndim = samples.shape
	nt     = int(n*sigma/100)
	dist   = np.sqrt((samples-MAP)**2).sum(axis=1)
	idx    = np.arange(n)
	idxs   = np.argsort(dist)[:nt]
	idx    = idx[idxs]
	cov    = np.cov(samples[idx],rowvar=False)
	return cov

@jit
def Epsilon(c):
	return 1.0 - (c[1]/c[0])






