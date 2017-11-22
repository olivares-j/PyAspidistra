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


@jit
def MassRj(rj):
	# From binney and tremain eq 8.106
	# and 3.84, values from bovy https://arxiv.org/abs/1610.07610
	a =  15.3  # km s-1 kpc-1
	b = -11.9  # km s-1 kpc-1
	Omg = (a-b)
	G = 4.3E-3                #pc Msol-1 (km/s)^2
	m = (4*a*Omg*(rj**3))/G #In solar masses
	# dif = rj-(G*m/(4*a*(a-b)))**(1.0/3.0)
	# print dif
	return m * 1e-6

@jit
def MassEps(rj,f):
	# from http://farside.ph.utexas.edu/teaching/336k/Newtonhtml/node111.html
	# Eq. 944
	e = np.sqrt(f*(2.0-f))
	m = (1.0/e)*(5.0/4.0)*MassRj(rj)
	return m







