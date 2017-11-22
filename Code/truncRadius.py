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
import os
import numpy as np
import pandas as pn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from astropy.io import fits
from astropy.table import Table

from Functions import Deg2pc
#########################################################################################
dir_   = os.path.expanduser('~') +"/PyAspidistra/"
ftycho = "/pcdisk/kool5/scratch-lsb/DANCe-R2/DANCe-Pleiades-Tycho-R2.fits"
fdance = "/pcdisk/kool5/scratch-lsb/DANCe-R2/DANCe-Pleiades-R2-ICRS.fits"
fout   = dir_+'Analysis/RadiiDistribution_Tycho+DANCe_Jmag.pdf'


Dist    = 134.4
centre  = [56.65,24.13]
rc0     = 6 
rc1     = 11.5 # pc

d2pc    = (180.0/np.pi)*Dist

# reads fits files -----
tycho = Table(fits.getdata(ftycho,1))
dance = Table(fits.getdata(fdance,1))

# print tycho.colnames
# sys.exit()


# extracts coordinates
cdtsT = np.c_[tycho['RAJ2000'],tycho['DEJ2000'],tycho['Jmag']]
cdtsD = np.c_[dance['RA'],dance['Dec'],dance["J"]]
cdts  = np.vstack([cdtsT,cdtsD])
#---- removes duplicateds --------
sumc  = np.sum(cdts[:,:2],axis=1)
idx   = np.unique(sumc,return_index=True)[1]
cdts  = cdts[idx]
sumc  = np.sum(cdts[:,:2],axis=1)
if len(sumc) != len(list(set(sumc))):
	sys.exit("Duplicated entries in Coordinates!")

#----------- Uniform to compare with -----------
# cdtsU       = np.c_[np.random.uniform(centre[0]-rc0,centre[0]+rc0,int(2e6)),
# 					np.random.uniform(centre[1]-rc0,centre[1]+rc0,int(2e6))]
# rad_us,the_us = Deg2pc(cdtsU,centre,Dist)
# idc = np.where(rad_us < rc1)[0]
# rad_us = rad_us[idc]
# the_us = the_us[idc]



radii,theta = Deg2pc(cdts,centre,Dist)

id_14 = np.where(cdts[:,2] <= 14)[0]
id_18 = np.where(cdts[:,2] <= 18)[0]
id_19 = np.where(cdts[:,2] <= 19)[0]
id_20 = np.where(cdts[:,2] <= 20)[0]


rad_14,the_14 = Deg2pc(cdts[id_14,:],centre,Dist)
rad_18,the_18 = Deg2pc(cdts[id_18,:],centre,Dist)
rad_19,the_19 = Deg2pc(cdts[id_19,:],centre,Dist)
rad_20,the_20 = Deg2pc(cdts[id_20,:],centre,Dist)

pdf = PdfPages(fout)
plt.figure()
# n, bins, patches = plt.hist(the_us,100, normed=1,lw=1,alpha=0.8,ec="blue",
# 	histtype='step',label="Synthetic")
n, bins, patches = plt.hist(the_14,100, normed=1,lw=2,alpha=0.8,
	histtype='step',label="J<14")
n, bins, patches = plt.hist(the_18,100, normed=1,lw=2,alpha=0.8,
	histtype='step',label="J<18")
n, bins, patches = plt.hist(the_19,100, normed=1,lw=2,alpha=0.8,
	histtype='step',label="J<19")
n, bins, patches = plt.hist(the_20,100, normed=1,lw=2,alpha=0.8,
	histtype='step',label="J<20")
# n, bins, patches = plt.hist(theta,100, normed=1, lw=2,alpha=0.8,ec="black",
# 	histtype='step',label="All")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=4, mode="expand", borderaxespad=0.)
# plt.vlines([6,11.5],0,1.1*np.max(n),colors="grey")
plt.xlabel('Position Angle [rad]')
plt.ylabel('Density [stars $\cdot$ rad$^{-1}$]')
pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
plt.close()


# n, bins, patches = plt.hist(rad_us,150, normed=1,lw=2,alpha=0.8,ec="blue",
# 	histtype='step',label="Synthetic")
n, bins, patches = plt.hist(rad_14,150, normed=1,lw=2,alpha=0.8,
	histtype='step',label="J<14")
n, bins, patches = plt.hist(rad_18,150, normed=1, lw=2,alpha=0.8,
	histtype='step',label="J<18")
n, bins, patches = plt.hist(rad_19,150, normed=1,lw=2,alpha=0.8,
	histtype='step',label="J<19")
n, bins, patches = plt.hist(rad_20,150, normed=1,lw=2,alpha=0.8,
	histtype='step',label="J<20")
# n, bins, patches = plt.hist(radii,150, normed=1, lw=2,alpha=0.8,ec="black",
# 	histtype='step',label="All")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=4, mode="expand", borderaxespad=0.)
plt.vlines([rc1],0,0.5,colors="grey")
plt.ylim(0.01,0.2)
plt.xlim(0.0,15.0)
plt.xlabel('Radius [pc]')
plt.ylabel('Density [stars pc$^{-1}$]')
plt.yscale("log")
pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
plt.close()
pdf.close()







