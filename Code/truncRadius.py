
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
fout   = dir_+'Analysis/RadiiDistribution_Tycho+DANCe.pdf'

Dist    = 136.0
centre  = [56.65,24.13]
rc0     = 6 
rc1     = 11.5 # pc

# reads fits files -----
tycho = Table(fits.getdata(ftycho,1))
dance = Table(fits.getdata(fdance,1))


# extracts coordinates
cdtsT = np.c_[tycho['RAJ2000'],tycho['DEJ2000']]
cdtsD = np.c_[dance['RA'],dance['Dec']]
cdts  = np.vstack([cdtsT,cdtsD])
#---- removes duplicateds --------
sumc  = np.sum(cdts[:,:2],axis=1)
idx   = np.unique(sumc,return_index=True)[1]
cdts  = cdts[idx]
sumc  = np.sum(cdts[:,:2],axis=1)
if len(sumc) != len(list(set(sumc))):
	sys.exit("Duplicated entries in Coordinates!")

# transform to radii and position angles
cdtsU       = np.c_[np.random.uniform(centre[0]-rc0,centre[0]+rc0,int(2e6)),
					np.random.uniform(centre[1]-rc0,centre[1]+rc0,int(2e6))]


rad_us,the_us = Deg2pc(cdtsU,centre,Dist)
idc = np.where(rad_us < rc1)[0]

rad_uc = rad_us[idc]
the_uc = the_us[idc]



rad_T,the_T = Deg2pc(cdtsT,centre,Dist)
rad_D,the_D = Deg2pc(cdtsD,centre,Dist)
radii,theta = Deg2pc(cdts,centre,Dist)

#----- plot distributions ------

pdf = PdfPages(fout)
plt.figure()
n, bins, patches = plt.hist(the_us,100, normed=1,lw=2,alpha=0.8,
	histtype='step',label="Uniform Square ($\pm$"+str(rc0)+"$^\circ$)")
n, bins, patches = plt.hist(the_uc,100, normed=1,lw=2,alpha=0.8,
	histtype='step',label="Uniform Circle ("+str(rc1)+" pc)")
n, bins, patches = plt.hist(the_T,100, normed=1,lw=2,alpha=0.8,
	histtype='step',label="Tycho")
n, bins, patches = plt.hist(the_D,100, normed=1,lw=2,alpha=0.8,
	histtype='step',label="DANCe")
n, bins, patches = plt.hist(theta,100, normed=1,lw=2,alpha=0.8,ec="black",
	histtype='step',label="DANCe+Tycho")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=3, mode="expand", borderaxespad=0.)
plt.xlabel('Position Angle [radians]')
plt.ylabel('Density [stars $\cdot$ radians$^{-1}$]')
pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
plt.close()
plt.figure()
n, bins, patches = plt.hist(rad_us,100, normed=1, lw=2,alpha=0.8,
	histtype='step',label="Uniform Square ($\pm$"+str(rc0)+"$^\circ$)")
n, bins, patches = plt.hist(rad_uc,100, normed=1, lw=2,alpha=0.8,
	histtype='step',label="Uniform Circle ("+str(rc1)+" pc)")
n, bins, patches = plt.hist(rad_T,100, normed=1, lw=2,alpha=0.8,
	histtype='step',label="Tycho")
n, bins, patches = plt.hist(rad_D,100, normed=1,lw=2,alpha=0.8,
	histtype='step',label="DANCe")
n, bins, patches = plt.hist(radii,100, normed=1, lw=2,alpha=0.8,ec="black",
	histtype='step',label="DANCe+Tycho")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=3, mode="expand", borderaxespad=0.)
plt.xlabel('Radius [pc]')
plt.ylabel('Density [stars $\cdot$ pc$^{-1}$]')
pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
plt.close()
plt.figure()
n, bins, patches = plt.hist(rad_T,100, normed=1, lw=2,alpha=0.8,
	histtype='step',label="Tycho")
n, bins, patches = plt.hist(rad_D,100, normed=1,lw=2,alpha=0.8,
	histtype='step',label="DANCe")
n, bins, patches = plt.hist(radii,100, normed=1, lw=2,alpha=0.8,ec="black",
	histtype='step',label="DANCe+Tycho")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=3, mode="expand", borderaxespad=0.)
plt.vlines([6,11.5],0,1.1*np.max(n),colors="grey")
plt.xlabel('Radius [pc]')
plt.ylabel('Density [stars $\cdot$ pc$^{-1}$]')
pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
plt.close()
pdf.close()






