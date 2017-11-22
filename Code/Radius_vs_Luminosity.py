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
fdance = "/pcdisk/kool5/scratch-lsb/DANCe-R2/DANCe-Pleiades-R2-ICRS.fits"
fout   = dir_+'Analysis/RadLumDistribution_DANCe.pdf'

Dist    = 180.0/np.pi
centre  = [56.65,24.13]
nbins   = 100

# reads fits files -----

dance = Table(fits.getdata(fdance,1))

# print dance.colnames
# sys.exit()

mag_labs =["Y [mag]","i-K [mag]","J [mag]","H [mag]","K [mag]"]
# extracts coordinates
cdts = np.c_[dance['RA'],dance['Dec'],dance['Y'],dance['i'],dance['J'],dance['H'],dance['K']]

cdts[:,3] = cdts[:,3] - cdts[:,6]

#---- removes duplicateds --------
sumc  = np.sum(cdts[:,:2],axis=1)
idx   = np.unique(sumc,return_index=True)[1]
cdts  = cdts[idx]
sumc  = np.sum(cdts[:,:2],axis=1)
if len(sumc) != len(list(set(sumc))):
	sys.exit("Duplicated entries in Coordinates!")

radii,theta = Deg2pc(cdts,centre,Dist)

#----- plot distributions ------


pdf = PdfPages(fout)
plt.figure()
for i in range(5):
	# compute 2D density PM
	X = np.c_[radii.copy(),cdts[:,2+i].copy()]
	X = X[np.where(np.sum(np.isfinite(X),axis=1)==2)[0]]

	Y, x_bins, y_bins = np.histogram2d(X[:,0], X[:,1],(nbins, nbins),normed=True)

	plt.imshow(Y.T, origin='lower', interpolation='nearest', aspect='auto',
	        extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]],
	        cmap="binary")
	plt.colorbar()
	plt.xlabel("Radius [$^\circ$]")
	plt.ylabel(mag_labs[i])
	pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
	plt.close()
pdf.close()






