from __future__ import absolute_import, unicode_literals, print_function
import json
import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import pymultinest
import math
import numpy as np
from pandas import read_csv
import os
import corner
from scipy.optimize import bisect

#########################################################################################
from functools import partial
from scipy.stats import halfcauchy,lognorm,norm
from scipy.special import erf
dir_  = os.path.expanduser('~') +"/PyAspidistra/"
real  = True
model = str(sys.argv[1])
rcut  = float(sys.argv[2])

################################### MODEL ######################################################
if model == "EFF":
	from Models.EFF import Module,Number,Density
	#--------- Initial parameters --------------
	namepar = ["$r_c$","$\gamma$"]
	mins    = np.array([0,  2.0])
	maxs    = np.array([10.0,10.0])
	params  = [3.0,3.0]
	#--------- arguments of logLike function

if model == "King":
	from Models.King import Module,Number,Density
	#--------- Initial parameters --------------
	namepar = ["$r_c$", "$r_t$"]
	mins    = np.array(([ 0.0, 5.0]))
	maxs    = np.array(([ 5.0, 50.0]))
	params  = [2.0,28.0]


if model == "MKing":
	from Models.MKing import Module,Number,Density
	#--------- Initial parameters --------------
	namepar = ["$r_c$","$r_t$","$a$","$b$"]
	mins    = np.array([ 0.01  ,5.0 , 0.01, 0.01])
	maxs    = np.array([ 1000.0, 1000.0, 5.0, 100.0])
	params  = [1.0,10.0,0.5,2.0]

if model == "M2King":
	from Models.MKing import Module,Number,Density
	#--------- Initial parameters --------------
	namepar = ["$r_c$","$r_t$","$a$"]
	mins    = np.array([ 0.01  ,5.0 , 0.01])
	maxs    = np.array([ 1000.0, 1000.0, 5.0])
	params  = [1.0,10.0,0.5]

if model == "GDP":
	from Models.GDP import Module,Number,Density
	#--------- Initial parameters --------------
	namepar = ["$r_c$","$a$","$b$","$\gamma$"]
	mins    = np.array([ 0.01  ,0.01,  0.01 , 0.01])
	maxs    = np.array([ 100.0  , 2.0, 50.0, 2.0])
	params  = [100.0,1,25.0,0.0]

if model == "MGDP":
	from Models.MGDP import Module,Number,Density
	#--------- Initial parameters --------------
	namepar = ["$r_c$","$a$","$b$"]
	mins    = np.array([ 0.01  ,0.01, 0.01])
	maxs    = np.array([ 100.0, 2.0, 50.0])
	params  = [1.0,0.5,2.0]

support = np.vstack([mins,maxs]).T

if real :
	dir_out  = dir_+'MultiNest/Samples/'+model+'_'+str(int(rcut))
else:
	dir_out  = dir_+'MultiNest/Samples/Synthetic/'+model+'_'+str(int(rcut))
if not os.path.exists(dir_out): os.mkdir(dir_out)

##################### DATA ##############################################################
Dist  = 136.0
D2R   = np.pi/180.
R2D   = 180./np.pi
cntr  = [56.65,24.13]
if real :
	#------- reads data ---------------
	fdata = dir_+'Data/OnlyTycho.csv'
	data  = np.array(read_csv(fdata,header=0,sep=','))
	cdtsT = np.array(data[:,[1,2,32]],dtype=np.float32)
	fdata = dir_+'Data/Members-0.84.csv'
	data  = np.array(read_csv(fdata,header=0,sep=','))
	cdtsD = np.array(data[:,[10,11,8]],dtype=np.float32)
	cdts  = np.vstack([cdtsT,cdtsD])
	#---- removes duplicateds --------
	sumc  = np.sum(cdts[:,:2],axis=1)
	idx   = np.unique(sumc,return_index=True)[1]
	cdts  = cdts[idx]
	sumc  = np.sum(cdts[:,:2],axis=1)
	if len(sumc) != len(list(set(sumc))):
		sys.exit("Duplicated entries in Coordinates!")
	pro   = cdts[:,2]
else :
	# ----- Creates synthetic data ---------
	Ntot     = 2000
	unifsyn  = np.random.uniform(size=Ntot)
	rsyn     = np.array(map(lambda x:bisect(lambda r:Number(r,params,rcut)-x,0.0,rcut),unifsyn))
	thetasyn = np.random.uniform(low=0.0,high=2.*np.pi,size=Ntot)
	cdts     = np.vstack([((rsyn/Dist)*np.cos(thetasyn)*R2D + cntr[0]).T,((rsyn/Dist)*np.sin(thetasyn)*R2D + cntr[1]).T]).T
	# cdts   = np.random.multivariate_normal([0.0,0.0],[[4.0, 0.0],[0.0, 4.0]],Ntot)
	pro   = np.repeat(1,Ntot)

#@################ Calculates Radii and PA ################

radii = np.arccos(np.sin(cntr[1]*D2R)*np.sin(cdts[:,1]*D2R)+
                np.cos(cntr[1]*D2R)*np.cos(cdts[:,1]*D2R)*
                np.cos((cntr[0]-cdts[:,0])*D2R))*Dist
theta = np.arctan2(np.sin((cdts[:,0]-cntr[0])*D2R),
                         np.cos(cntr[1]*D2R)*np.tan(cdts[:,1]*D2R)-
                         np.sin(cntr[1]*D2R)*np.cos((cdts[:,0]-cntr[0])*D2R))+np.pi

idx        = np.where(radii <  rcut)[0]
radii      = np.array(radii[idx])
theta      = np.array(theta[idx])
pro        = np.array(pro[idx])

Rmax       = max(radii)
bins       = np.linspace(0,Rmax+0.1,101)
idx        = np.argsort(radii)
radii      = np.array(radii[idx])
theta      = np.array(theta[idx])
pro        = np.array(pro[idx])

hist       = np.histogram(radii,bins=bins)[0]
bins       = bins[1:]
dr         = np.hstack([bins[0]/2,np.diff(bins)])
da         = 2*np.pi*bins*dr
densi      = hist/da
densi      = densi/sum(densi*bins*dr)
Nr         = np.cumsum(pro)

#------------ Load Module -------
Module  = Module(radii,pro,Rmax,support)

#========================================================
#--------- Dimension, walkers and inital positions -------------------------
# number of dimensions our problem has
ndim     = len(namepar)
n_params = len(namepar)

pymultinest.run(Module.LogLike,Module.Priors, n_params,resume = False, verbose = True,#n_live_points=12,
	outputfiles_basename=dir_out+'/2-',multimodal=True, max_modes=10,sampling_efficiency = 'model')

# lets analyse the results

a = pymultinest.Analyzer(n_params = n_params, outputfiles_basename=dir_out+'/2-')
s   = a.get_stats()
MAP = np.array(s['modes'][0]['maximum a posterior'])
samples = a.get_data()[:,2:]

print()
print("-" * 30, 'ANALYSIS', "-" * 30)
print("Global Evidence:\n\t%.15e +- %.15e" % ( s['nested sampling global log-evidence'], s['nested sampling global log-evidence error'] ))


with PdfPages(dir_out+'/Fit.pdf') as pdf:

	plt.scatter(radii,Nr,s=1,color="black")
	plt.plot(radii,np.max(Nr)*Number(radii,MAP,Rmax), linewidth=1,color="red")
	plt.ylim((0,1.1*max(Nr)))
	plt.xlim((0,1.1*rcut))
	pdf.savefig()  # saves the current figure into a pdf page
	plt.close()

	plt.scatter(bins,densi,s=1,color="black")
	plt.plot(radii,Density(radii,MAP,Rmax), linewidth=1,color="red")
	pdf.savefig()  # saves the current figure into a pdf page
	plt.close()

	plt.figure()
	n, bins, patches = plt.hist(theta,50, normed=1, facecolor='green', alpha=0.5)
	plt.xlabel('Position Angle')
	plt.ylabel('Density')
	pdf.savefig()  # saves the current figure into a pdf page
	plt.close()

	corner.corner(samples, labels=namepar)
	pdf.savefig()
	plt.close()

plt.clf()

p = pymultinest.PlotMarginalModes(a)
plt.figure(figsize=(5*n_params, 5*n_params))
#plt.subplots_adjust(wspace=0, hspace=0)
for i in range(n_params):
	plt.subplot(n_params, n_params, n_params * i + i + 1)
	p.plot_marginal(i, with_ellipses = True, with_points = False, grid_points=50)
	plt.ylabel("Probability")
	plt.xlabel(namepar[i])
	
	for j in range(i):
		plt.subplot(n_params, n_params, n_params * j + i + 1)
		#plt.subplots_adjust(left=0, bottom=0, right=0, top=0, wspace=0, hspace=0)
		p.plot_conditional(i, j, with_ellipses = False, with_points = True, grid_points=30)
		plt.xlabel(namepar[i])
		plt.ylabel(namepar[j])

plt.savefig(dir_out+"/marginals_multinest.pdf") #, bbox_inches='tight')


print("Take a look at the pdf files in "+dir_out) 



 
