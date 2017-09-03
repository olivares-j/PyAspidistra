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
from scipy.optimize import brentq

#########################################################################################
from functools import partial
from scipy.stats import halfcauchy,lognorm,norm
from scipy.special import erf
dir_  = os.path.expanduser('~') +"/PyAspidistra/"
real  = True
Ntot  = 10000  # Number of stars if synthetic (real = False)
model = str(sys.argv[1])
Rmax  = float(sys.argv[2])

################################### MODEL ######################################################
if model == "EFF":
	from Models.EFF import Module,Number,Density,cdf
	#--------- Initial parameters --------------
	namepar = ["$r_c\ \ [pc]$","$\gamma$"]
	params  = [2.0,3.0]
	hyp     = np.array([10,10])
	rng     = None#[[0,hyp[0]],[0,hyp[1]]]
	#--------- arguments of logLike function

if model == "King":
	from Models.King import Module,Number,Density,cdf
	#--------- Initial parameters --------------
	namepar = ["$r_c\ \ [pc]$", "$r_t\ \ [pc]$"]
	params  = [2.0,20.0]
	hyp     = np.array([10,10])
	rng     = [[0,6],[10,30]]

if model == "RGDP1":
	from Models.RGDP1 import Module,Number,Density,cdf
	#--------- Initial parameters --------------
	namepar = ["$a$","$b$","$\gamma$"]
	params  = [0.5,2.0,0.0]
	hyp     = np.array([2,4])
	rng     = None#[[0,2],[3,40],[0,3]]
	
if model == "GDP":
	from Models.GDP import Module,Number,Density,cdf
	#--------- Initial parameters --------------
	namepar = ["$r_c\ \ [pc]$","$a$","$b$","$\gamma$"]
	params  = [2.0,0.5,2.0,0.1]
	hyp     = np.array([1,2,100])
	rng     = [[0,100],[0,2],[0,100],[0,1]]

if model == "RGDP2":
	from Models.RGDP2 import Module,Number,Density,cdf
	#--------- Initial parameters --------------
	namepar = ["$r_c\ \ [pc]$","$a$","$\gamma$"]
	params  = [4.0,0.5,0.1]
	hyp     = np.array([10,2])
	rng     = None#[[0,hyp[0]],[0,hyp[1]],[2,hyp[2]]]

if model == "GKing":
	from Models.GKing import Module,Number,Density,cdf
	#--------- Initial parameters --------------
	namepar = ["$r_c\ \ [pc]$","$r_t\ \ [pc]$","$a$","$b$"]
	params  = [2.0,20.0,0.5,2.0]
	hyp     = np.array([10,10,4,4])
	rng     = [[0,20],[0,40],[0,5],[0,5]]

if model == "OGKing":
	from Models.OGKing import Module,Number,Density,cdf
	#--------- Initial parameters --------------
	namepar = ["$r_c\ \ [pc]$","$r_t\ \ [pc]$"]
	params  = [2.0,20.0]
	hyp     = np.array([10,10])
	rng     = None#[[0,hyp[0]],[0,4*hyp[1]]]


if real :
	dir_out  = dir_+'Analysis/NoCentre/Samples/'+model+'_'+str(int(Rmax))
else:
	dir_out  = dir_+'Analysis/Synthetic/NoCentre/'+model+'_'+str(int(Rmax))+'_'+str(Ntot)
	fsyn     = dir_out+'/2-data.csv'
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
	cdtsT = np.array(data[:,[1,2,32]],dtype=np.float64)
	fdata = dir_+'Data/Members-0.84.csv'
	data  = np.array(read_csv(fdata,header=0,sep=','))
	cdtsD = np.array(data[:,[10,11,8]],dtype=np.float64)
	cdts  = np.vstack([cdtsT,cdtsD])
	#---- removes duplicateds --------
	sumc  = np.sum(cdts[:,:2],axis=1)
	idx   = np.unique(sumc,return_index=True)[1]
	cdts  = cdts[idx]
	sumc  = np.sum(cdts[:,:2],axis=1)
	if len(sumc) != len(list(set(sumc))):
		sys.exit("Duplicated entries in Coordinates!")
	#@################ Calculates Radii and PA ################
	radii = np.arccos(np.sin(cntr[1]*D2R)*np.sin(cdts[:,1]*D2R)+
                np.cos(cntr[1]*D2R)*np.cos(cdts[:,1]*D2R)*
                np.cos((cntr[0]-cdts[:,0])*D2R))*Dist
	theta = np.arctan2(np.sin((cdts[:,0]-cntr[0])*D2R),
                         np.cos(cntr[1]*D2R)*np.tan(cdts[:,1]*D2R)-
                         np.sin(cntr[1]*D2R)*np.cos((cdts[:,0]-cntr[0])*D2R))+np.pi

	cdts  = np.c_[cdts,radii,theta]

else :
	# --------------------
	# cdts     = np.array(read_csv(fsyn,header=0,sep=','))
	# ----- Creates synthetic data ---------
	unifsyn  = np.random.uniform(low=0.0,high=1.0,size=Ntot)
	radii    = np.empty(Ntot)
	for s,st in enumerate(unifsyn):
		radii[s]= brentq(lambda x:cdf(x,params,Rmax)-st,a=0.0,b=Rmax)
	theta    = np.random.uniform(low=0.0,high=2.*np.pi,size=Ntot)
	cdts     = np.empty((Ntot,5))
	cdts[:,0]= (radii/Dist)*np.cos(theta)*R2D + cntr[0]
	cdts[:,1]= (radii/Dist)*np.sin(theta)*R2D + cntr[1]
	cdts[:,2]= np.repeat(1,Ntot)
	cdts[:,3]= radii
	cdts[:,4]= theta
	np.savetxt(fsyn,cdts, delimiter=",")

	

idx        = np.where(cdts[:,3] <  Rmax)[0]
cdts       = cdts[idx] 
idx        = np.argsort(cdts[:,3])
cdts       = cdts[idx]
radii      = cdts[:,3]
theta      = cdts[:,4]

Rmax       = np.max(radii)
print("Maximum radius: ",Rmax)
print("Minimum radius: ",np.min(radii))

bins       = np.linspace(0,Rmax+0.1,51)

hist       = np.histogram(radii,bins=bins)[0]
bins       = bins[1:]
dr         = np.hstack([bins[0]/2,np.diff(bins)])
da         = 2*np.pi*bins*dr
dens       = np.empty((len(hist),2))
dens[:,0]  = hist/da
dens[:,1]  = np.sqrt(hist)/da
dens       = dens/np.sum(dens[:,0]*bins*dr)
Nr         = np.arange(len(radii))

#------------ Load Module -------
Module  = Module(cdts,Rmax,hyp,Dist)

#========================================================
#--------- Dimension, walkers and inital positions -------------------------
# number of dimensions our problem has
ndim     = len(namepar)
n_params = len(namepar)

pymultinest.run(Module.LogLike,Module.Priors, n_params,resume = False, verbose = True,#n_live_points=12,
	outputfiles_basename=dir_out+'/0-',multimodal=True, max_modes=2,sampling_efficiency = 'model',
	max_iter=10000)

# lets analyse the results

a = pymultinest.Analyzer(n_params = n_params, outputfiles_basename=dir_out+'/0-')
summary = a.get_stats()
samples = a.get_data()[:,2:]
np.savetxt(dir_out+'/2-foo.csv',samples, delimiter=",")

MAP   = np.array(summary["modes"][0]["maximum"])
# MAP   = np.array(summary["modes"][0]["maximum a posterior"])

samp = samples[np.random.choice(np.arange(len(samples)),size=100,replace=False)]
x = np.linspace(0.1,Rmax,50)

print()
print("-" * 30, 'ANALYSIS', "-" * 30)
print("Global Evidence:\n\t%.15e +- %.15e" % ( summary['nested sampling global log-evidence'], summary['nested sampling global log-evidence error'] ))


pdf = PdfPages(dir_out+'/Fit_'+model+'.pdf')
plt.figure()
for s,par in enumerate(samp):
	plt.plot(x,Number(x,par,Rmax,np.max(Nr)),lw=1,color="orange",alpha=0.2,zorder=1)
plt.fill_between(radii, Nr+np.sqrt(Nr), Nr-np.sqrt(Nr), facecolor='grey', alpha=0.5,zorder=2)
plt.plot(radii,Nr,lw=1,color="black",zorder=3)
plt.plot(x,Number(x,MAP,Rmax,np.max(Nr)), lw=1,color="red",zorder=4)
plt.ylim((0,1.1*max(Nr)))
plt.xlim((0,1.1*Rmax))
plt.ylabel("Number of stars")
plt.xlabel("Radius [pc]")
pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
plt.close()


for s,par in enumerate(samp):
	plt.plot(x,Density(x,par,Rmax),lw=1,color="orange",alpha=0.2,zorder=1)
plt.errorbar(bins,dens[:,0],yerr=dens[:,1],fmt="o",color="black",lw=1,ms=2,zorder=3)
plt.plot(x,Density(x,MAP,Rmax), linewidth=1,color="red",zorder=2)
plt.ylabel("Density [stars $\cdot$ pc$^{-2}$]")
plt.xlabel("Radius [pc]")
plt.ylim(1e-4,0.5)
plt.yscale("log")
pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
plt.close()

plt.figure()
n, bins, patches = plt.hist(theta,50, normed=1, facecolor='green', alpha=0.5)
plt.xlabel('Position Angle [radian]')
plt.ylabel('Density [stars $\cdot$ radian$^{-1}$]')
pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
plt.close()

if real :
	corner.corner(samples, labels=namepar,truths=MAP,truth_color="red",range=rng,
		reverse=False,plot_datapoints=False,fill_contours=True)
else:
	corner.corner(samples, labels=namepar,truths=params,truth_color="blue",range=rng,
		reverse=False,plot_datapoints=False,fill_contours=True)
pdf.savefig(bbox_inches='tight')
pdf.close()

# plt.clf()

# p = pymultinest.PlotMarginalModes(a)
# plt.figure(figsize=(5*n_params, 5*n_params))
# #plt.subplots_adjust(wspace=0, hspace=0)
# for i in range(n_params):
# 	plt.subplot(n_params, n_params, n_params * i + i + 1)
# 	p.plot_marginal(i, with_ellipses = True, with_points = False, grid_points=50)
# 	plt.ylabel("Probability")
# 	plt.xlabel(namepar[i])
	
# 	for j in range(i):
# 		plt.subplot(n_params, n_params, n_params * j + i + 1)
# 		#plt.subplots_adjust(left=0, bottom=0, right=0, top=0, wspace=0, hspace=0)
# 		p.plot_conditional(i, j, with_ellipses = False, with_points = True, grid_points=30)
# 		plt.xlabel(namepar[i])
# 		plt.ylabel(namepar[j])

# plt.savefig(dir_out+"/marginals_multinest.pdf", bbox_inches='tight')


print("Take a look at the pdf files in "+dir_out) 



 
