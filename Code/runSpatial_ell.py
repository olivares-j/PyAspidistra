from __future__ import absolute_import, unicode_literals, print_function
import json
import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pymultinest
import math
import numpy as np
from pandas import read_csv
import os
import corner

from scipy.optimize import bisect
from scipy.stats import truncnorm as tnorm


#########################################################################################
from functools import partial
from scipy.stats import halfcauchy,lognorm,norm
from scipy.special import erf
dir_  = os.path.expanduser('~') +"/PyAspidistra/"
real  = False
Ntot  = 10000
model = str(sys.argv[1])
rcut  = float(sys.argv[2])

################################### MODEL ######################################################
nameparCtr = ["$x_0$","$y_0$","$\epsilon$","$\delta$"]
minsCtr    = np.array([-0.5,-0.5, 0.0,0.0])
maxsCtr    = np.array([ 0.5, 0.5, 1.0,2.0*np.pi])
paramsCtr  = np.array([0.0,0.0,0.1,np.pi/2.0])

if model == "EFF":
	from ModelsEll.EFF import Module,Number,Density
	#--------- Initial parameters --------------
	namepar = ["$r_c$","$\gamma$"]
	mins    = np.array([0,  2.0])
	maxs    = np.array([5.0,5.0])
	paramsR  = [3.0,3.0]

if model == "King":
	from ModelsEll.King import Module,Number,Density
	#--------- Initial parameters --------------
	namepar = ["$r_c$", "$r_t$"]
	mins    = np.array(([ 0.0, 10.0]))
	maxs    = np.array(([ 5.0, 50.0]))


if model == "MKing":
	from ModelsEll.MKing import Module,Number,Density
	#--------- Initial parameters --------------
	namepar = ["$r_c$","$r_t$","$a$","$b$"]
	mins    = np.array([ 0.01  ,10.0 , 0.01, 0.01])
	maxs    = np.array([ 500.0, 1000.0, 10.0, 50.0])
	#--------- arguments of logLike function

if model == "MKingRC":
	from ModelsEll.MKingRC import Module,Number,Density
	#--------- Initial parameters --------------
	namepar = ["$r_c$","$r_t$"]
	mins    = np.array([ 0.0  ,10.0])
	maxs    = np.array([ 10.0, 50.0])
	#--------- arguments of logLike function

if model == "GDP":
	from ModelsEll.GDP import Module,Number,Density
	#--------- Initial parameters --------------
	namepar = ["$r_c$","$a$","$b$","$\gamma$"]
	mins    = np.array([ 0.01  ,0.01,  0.01 , 0.01])
	maxs    = np.array([ 200.0, 2.0, 100.0, 2.0])
	#--------- arguments of logLike function

if model == "MGDP":
	from ModelsEll.MGDP import Module,Number,Density
	#--------- Initial parameters --------------
	namepar = ["$r_c$","$a$","$b$"]
	mins    = np.array([ 0.01  ,0.01, 0.01])
	maxs    = np.array([ 1000.0, 2.0, 100.0])
	#--------- arguments of logLike function

if model == "Centre":
	from Models.Centre import Module,Number,Density
	#--------- Initial parameters --------------
	namepar = nameparCtr
	support = np.vstack([minsCtr,maxsCtr]).T
	params  = paramsCtr
else:
	namepar = nameparCtr + namepar
	support = np.vstack([np.hstack([minsCtr,mins]),np.hstack([maxsCtr,maxs])]).T
	params  = np.hstack([paramsCtr,paramsR])

if real :
	dir_out  = dir_+'MultiNest/Samples/'+model+'_Ell_'+str(int(rcut))
else:
	dir_out  = dir_+'MultiNest/Samples/Synthetic/'+model+'_Ell_'+str(int(rcut))+'_'+str(Ntot)
if not os.path.exists(dir_out): os.mkdir(dir_out)

##################### DATA ##############################################################
Dist  = 136.0
D2R   = np.pi/180.
R2D   = 180./np.pi
cntr  = [56.65,24.13]
if real :
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
else :
	# ----- Creates synthetic data ---------
	unifsyn  = np.random.uniform(size=Ntot)
	rsyn     = np.array(map(lambda x:bisect(lambda r:Number(r,params,rcut)-x,0.0,rcut),unifsyn))
	tsyn     = np.random.uniform(low=0.0,high=2*np.pi,size=Ntot)
	xn_syn   = (rsyn/np.sqrt(1+(np.tan(tsyn)/(1-paramsCtr[2]))**2))*np.sign(np.random.uniform(size=Ntot) - 0.5)
	yn_syn   = xn_syn*np.tan(tsyn)
	# xn_syn   = np.random.uniform(low=-0.2*rcut,high=0.2*rcut,size=Ntot)
	# yn_syn   = np.sign(np.random.uniform(size=Ntot) - 0.5)*np.sqrt((rsyn**2  - xn_syn**2)*(1-paramsCtr[2])**2)
	xn_syn   = xn_syn/Dist
	yn_syn   = yn_syn/Dist
	x_syn    = xn_syn*np.sin(paramsCtr[3]) + yn_syn*np.cos(paramsCtr[3]) + paramsCtr[0]   # in radians
	y_syn    = xn_syn*np.cos(paramsCtr[3]) - yn_syn*np.sin(paramsCtr[3]) + paramsCtr[1]   # in radians
	cdts     = np.empty((Ntot,4))


	#------ small angle approxmation -----
	
	cdts[:,1]= cntr[1] + y_syn*R2D
	cdts[:,0]= cntr[0] + (x_syn/np.cos(cdts[:,1]*D2R))*R2D
	cdts[:,2]= np.repeat(1,Ntot)

	print("Synthetic data generated!")

#============== Cut on distance to cluster center ======================
radii     = np.arccos(np.sin(cntr[1]*D2R)*np.sin(cdts[:,1]*D2R)+
	            np.cos(cntr[1]*D2R)*np.cos(cdts[:,1]*D2R)*
	            np.cos((cntr[0]-cdts[:,0])*D2R))*Dist
idx       = np.where(radii <  rcut)[0]
cdts      = cdts[idx]
#========================================

#------------ Load Module -------
Module  = Module(cdts,rcut,support,Dist)
#========================================================
#--------- Dimension, walkers and inital positions -------------------------
# number of dimensions our problem has
ndim     = len(namepar)
n_params = len(namepar)

pymultinest.run(Module.LogLike,Module.Priors, n_params,resume = False, verbose = True,#n_live_points=12,
	outputfiles_basename=dir_out+'/0-',multimodal=True, max_modes=10,sampling_efficiency = 'model')

# lets analyse the results

a = pymultinest.Analyzer(n_params = n_params, outputfiles_basename=dir_out+'/0-')
s   = a.get_stats()
MAP = np.array(s['modes'][0]['maximum a posterior'])
a0,d0,epsi,delta,rc,gamma  = MAP[0],MAP[1],MAP[2],MAP[3],MAP[4],MAP[5]
samples = a.get_data()[:,2:]
#@################ Calculates Radii and PA ################

#---------- Synthetic --------
if not real :
	x     = np.sin((cdts[:,0]-(cntr[0]-params[0]))*D2R)*np.cos(cdts[:,1]*D2R)
	y     = (np.cos((cntr[1]-params[1])*D2R)*np.sin(cdts[:,1]*D2R)-
	        np.sin((cntr[1]-params[1])*D2R)*np.cos(cdts[:,1]*D2R)*np.cos((cdts[:,0]-(cntr[0]-params[0]))*D2R))

	x_new = x*np.sin(params[3]) + y*np.cos(params[3])
	y_new = x*np.cos(params[3]) - y*np.sin(params[3])

	radii = (np.sqrt((x_new*(1.0-params[2]))**2 + y_new**2)/(1.0-params[2]))*Dist

	idx        = np.argsort(radii)

	radii_syn  = np.array(radii[idx])

#---------------------------------------------
x     = np.sin((cdts[:,0]-(cntr[0]-a0))*D2R)*np.cos(cdts[:,1]*D2R)
y     = (np.cos((cntr[1]-d0)*D2R)*np.sin(cdts[:,1]*D2R)-
        np.sin((cntr[1]-d0)*D2R)*np.cos(cdts[:,1]*D2R)*np.cos((cdts[:,0]-(cntr[0]-a0))*D2R))

x_new = x*np.sin(delta) + y*np.cos(delta)
y_new = x*np.cos(delta) - y*np.sin(delta)

radii = (np.sqrt((x_new*(1.0-epsi))**2 + y_new**2)/(1.0-epsi))*Dist

# radii = np.arccos(np.sin(cntr[1]*D2R)*np.sin(self.cdts[:,1]*D2R)+
#         np.cos(cntr[1]*D2R)*np.cos(self.cdts[:,1]*D2R)*
#         np.cos((cntr[0]-self.cdts[:,0])*D2R))*self.Dist + 1e-20
# theta = np.arctan2(np.sin((cdts[:,0]-cntr[0])*D2R),
#                  np.cos(cntr[1]*D2R)*np.tan(cdts[:,1]*D2R)-
#                  np.sin(cntr[1]*D2R)*np.cos((cdts[:,0]-cntr[0])*D2R))
theta   = np.arctan2(x_new,y_new)

idx        = np.argsort(radii)

radii      = np.array(radii[idx])
theta      = np.array(theta[idx])
pro        = np.array(cdts[idx,2])

Rmax       = max(radii)
bins       = np.linspace(0,Rmax+0.1,101)


hist       = np.histogram(radii,bins=bins)[0]
bins       = bins[1:]
dr         = np.hstack([bins[0]/2,np.diff(bins)])
da         = 2*np.pi*bins*dr
densi      = hist/da
densi      = densi/sum(densi*bins*dr)
Nr         = np.cumsum(pro)

print()
print("-" * 30, 'ANALYSIS', "-" * 30)
print("Global Evidence:\n\t%.15e +- %.15e" % ( s['nested sampling global log-evidence'], s['nested sampling global log-evidence error'] ))

with PdfPages(dir_out+'/Fit_'+model+'.pdf') as pdf:

	plt.scatter(radii,Nr,s=1,color="black")
	plt.plot(radii,np.max(Nr)*Number(radii,MAP,Rmax), linewidth=1,color="red")
	if not real :
		plt.plot(radii_syn,np.max(Nr)*Number(radii_syn,params,Rmax), linewidth=1,color="blue")
	plt.ylim((0,1.1*max(Nr)))
	plt.xlim((0,1.1*rcut))
	pdf.savefig()  # saves the current figure into a pdf page
	plt.close()

	plt.scatter(bins,densi,s=1,color="black")
	plt.plot(radii,Density(radii,MAP,Rmax), linewidth=1,color="red")
	if not real:
		plt.plot(radii_syn,Density(radii_syn,params,Rmax), linewidth=1,color="blue")
	pdf.savefig()  # saves the current figure into a pdf page
	plt.close()

	# plt.figure()
	# n, bins, patches = plt.hist(theta,50, normed=1, facecolor='green', alpha=0.5)
	# plt.axvline(x=delta, ymin=0, ymax=1,color='red')
	# # if not real :
	# 	# plt.axvline(x=paramsCtr[3], ymin=0, ymax=1,color='blue')
	# plt.xlabel('Position Angle [rad]')
	# plt.ylabel('Density')
	# pdf.savefig()  # saves the current figure into a pdf page
	# plt.close()

	plt.figure()
	ax=plt.gca()
	plt.scatter(cdts[:,0],cdts[:,1],s=1,color="black")
	plt.xlabel('RA [deg]')
	plt.ylabel('Dec [deg]')
	plt.axes().set_aspect('equal', 'datalim')	
	ax.add_patch(Ellipse(xy=[cntr[0]+a0,cntr[1]+d0], width=(rc/Dist)*R2D, height=(1-epsi)*(rc/Dist)*R2D,
					 angle=450 - delta*R2D,fc='none',ec='red'))
	if not real :
		ax.add_patch(Ellipse(xy=[cntr[0]+params[0],cntr[1]+params[1]], width=(params[4]/Dist)*R2D, 
						height=(1-params[2])*(params[4]/Dist)*R2D, angle=450 - params[3]*R2D,fc='none',ec='blue'))
	pdf.savefig()  # saves the current figure into a pdf page
	plt.close()

	plt.scatter(theta,radii,s=1,color="black")
	plt.ylabel('Radius [pc]')
	plt.xlabel('$PA_0$  [rad]')
	# plt.axes().set_aspect('equal', 'datalim')	
	pdf.savefig()  # saves the current figure into a pdf page
	plt.close()

	if real :
		corner.corner(samples, labels=namepar,truths=MAP,truth_color="red")
	else:
		corner.corner(samples, labels=namepar,truths=params,truth_color="blue")
	pdf.savefig()
	plt.close()

plt.clf()

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

# plt.savefig(dir_out+"/marginals_multinest.pdf") #, bbox_inches='tight')


print("Take a look at the pdf files in "+dir_out) 



 
