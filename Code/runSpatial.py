#!/usr/local/python-2.7.10/bin/python
from __future__ import absolute_import, unicode_literals, print_function
import sys
sys.path.insert(0, "/pcdisk/boneym5/jolivares/PyAspidistra/Code")
import os
import numpy as np
import importlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pymultinest
import corner

from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from scipy.optimize import brentq
from Functions import Deg2pc,TruncSort,DenNum,RotRadii,fMAP,fCovar

#########################################################################################
dir_  = os.path.expanduser('~') +"/PyAspidistra/"
real  = True
Ntot  = 10000  # Number of stars if synthetic (real = False)

nargs = len(sys.argv)
if nargs == 4:
	na = 0
if nargs == 3:
	na = 1
model = str(sys.argv[1-na])
Rcut  = int(sys.argv[2-na])
exte  = str(sys.argv[3-na])

if exte == "None":
	dext = "Models"
	fext = "NoCentre/"
elif exte == "Ctr":
	dext = "ModelsCtr"
	fext = "Centre/"
elif exte == "Ell":
	dext = "ModelsEll"
	fext = "Elliptic/"
elif exte == "Seg":
	dext = "ModelsSeg"
	fext = "Segregated/"
else :
	print("Extention not recognised!")
	print("Available ones are: None, Centre (Ctr), Elliptic (Ell), and Segregated (Seg")
	sys.exit()

########### Load module ######################
mod = importlib.import_module(dext+"."+model)
##############################################

############## Directory of outputs #########################
if real :
	dir_out  = dir_+'Analysis/'+fext+model+'_'+str(Rcut)
else:
	dir_out  = dir_+'Analysis/Synthetic/'+fext+model+'_'+str(Rcut)+'_'+str(Ntot)
	fsyn     = dir_out+'/2-data.csv'
if not os.path.exists(dir_out): os.mkdir(dir_out)
################################### MODEL ######################################################
if model == "EFF":
	#--------- Initial parameters --------------
	namepar = ["$r_c$ [pc]","$\gamma$"]
	params  = [2.0,3.0]
	hyp     = np.array([10,10])
	rng     = [[0,10],[0,10]]


if model == "King":
	#--------- Initial parameters --------------
	namepar = ["$r_c$ [pc]", "$r_t$ [pc]"]
	params  = [2.0,20.0]
	hyp     = np.array([10,10])
	rng     = [[0,5],[0,50]]

if model == "GKing":
	#--------- Initial parameters --------------
	namepar = ["$r_c$ [pc]","$r_t$ [pc]","$\\alpha$","$\\beta$"]
	params  = [2.0,20.0,0.5,2.0]
	hyp     = np.array([10,10,4,4])
	rng     = [[0,20],[0,50],[0,5],[0,5]]

if model == "OGKing":
	#--------- Initial parameters --------------
	namepar = ["$r_c$ [pc]","$r_t$ [pc]"]
	params  = [2.0,20.0]
	hyp     = np.array([10,10])
	rng     = [[0,10],[0,50]]

if model == "GDP":
	#--------- Initial parameters --------------
	namepar = ["$r_c$ [pc]","$\\alpha$","$\\beta$","$\gamma$"]
	params  = [2.0,0.5,2.0,0.1]
	hyp     = np.array([10,2,100,2])
	rng     = [[0,50],[0,2],[0,100],[0,1]]

if model == "RGDP":
	#--------- Initial parameters --------------
	namepar = ["$r_c$ [pc]","$\\alpha$","$\\beta$"]
	params  = [2.0,0.5,2.0]
	hyp     = np.array([10,2,100])
	rng     = [[0,50],[0,2],[0,100]]


# if model == "RGDP1":
# 	#--------- Initial parameters --------------
# 	namepar = ["$a$","$b$","$\gamma$"]
# 	params  = [0.5,2.0,0.0]
# 	hyp     = np.array([2,4])
# 	rng     = [[0,2],[3,40],[0,3]]
	
# if model == "RGDP2":
# 	#--------- Initial parameters --------------
# 	namepar = ["$r_c\ \ [pc]$","$a$","$\gamma$"]
# 	params  = [4.0,0.5,0.1]
# 	hyp     = np.array([10,2])
# 	rng     = [[0,100],[0,2],[0,1]]

######### parameters of centre ####################
nameparCtr = ["$\\alpha_c\ \ [^\circ]$","$\\delta_c\ \ [^\circ]$"]
hypCtr    = np.array([1.0,1.0])
rngCtr    = [[56,57],[23,25]]

######## parameter of elliptic ############################
if model == "King" and exte == "Ell":
	#--------- Initial parameters --------------
	namepar = ["$\phi$ [radians]","$r_{ca}$ [pc]","$r_{ta}$ [pc]",
				 "$r_{cb}$ [pc]","$r_{tb}$ [pc]"]
	params  = [np.pi/4,2.0,20.0,2.0,20.0]
	hyp     = np.array([10,10])
	rng     = [[-3,3],[0,5],[0,50],[0,5],[0,50]]

if model == "GKing" and exte == "Ell":
	#--------- Initial parameters --------------
	namepar = ["$\phi$ [radians]","$r_{ca}$ [pc]","$r_{ta}$ [pc]", 
				"$r_{cb}$ [pc]","$r_{tb}$ [pc]","$\\alpha$","$\\beta$"]
	params  = [np.pi/4,2.0,20.0,2.0,20.0,0.5,2.0]
	hyp     = np.array([10,10,10,10,4,4])
	rng     = [[-3,3],[0,20],[0,50],[0,20],[0,50],[0,5],[0,5]]

if model == "OGKing" and exte == "Ell":
	#--------- Initial parameters --------------
	namepar = ["$\phi$ [radians]","$r_{ca}$ [pc]","$r_{ta}$ [pc]$",
				 "$r_{cb}$ [pc]","$r_{tb}$ [pc]"]
	params  = [np.pi/4,2.0,20.0,2.0,20.0]
	hyp     = np.array([10,10,10,10])
	rng     = [[-3,3],[0,20],[0,50],[0,20],[0,50]]

if model == "RGDP" and exte == "Ell":
	#--------- Initial parameters --------------
	namepar = ["$\phi$ [radians]","$r_{ca}$ [pc]","$r_{cb}$ [pc]",
				"$\\alpha$","$\\beta$"]
	params  = [np.pi/4,2.0,2.0,0.5,2.0]
	hyp     = np.array([10,2,100])
	rng     = [[-3,3],[0,50],[0,50],[0,2],[0,100]]

if model == "GDP" and exte == "Ell":
	#--------- Initial parameters --------------
	namepar = ["$\phi$ [radians]","$r_{ca}$ [pc]","$r_{cb}$ [pc]",
				"$\\alpha$","$\\beta$","$\gamma$"]
	params  = [np.pi/4,2.0,2.0,0.5,2.0,0.0]
	hyp     = np.array([10,2,100,2])
	rng     = [[-3,3],[0,50],[0,50],[0,2],[0,100],[0,2]]

#####################################################

#------- concatenate parameters
if not exte == "None":
	namepar = nameparCtr + namepar
	hyp     = np.append(hypCtr,hyp)
	rng     = sum([rngCtr,rng],[])
##################### DATA ##############################################################
Dist    = 136.0
centre  = [56.65,24.13]
if real :
	#------- reads data ---------------
	fdata = dir_+'Data/OnlyTycho.csv'
	data  = np.array(pd.read_csv(fdata,header=0,sep=','))
	cdtsT = np.array(data[:,[1,2,32]],dtype=np.float64)
	fdata = dir_+'Data/Members-0.84.csv'
	data  = np.array(pd.read_csv(fdata,header=0,sep=','))
	cdtsD = np.array(data[:,[10,11,8]],dtype=np.float64)
	cdts  = np.vstack([cdtsT,cdtsD])
	#---- removes duplicateds --------
	sumc  = np.sum(cdts[:,:2],axis=1)
	idx   = np.unique(sumc,return_index=True)[1]
	cdts  = cdts[idx]
	sumc  = np.sum(cdts[:,:2],axis=1)
	if len(sumc) != len(list(set(sumc))):
		sys.exit("Duplicated entries in Coordinates!")
else :
	# -------- Check if file exists -------
	if os.path.exists(fsyn):
		cdts     = np.array(pd.read_csv(fsyn,header=0,sep=','))
	else:
		# ----- Creates synthetic data ---------
		unifsyn  = np.random.uniform(low=0.0,high=1.0,size=Ntot)
		radii    = np.empty(Ntot)
		for s,st in enumerate(unifsyn):
			radii[s]= brentq(lambda x:mod.cdf(x,params,float(Rcut))-st,a=0.0,b=float(Rcut))
		theta    = np.random.uniform(low=0.0,high=2.*np.pi,size=Ntot)
		cdts     = np.empty((Ntot,3))
		cdts[:,0]= (radii/Dist)*np.cos(theta)*R2D + centre[0]
		cdts[:,1]= (radii/Dist)*np.sin(theta)*R2D + centre[1]
		cdts[:,2]= np.repeat(1,Ntot)
		radii = None
		theta = None
		np.savetxt(fsyn,cdts, delimiter=",")

################# Calculates Radii and PA ################
#------------ Load Module -------
Module  = mod.Module(cdts,Rcut,hyp,Dist,centre)

#========================================================
#--------- Dimension, walkers and inital positions -------------------------
# number of dimensions our problem has
ndim     = len(namepar)
n_params = len(namepar)

pymultinest.run(Module.LogLike,Module.Priors, n_params,resume = False, verbose = True,#n_live_points=12,
	outputfiles_basename=dir_out+'/0-',multimodal=True, max_modes=2,sampling_efficiency = 'model',
	max_iter=10000)

# analyse the results
ana = pymultinest.Analyzer(n_params = n_params, outputfiles_basename=dir_out+'/0-')
summary = ana.get_stats()
samples = ana.get_data()[:,2:]
np.savetxt(dir_out+'/0-foo.csv',samples, delimiter=",")

# MAP   = np.array(summary["modes"][0]["maximum"])
# MAP   = np.array(summary["modes"][0]["maximum a posterior"])
# MAP   = np.array(summary["modes"][0]["mean"])

MAP   = fMAP(samples)
print()
print("-" * 30, 'ANALYSIS', "-" * 30)
print("Global Evidence:\n\t%.15e +- %.15e" % (summary['nested sampling global log-evidence'], 
	summary['nested sampling global log-evidence error'] ))

########################## PLOTS #########################
# ----- select only 100 parameters to plot
samp = samples[np.random.choice(np.arange(len(samples)),size=100,replace=False)]



# ------ establish new centre (if needed) ------
if not exte == "None":
	centre = MAP[:2]
# -------- prepare data for plots ---------
rad,thet              = Deg2pc(cdts,centre,Dist)
cdts,radii,theta,Rmax = TruncSort(cdts,rad,thet,Rcut)

if exte == "Ell":
	radii,theta       = RotRadii(cdts,centre,Dist,MAP[3])

Nr,bins,dens          = DenNum(radii,Rmax)
x                     = np.linspace(0.1,Rmax,50)


# print(mod.Density(x,samp[0],Rmax))
# sys.exit()
pdf = PdfPages(dir_out+"/"+model+'_fit.pdf')

if Rcut < 10:
	lmin_den = 4e-3
else:
	lmin_den = 1e-4
plt.figure()
for s,par in enumerate(samp):
	plt.plot(x,mod.Number(x,par,Rmax,np.max(Nr)),lw=1,color="orange",alpha=0.2,zorder=1)
plt.fill_between(radii, Nr+np.sqrt(Nr), Nr-np.sqrt(Nr), facecolor='grey', alpha=0.5,zorder=2)
plt.plot(radii,Nr,lw=1,color="black",zorder=3)
plt.plot(x,mod.Number(x,MAP,Rmax,np.max(Nr)), lw=1,color="red",zorder=4)
plt.ylim((0,1.1*max(Nr)))
plt.xlim((0,1.1*Rmax))
plt.ylabel("Number of stars")
plt.xlabel("Radius [pc]")
pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
plt.close()


for s,par in enumerate(samp):
	plt.plot(x,mod.Density(x,par,Rmax),lw=1,color="orange",alpha=0.2,zorder=1)
plt.errorbar(bins,dens[:,0],yerr=dens[:,1],fmt="o",color="black",lw=1,ms=2,zorder=3)
plt.plot(x,mod.Density(x,MAP,Rmax), linewidth=1,color="red",zorder=2)
plt.ylabel("Density [stars $\cdot$ pc$^{-2}$]")
plt.xlabel("Radius [pc]")
plt.ylim(lmin_den,0.5)
plt.yscale("log")
pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
plt.close()

plt.figure()
n, bins, patches = plt.hist(theta,50, normed=1, facecolor='green', alpha=0.5)
plt.xlabel('Position Angle [radians]')
plt.ylabel('Density [stars $\cdot$ radians$^{-1}$]')
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

############### Stores the MAP ##############

np.savetxt(dir_out+"/"+model+"_map.txt",MAP.reshape(1,len(MAP)),
		fmt=str('%2.3f'),delimiter="\t")

############### Stores the covariance matrix ##############
print("Finding covariance matrix around MAP ...")
covar = fCovar(samples,MAP)
np.savetxt(dir_out+"/"+model+"_covariance.txt",covar,
		fmt=str('%2.3f'),delimiter=" & ",newline=str("\\\ \n"))

print("Take a look at the pdf files in "+dir_out) 








 
