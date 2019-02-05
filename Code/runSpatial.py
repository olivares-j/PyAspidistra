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
from __future__ import absolute_import, unicode_literals, print_function
import sys
import os
import numpy as np
import importlib
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import pymultinest
import corner
import math

from astropy import units as u
from astropy.coordinates import SkyCoord

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Ellipse
import pandas as pd
from scipy.optimize import brentq
import scipy.stats as st
from Functions import *

from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=18, usetex=True)

nargs = len(sys.argv)
if nargs == 4:
	na = 0
if nargs == 3:
	na = 1
model = str(sys.argv[1-na])
Rcut  = float(sys.argv[2-na])
exte  = str(sys.argv[3-na])

############ GLOBAL VARIABLES #################################################
dir_  = "/home/jromero/Repos/PyAspidistra/"  # Specify the full path to the Aspidistra path
real  = True
Ntot  = 10000  # Number of stars if synthetic (real = False)

Dist    = 309
centre  = [289.10,-16.38]

dir_analysis = dir_ + "Analysis/"

file_members = dir_+'Data/'+'members_ALL.csv' # Put here the name of the members file.
#"Make sure you have data in right order! (R.A., Dec. Probability, Band,e_band)")
list_observables = ["RAJ2000","DEJ2000","probability","G","G_error"]

mag_limit = 25.0
pro_limit = 0.5


#-------------------- MODEL PRIORS -----------------------------------------------
texp = 100.0 # truncation of exponential prior for exponents
sexp = 1.0  # scale of  ""      """
src  = 1.0  # scale of half-cauchy for core radius
srt  = 10.0 #   ""        """            tidal radius


if model == "EFF":
	if exte == "None" or exte =="Ctr":
		#----------------------------------
		namepar = ["$r_c$ [pc]","$\gamma$"]
		params  = [2.0,3.0]
		rng     = [[0,4],[2,4]]
	if exte == "Ell" or exte =="Seg":
		#--------- Initial parameters --------------
		namepar = ["$\phi$ [radians]","$r_{ca}$ [pc]","$r_{cb}$ [pc]","$\gamma$"]
		params  = [np.pi/4,5.0,1.0,3.0]
		rng     = [[0,np.pi],[0,10],[0,5],[2,4]]
		id_rc   = [3,4]
	hyp     = np.array([src,texp,sexp])

if model == "GDP":
	if exte == "None" or exte =="Ctr":
	#--------- Initial parameters --------------
		namepar = ["$r_c$ [pc]","$\\alpha$","$\\beta$","$\gamma$"]
		params  = [2.0,0.5,2.0,0.1]
		rng     = [[0,10],[0,2],[0,10],[0,1]]

	if exte == "Ell" or exte =="Seg":
		#--------- Initial parameters --------------
		namepar = ["$\phi$ [radians]","$r_{ca}$ [pc]","$r_{cb}$ [pc]",
					"$\\alpha$","$\\beta$","$\gamma$"]
		params  = [np.pi/4,2.0,2.0,0.5,2.0,0.0]
		rng     = [[0,np.pi],[0,10],[0,10],[0,2],[0,10],[0,2]]
		id_rc   = [3,4]
	hyp     = np.array([src,texp,sexp])

if model == "GKing":
	if exte == "None" or exte =="Ctr":
		#--------- Initial parameters --------------
		namepar = ["$r_c$ [pc]","$r_t$ [pc]","$\\alpha$","$\\beta$"]
		params  = [2.0,30.0,0.5,2.0]
		rng     = [[0,5],[10,100],[0,2],[0,5]]
		id_rt   = [3]

	if exte == "Ell" or exte =="Seg":

		#--------- Initial parameters --------------
		namepar = ["$\phi$ [radians]","$r_{ca}$ [pc]","$r_{ta}$ [pc]", 
					"$r_{cb}$ [pc]","$r_{tb}$ [pc]","$\\alpha$","$\\beta$"]
		params  = [np.pi/4,2.0,20.0,2.0,20.0,0.5,2.0]
		rng     = [[0,np.pi],[0,5],[10,100],[0,5],[10,100],[0,2],[0,5]]
		id_rc   = [3,5]
		id_rt   = [4,6]
	hyp     = np.array([src,srt,texp,sexp])

if model == "King":
	if exte == "None" or exte =="Ctr":
		#--------- Initial parameters --------------
		namepar = ["$r_c$ [pc]", "$r_t$ [pc]"]
		params  = [2.0,30.0]
		rng     = [[0,5],[10,100]]
		id_rt   = [3]
	if exte == "Ell" or exte =="Seg":
		#--------- Initial parameters --------------
		namepar = ["$\phi$ [radians]","$r_{ca}$ [pc]","$r_{ta}$ [pc]",
					 "$r_{cb}$ [pc]","$r_{tb}$ [pc]"]
		params  = [np.pi/4,2.0,30.0,2.0,30.0]
		rng     = [[0,np.pi],[0,5],[10,100],[0,5],[10,100]]
		id_rc   = [3,5]
		id_rt   = [4,6]
	hyp     = np.array([src,srt])

if model == "OGKing":
	if exte == "None" or exte =="Ctr":
		#--------- Initial parameters --------------
		namepar = ["$r_c$ [pc]","$r_t$ [pc]"]
		params  = [2.0,30.0]
		rng     = [[0,5],[10,100]]
		id_rt   = [3]

	if exte == "Ell" or exte =="Seg":
		#--------- Initial parameters --------------
		namepar = ["$\phi$ [radians]","$r_{ca}$ [pc]","$r_{ta}$ [pc]",
					 "$r_{cb}$ [pc]","$r_{tb}$ [pc]"]
		params  = [np.pi/4,2.0,30.0,2.0,30.0]
		rng     = [[0,np.pi],[0,4],[10,100],[0,4],[10,100]]
		id_rc   = [3,5]
		id_rt   = [4,6]
	hyp     = np.array([src,srt])

if model == "RGDP":
	if exte == "None" or exte =="Ctr":
		#--------- Initial parameters --------------
		namepar = ["$r_c$ [pc]","$\\alpha$","$\\beta$"]
		params  = [2.0,0.5,2.0]
		rng     = [[0,10],[0,2],[0,10]]

	if exte == "Ell" or exte =="Seg":
		#--------- Initial parameters --------------
		namepar = ["$\phi$ [radians]","$r_{ca}$ [pc]","$r_{cb}$ [pc]",
					"$\\alpha$","$\\beta$"]
		params  = [np.pi/4,2.0,2.0,0.5,2.0]
		rng     = [[0,np.pi],[0,10],[0,10],[0,2],[0,10]]
		id_rc   = [3,4]
	hyp     = np.array([src,texp,sexp])


#------------parameters of centre ----------------------------------
nameparCtr = ["$\\alpha_c\ \ [^\circ]$","$\\delta_c\ \ [^\circ]$"]
hypCtr    = np.array([1.0,1.0])
rngCtr    = [[centre[0]-1.0,centre[0]+1.0],[centre[1]-1.0,centre[1]+1.0]]
######### Parameter of luminosity segregation #########
nameSg    = ["$\kappa$ [pc$\cdot \\rm{mag}^{-1}$]"]
hypSg     = np.array([0,0.5]) # Normal prior at hypSg[0] with scale hypSg[1]
rngSg     = [[-0.6,1.2]]


#####################################################

############################################################

#------- concatenate parameters--------
if not exte == "None" :
	namepar = nameparCtr + namepar
	params  = np.append(centre,params)
	hyp     = np.append(hypCtr,hyp)
	rng     = sum([rngCtr,rng],[])

if exte == "Seg":
	namepar = namepar + nameSg
	hyp     = np.append(hyp,hypSg)
	rng     = sum([rng,rngSg],[])

#------ Determine dext and fext -----

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
###############################################################################

#------ Creates directory of extension -------
dir_fext   = dir_analysis+fext
if not os.path.exists(dir_fext): os.mkdir(dir_fext)


############## Directory of outputs ##################################################
dir_out  = dir_fext+model+'_'+str(int(Rcut))
if not real :
	dir_out  = dir_out+'_'+str(Ntot)
	fsyn     = dir_out+'/0-data.csv'
if not os.path.exists(dir_out): os.mkdir(dir_out)
#########################################################################################

########### Load module ######################
mod = importlib.import_module(dext+"."+model)
##############################################

##################### DATA ##############################################################
if real :
	#------- reads data ---------------
	df_cdts = pd.read_csv(file_members,usecols=list_observables)
	df_cdts = df_cdts[list_observables]
	#---- removes duplicateds --------
	df_cdts = df_cdts.drop_duplicates()
	##----- Select objects with band observed and 
	ido      = np.where(df_cdts.loc[:,list_observables[3]] > mag_limit)[0]
	df_cdts  = df_cdts.drop(ido)
	##----- Select objects with membership probability larger than treshold
	ido      = np.where(df_cdts.loc[:,list_observables[2]] < pro_limit)[0]
	df_cdts  = df_cdts.drop(ido)
	#---- Transform to numpy array ---------
	cdts  = df_cdts.values
else :
	# ----- Creates synthetic data ---------
	unif_syn  = np.random.uniform(low=0.0,high=1.0,size=Ntot)
	theta_syn = np.random.uniform(low=0.0,high=2*np.pi,size=Ntot)
	cdts      = np.empty((Ntot,3))

	for s,(val,t) in enumerate(zip(unif_syn,theta_syn)):
		rad       = brentq(lambda x:mod.cdf(x,t,params,float(Rcut))-val,a=0.0,b=float(Rcut))

		xn = np.rad2deg(rad/Dist)*np.cos(t)
		yn = np.rad2deg(rad/Dist)*np.sin(t)

		if exte == "Ell" or exte == "Seg":
			y = xn*np.sin(params[2]) + yn*np.cos(params[2])
			x = xn*np.cos(params[2]) - yn*np.sin(params[2])

		else:
			x = xn
			y = yn

		cdts[s,0] = x + centre[0]
		cdts[s,1] = y + centre[1]
		cdts[s,2] = 1

radii,theta           = RotRadii(cdts,centre,Dist,0.0)
Rmax = np.max(radii)

################# Calculates Radii and PA ################
#------------ Load Module -------
Module  = mod.Module(cdts,Rmax,hyp,Dist,centre)

#========================================================
#--------- Dimension, walkers and inital positions -------------------------
# number of dimensions our problem has
ndim     = len(namepar)
n_params = len(namepar)

if not os.path.isfile(dir_out+'/0-.txt'):
	pymultinest.run(Module.LogLike,Module.Priors, n_params,resume = False, verbose = True,#n_live_points=500,
		outputfiles_basename=dir_out+'/0-',multimodal=True, max_modes=2,sampling_efficiency = 'model')


# analyse the results
ana = pymultinest.Analyzer(n_params = n_params, outputfiles_basename=dir_out+'/0-')
summary = ana.get_stats()
samples = ana.get_data()[:,2:]

# MAP   = np.array(summary["modes"][0]["maximum"])
# MAP   = np.array(summary["modes"][0]["maximum a posterior"])
# sigma = np.array(summary["modes"][0]["sigma"])
# MAP   = np.array(summary["modes"][0]["mean"])
MAP   = fMAP(samples)
print(MAP)
# print(sigma)
print()
print("-" * 30, 'ANALYSIS', "-" * 30)
print("Global Evidence:\n\t%.15e +- %.15e" % (summary['nested sampling global log-evidence'], 
	summary['nested sampling global log-evidence error'] ))

########################## PLOTS #########################
# ----- select only 100 parameters to plot
samp = samples[np.random.choice(np.arange(len(samples)),size=100,replace=False)]

############### Stores the covariance matrix ##############
print("Finding covariance matrix around MAP ...")
covar = fCovar(samples,MAP)
np.savetxt(dir_out+"/"+model+"_covariance.txt",covar,
		fmt=str('%2.3f'),delimiter=" & ",newline=str("\\\ \n"))
############### Stores the MAP ##############
np.savetxt(dir_out+"/"+model+"_map.txt",MAP.reshape(1,len(MAP)),
		fmt=str('%2.3f'),delimiter="\t")


# ------ establish new centre (if needed) ------
if not exte == "None":
	centre = MAP[:2]

if exte == "Ell" or exte == "Seg":
	radii,theta        = RotRadii(cdts,centre,Dist,MAP[2])
	_,radii,theta,Rmax = TruncSort(cdts,radii,theta,Rcut)
else:
	radii,theta        = RotRadii(cdts,centre,Dist,0.0)
	_,radii,theta,Rmax = TruncSort(cdts,radii,theta,Rcut)

Nr,bins,dens = DenNum(radii,Rmax,nbins=40)
x = np.linspace(0,Rmax,50)


################## PLOTS ###############################################################
figsize = (10,10)
pdf = PdfPages(dir_out+"/"+model+'_fit.pdf')

plt.figure(figsize=figsize)
plt.fill_between(radii, Nr+np.sqrt(Nr), Nr-np.sqrt(Nr), facecolor='grey', alpha=0.5,zorder=4)
plt.plot(radii,Nr,lw=1,color="black",zorder=3,label="Data")
for s,par in enumerate(samp):
	plt.plot(x,mod.Number(x,0,par,Rmax,np.max(Nr)),lw=1,color="orange",alpha=0.2,zorder=1)
	plt.plot(x,mod.Number(x,np.pi/2,par,Rmax,np.max(Nr)),lw=1,color="orange",alpha=0.2,zorder=1)
plt.plot(x,mod.Number(x,0,MAP,Rmax,np.max(Nr)),linestyle=":",lw=1,color="red",zorder=2,label=r"$r_{ca}$")
plt.plot(x,mod.Number(x,np.pi/2,MAP,Rmax,np.max(Nr)),linestyle="--",lw=1,color="red",zorder=2,label=r"$r_{cb}$")
plt.ylim((0,1.1*max(Nr)))
plt.xlim((0,Rmax))
plt.ylabel("Number of stars")
plt.xlabel("Radius [pc]")
# plt.legend(loc="best")
pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
plt.close()


plt.figure(figsize=figsize)
for s,par in enumerate(samp):
	plt.plot(x,mod.Density(x,0,par,Rmax),lw=1,color="grey",alpha=0.2,zorder=1)
	plt.plot(x,mod.Density(x,np.pi/2,par,Rmax),lw=1,color="grey",alpha=0.2,zorder=1)
plt.errorbar(bins,dens[:,0],yerr=dens[:,1],fmt="o",color="black",lw=2,ms=5,zorder=4)
plt.plot(x,mod.Density(x,0,MAP,Rmax),linestyle=":", linewidth=3,color="red",zorder=2)
plt.plot(x,mod.Density(x,np.pi/2,MAP,Rmax),linestyle="--", linewidth=3,color="red",zorder=3)
plt.ylabel("Density [stars $\cdot$ pc$^{-2}$]")
plt.xlabel("Radius [pc]")
plt.yscale("log")
plt.ylim((0.9*(dens[-1,0]-dens[-1,1]),1.2*(dens[0,0]+dens[0,1])))
plt.xlim((0,Rmax))
pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
plt.close()

plt.figure(figsize=figsize)
n, bins, patches = plt.hist(theta,50,density=True, facecolor='green', alpha=0.5)
plt.xlabel('Position Angle [radians]')
plt.ylabel('Density [stars $\cdot$ radians$^{-1}$]')
pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
plt.close()



############################ SKY CHART ################################################
fig, ax = plt.subplots(figsize=figsize)
#--------------- Create grid in ra dec ----------
fk5_centre = SkyCoord(ra=centre[0],dec=centre[1],frame='icrs',unit='deg')
gal_centre = np.array([fk5_centre.galactic.l.degree,fk5_centre.galactic.b.degree])

fk5_cdts = SkyCoord(ra=cdts[:,0],dec=cdts[:,1],frame='icrs',unit='deg')

gal_cdts = np.empty((len(cdts[:,0]),2))
gal_cdts[:,0] = fk5_cdts.galactic.l.degree
gal_cdts[:,1] = fk5_cdts.galactic.b.degree

#---------- plot stars -----------------------------------
plt.scatter(gal_cdts[:,0],gal_cdts[:,1],s=10*cdts[:,3], edgecolor='grey', facecolor='none', marker='*',zorder=1)
plt.xlim(np.min(gal_cdts[:,0])-1.0,np.max(gal_cdts[:,0])+1)
plt.ylim(np.min(gal_cdts[:,1])-1.0,np.max(gal_cdts[:,1])+1)
ax.set_aspect('equal')
if exte == "Ell" or exte == "Seg":
	rca = math.degrees(MAP[3]/Dist)
	p1 = centre + 5*rca*np.array([np.cos(MAP[2]),np.sin(MAP[2])])
	# plt.scatter(p1[0],p1[1],s=20,color="red", marker='*',zorder=1)

if exte == "Ctr":
	rca = math.degrees(MAP[2]/Dist)
	rcb = rca
	p1 = centre + 5*rca


fk5_p1 = SkyCoord(ra=p1[0],dec=p1[1],frame='icrs',unit='deg')
gal_p1 = np.array([fk5_p1.galactic.l.degree,fk5_p1.galactic.b.degree])
angle = math.degrees(np.arctan2(gal_p1[1]-gal_centre[1],gal_p1[0]-gal_centre[0]))


if exte == "Ell" or exte == "Seg":
	if model in ["GKing","King","OGKing"]:
		rcb = math.degrees(MAP[5]/Dist)
		rta = math.degrees(MAP[4]/Dist)
		rtb = math.degrees(MAP[6]/Dist)
		ell  = Ellipse(gal_centre,width=2*rta,height=2*rtb,angle=angle,clip_box=ax.bbox,
			edgecolor="black",facecolor=None,fill=False,linewidth=2)
		ax.add_artist(ell)

	else:
		rcb = math.degrees(MAP[4]/Dist)

	p1 = gal_centre + 5*rca*np.array([np.cos(math.radians(angle+180)),np.sin(math.radians(angle+180))])
	p2 = gal_centre + 5*rca*np.array([np.cos(math.radians(angle)),np.sin(math.radians(angle))])
	p3 = gal_centre + 5*rcb*np.array([np.cos(math.radians(angle+90+180)),np.sin(math.radians(angle+90+180))])
	p4 = gal_centre + 5*rcb*np.array([np.cos(math.radians(angle+90)),np.sin(math.radians(angle+90))])

	line_a = Line2D([p1[0],p2[0]], [p1[1], p2[1]],
	        lw=2, color='black', axes=ax)
	line_b = Line2D([p3[0],p4[0]], [p3[1], p4[1]],
	        lw=2, color='black', axes=ax)

	ax.add_line(line_a)
	ax.add_line(line_b)

	
ell  = Ellipse(gal_centre,width=2*rca,height=2*rcb,angle=angle,clip_box=ax.bbox,
			edgecolor="black",facecolor=None,fill=False,linewidth=2)
ax.add_artist(ell)

plt.xlabel('$l$ [deg]')
plt.ylabel('$b$ [deg]')

pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
plt.close()


####################################### CORNER PLOT #########################################
plt.rcParams["figure.figsize"] = figsize
if real :
	figure = corner.corner(samples, labels=namepar,truths=MAP,truth_color="red",range=rng,
		reverse=False,plot_datapoints=False,fill_contours=True,
		quantiles=[0.16, 0.5, 0.84],
        show_titles=True, title_kwargs={"fontsize": 12})
else:
	figure = corner.corner(samples, labels=namepar,truths=params,truth_color="blue",range=rng,
		reverse=False,plot_datapoints=False,fill_contours=True,
		quantiles=[0.16, 0.5, 0.84],
        show_titles=True, title_kwargs={"fontsize": 12})

pdf.savefig(bbox_inches='tight')
plt.close()

if exte == "Seg":
	#------ separates data in bins -----------
	low  = 12.0
	up   = 15.0
	idl  = np.where(cdts[:,3] <= low)[0]
	idu  = np.where(cdts[:,3] >= up)[0]
	idm  = np.where(np.logical_and(cdts[:,3] > low,cdts[:,3] < up))[0]

	radii_l = radii[idl]
	radii_m = radii[idm]
	radii_u = radii[idu]

	Nr_l,bins,dens_l = DenNum(radii_l,Rmax,nbins=21)
	Nr_m,bins,dens_m = DenNum(radii_m,Rmax,nbins=21)
	Nr_u,bins,dens_u = DenNum(radii_u,Rmax,nbins=21)

	dlt_l  = np.mean(cdts[idl,3]) - Module.mode
	dlt_m  = np.mean(cdts[idm,3]) - Module.mode
	dlt_u  = np.mean(cdts[idu,3]) - Module.mode

	# plt.figure()
	# plt.errorbar(radii[idl],cdts[idl,3],yerr=cdts[idl,4],fmt="o",color="green",ecolor="grey",lw=1,ms=2)
	# plt.errorbar(radii[idm],cdts[idm,3],yerr=cdts[idm,4],fmt="o",color="cyan",ecolor="grey",lw=1,ms=2)
	# plt.errorbar(radii[idu],cdts[idu,3],yerr=cdts[idu,4],fmt="o",color="magenta",ecolor="grey",lw=1,ms=2)
	# plt.plot(x,Module.mode+(x*MAP[7]))
	# plt.xlabel("Radius [pc]")
	# plt.ylabel('J [mag]')
	# pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
	# plt.close()

	

	for s,par in enumerate(samp):
		plt.plot(x,mod.DenSeg(x,0,par,Rmax,dlt_l),
			lw=1,color="grey",alpha=0.2,zorder=1)

	plt.errorbar(bins,dens_l[:,0],yerr=dens_l[:,1],fmt="o",color="green",ecolor="grey",lw=1,ms=2,zorder=3)
	plt.plot(x,mod.DenSeg(x,0,MAP,Rmax,dlt_l),
			 linewidth=1,color="green",zorder=2)

	plt.errorbar(bins,dens_m[:,0],yerr=dens_m[:,1],fmt="o",color="cyan",ecolor="grey",lw=1,ms=2,zorder=3)
	plt.plot(x,mod.DenSeg(x,0,MAP,Rmax,dlt_m),
			 linewidth=1,color="cyan",zorder=2)

	plt.errorbar(bins,dens_u[:,0],yerr=dens_u[:,1],fmt="o",color="magenta",ecolor="grey",lw=1,ms=2,zorder=3)
	plt.plot(x,mod.DenSeg(x,0,MAP,Rmax,dlt_u),
			 linewidth=1,color="magenta",zorder=2)

	plt.ylabel("Density [stars $\cdot$ pc$^{-2}$]")
	plt.xlabel("Radius [pc]")
	plt.ylim(1e-3,0.5)
	plt.yscale("log")
	pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
	plt.close()

#######################  ELLIPTICITIES ##############################
if exte == "Ell" or exte == "Seg":
	#--------- computes ellipticity of core radius
	eps_rc     = np.array(map(Epsilon,samples[:,id_rc]))
	samples    = np.column_stack((samples,eps_rc))
	epsrc_mode = st.mode(eps_rc)[0]
	percent    = np.percentile(eps_rc,[14,50,86])
	nerc,bins,_= plt.hist(eps_rc,30, density=True,
					ec="black",histtype='step', linestyle='solid',label="$\epsilon_{rc}$")
	plt.annotate('$\epsilon_{rc}$=[%0.2f,%0.2f,%0.2f]' % ( tuple(percent) ),[0.1,0.95],xycoords="axes fraction")
	epsilons = np.array([epsrc_mode])

	if model in ["GKing","King","OGKing"]:
		eps_rt      = np.array(map(Epsilon,samples[:,id_rt]))
		samples     = np.column_stack((samples,eps_rt))
		epsrt_mode  = st.mode(eps_rt)[0]
		percent     = np.percentile(eps_rt,[14,50,86])
		nert,bins,_ = plt.hist(eps_rt,30, density=True, range=[0,1], 
						ec="black",histtype='step', linestyle='dashed',label="$\epsilon_{rt}$")
		plt.annotate('$\epsilon_{rt}$=[%0.2f,%0.2f,%0.2f]' % ( tuple(percent) ),[0.7,0.95],xycoords="axes fraction")
		epsilons = np.array([epsrc_mode,epsrt_mode]).reshape((1,2))

	#---------------------------
	plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.02f')) 
	plt.xlim(0,1)
	plt.xlabel('$\epsilon$')
	plt.legend(shadow = False,
	bbox_to_anchor=(0., 1.001, 1., .102),
    borderaxespad=0.,
	frameon = True,
	fancybox = True,
	ncol = 2,
	fontsize = 'smaller',
	mode = 'expand',
	loc = 'upper center')
	pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
	plt.close()

	MAP = np.append(MAP,epsilons)

#--------- computes the distribution of number of stars -----
if model in ["GKing","King","OGKing"]:
	Nrt = np.empty(len(samples))
	for s,par in enumerate(samples):
		Nrt[s] = mod.Number(par[id_rt[0]],0,par,Rmax,np.max(Nr))
	Nrt_mode = st.mode(eps_rc)[0]
	percent  = np.percentile(Nrt,[14,50,86])
	plt.hist(Nrt,50, density=True, ec="black",histtype='step')
	plt.ylabel("Density")
	plt.xlabel("Number of stars within $r_t$")
	plt.annotate('Number = [%3.0f,%3.0f,%3.0f]' % ( tuple(percent) ),(0.5,0.8),xycoords="axes fraction")
	plt.yscale("log")
	pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
	plt.close()

	# print(bins.shape)

	Nrt        = Nrt[np.where(Nrt < 1e4)[0]]
	kde        = st.gaussian_kde(Nrt)
	x          = np.linspace(np.min(Nrt),np.max(Nrt),num=1000)
	Nrt_mode   = np.array(x[kde(x).argmax()]).reshape((1,))
	bins       = np.linspace(0.9*np.min(Nrt),1.1*np.max(Nrt),num=50)   
	nrt,_,     = np.histogram(Nrt,bins=bins)
	nrt        = np.append(nrt,0)

	f_e = file(dir_out+"/"+model+"_numbers.txt", 'w')
	np.savetxt(f_e, Nrt_mode,fmt=str('%2.3f'),delimiter=" ")
	np.savetxt(f_e, np.c_[bins,nrt],fmt=str('%2.5e'),delimiter=" ")
	f_e.close()
#--------- computes the distribution of total mass -----
	bins          = np.linspace(0,1e4,num=100) 
	massrj        = MassRj(samples[:,id_rt[0]])
	massrj        = massrj[np.where(massrj < 1e4)[0]]
	kde           = st.gaussian_kde(massrj)
	x             = np.linspace(np.min(massrj),np.max(massrj),num=1000)
	massrj_mode   = np.array(x[kde(x).argmax()]).reshape((1,))

	nmr,_,_       = plt.hist(massrj,bins=bins, density=True,ec="black",ls="dashed",histtype='step',label="$M_{rt}$")
	nmr           = np.append(nmr,0)
	mass_mode     = np.array([massrj_mode]).reshape((1,1))
	nms           = np.c_[bins,nmr]

	  
	if not exte == "Ctr":
		massep        = MassEps(samples[:,id_rt[0]],eps_rt)
		massep        = massep[np.where(massep < 1e4)[0]]
		kde           = st.gaussian_kde(massep)
		x             = np.linspace(np.min(massep),np.max(massep),num=1000)
		massep_mode   = np.array(x[kde(x).argmax()]).reshape((1,))
		nme,_,_       = plt.hist(massep,bins=bins, density=True,ec="black",histtype='step',label="$M_{\epsilon}$")
		nme           = np.append(nme,0)
		mass_mode     = np.array([massrj_mode,massep_mode]).reshape((1,2))
		nms           = np.c_[bins,nmr,nme]

	plt.legend()
	plt.ylabel("Density")
	plt.xlabel("Total mass [$M_\odot$]")
	plt.yscale("log")
	# plt.ylim(1e-5,1)
	pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
	plt.close()

	f_e = file(dir_out+"/"+model+"_mass.txt", 'w')
	np.savetxt(f_e, mass_mode,fmt=str('%2.3f'),delimiter=" ")
	np.savetxt(f_e, nms,fmt=str('%2.5e'),delimiter=" ")
	f_e.close()

pdf.close()


#------------------- saves chain -----------
np.savetxt(dir_out+'/chain.csv',samples, delimiter=",")

#------------ END --------------------------------

print("Take a look at the pdf files in "+dir_out) 








 
