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
from Functions import TruncSort,DenNum,RotRadii,fMAP,fCovar,Epsilon,MassRj,MassEps

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

from globals_NGC2244 import *

#------ Determine dext and fext -----

if profile["extension"] == "None":
	dext = "Models"
	fext = "NoCentre/"
elif profile["extension"] == "Ctr":
	dext = "ModelsCtr"
	fext = "Centre/"
elif profile["extension"] == "Ell":
	dext = "ModelsEll"
	fext = "Elliptic/"
elif profile["extension"] == "Seg":
	dext = "ModelsSeg"
	fext = "Segregated/"
else :
	print("Extention not recognised!")
	print("Available ones are: None, Centre (Ctr), Elliptic (Ell), and Segregated (Seg")
	sys.exit()
###############################################################################

#------ Creates directory of profile["extension"]nsion -------
dir_fext   = dir_analysis+fext
if not os.path.exists(dir_fext): os.mkdir(dir_fext)


############## Directory of outputs ##################################################
dir_out  = dir_fext+profile["model"]+'_'+str(int(Rcut))
if not real :
	dir_out  = dir_out+'_'+str(Ntot)
	fsyn     = dir_out+'/0-data.csv'
if not os.path.exists(dir_out): os.mkdir(dir_out)
#########################################################################################

########### Load module ######################
mod = importlib.import_module(dext+"."+profile["model"])
##############################################

##################### Reads or create data ##############################################################
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
		rad       = brentq(lambda x:mod.cdf(x,t,profile["initial_values"],float(Rcut))-val,a=0.0,b=float(Rcut))

		xn = np.rad2deg(rad/distance)*np.cos(t)
		yn = np.rad2deg(rad/distance)*np.sin(t)

		if profile["extension"] == "Ell" or profile["extension"] == "Seg":
			y = xn*np.sin(profile["initial_values"][2]) + yn*np.cos(profile["initial_values"][2])
			x = xn*np.cos(profile["initial_values"][2]) - yn*np.sin(profile["initial_values"][2])

		else:
			x = xn
			y = yn

		cdts[s,0] = x + centre[0]
		cdts[s,1] = y + centre[1]
		cdts[s,2] = 1

radii,theta           = RotRadii(cdts,centre,distance,0.0)
Rmax = np.max(radii)
print(Rmax)


################# Calculates Radii and PA ################
#------------ Load Module -------
Module  = mod.Module(cdts,Rmax,profile["hyper-parameters"],distance,centre)

#========================================================
#--------- Dimension, walkers and inital positions -------------------------
# number of dimensions our problem has
dimension    = len(profile["parameter_names"])

if not os.path.isfile(dir_out+'/0-.txt'):
	pymultinest.run(Module.LogLike,Module.Priors,dimension,resume = False, verbose = True,#n_live_points=500,
		outputfiles_basename=dir_out+'/0-',multimodal=True, max_modes=2,sampling_efficiency = 'model')


# analyse the results
ana = pymultinest.Analyzer(n_params = dimension, outputfiles_basename=dir_out+'/0-')
summary = ana.get_stats()
samples = ana.get_data()[:,2:]

#----------------- MAPS------------------------------
# MAP   = np.array(summary["modes"][0]["maximum"])
MAP   = np.array(summary["modes"][0]["maximum a posterior"])
# sigma = np.array(summary["modes"][0]["sigma"])
# MAP   = np.array(summary["modes"][0]["mean"])
# MAP   = fMAP(samples)
print(MAP)
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
np.savetxt(dir_out+"/"+profile["extension"]+"_covariance.txt",covar,
		fmt=str('%2.3f'),delimiter=" & ",newline=str("\\\ \n"))
############### Stores the MAP ##############
np.savetxt(dir_out+"/"+profile["extension"]+"_map.txt",MAP.reshape(1,len(MAP)),
		fmt=str('%2.3f'),delimiter="\t")


# ------ establish new centre (if needed) ------
if not profile["extension"] == "None":
	centre = MAP[:2]

if profile["extension"] == "Ell" or profile["extension"] == "Seg":
	radii,theta        = RotRadii(cdts,centre,distance,MAP[2])
	cdts,radii,theta,Rmax = TruncSort(cdts,radii,theta,Rcut)
else:
	radii,theta        = RotRadii(cdts,centre,distance,0.0)
	cdts,radii,theta,Rmax = TruncSort(cdts,radii,theta,Rcut)

Nr,bins,dens = DenNum(radii,Rmax,nbins=40)
x = np.linspace(0,Rmax,50)


lim_dens = 0.5*np.min(dens[dens != 0.0]),1.1*np.max(dens)


################## PLOTS ###############################################################
figsize = (10,10)
pdf = PdfPages(dir_out+"/"+profile["extension"]+'_fit.pdf')

plt.figure(figsize=figsize)
plt.fill_between(radii, Nr+np.sqrt(Nr), Nr-np.sqrt(Nr), facecolor='grey', alpha=0.5,zorder=4)
plt.plot(radii,Nr,lw=1,color="black",zorder=3,label="Data")
for s,par in enumerate(samp):
	plt.plot(x,mod.Number(x,0,par,Rmax,np.max(Nr)),lw=1,color="orange",alpha=0.2,zorder=1)
	plt.plot(x,mod.Number(x,np.pi/2,par,Rmax,np.max(Nr)),lw=1,color="orange",alpha=0.2,zorder=1)
plt.plot(x,mod.Number(x,0,MAP,Rmax,np.max(Nr)),linestyle=":",lw=1,color="red",zorder=2,label=r"$r_{ca}$")
plt.plot(x,mod.Number(x,np.pi/2,MAP,Rmax,np.max(Nr)),linestyle="--",lw=1,color="red",zorder=2,label=r"$r_{cb}$")
plt.ylim(0,1.1*max(Nr))
plt.xlim(0,Rmax)
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
plt.ylim(lim_dens)
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
if profile["extension"] == "Ell" or profile["extension"] == "Seg":
	rca = math.degrees(MAP[3]/distance)
	p1 = centre + 5*rca*np.array([np.cos(MAP[2]),np.sin(MAP[2])])
	# plt.scatter(p1[0],p1[1],s=20,color="red", marker='*',zorder=1)

if profile["extension"] == "Ctr":
	rca = math.degrees(MAP[2]/distance)
	rcb = rca
	p1 = centre + 5*rca

if profile["extension"] == "None":
	rca = math.degrees(MAP[0]/distance)
	rcb = rca
	p1 = np.array(centre) + 5*rca


fk5_p1 = SkyCoord(ra=p1[0],dec=p1[1],frame='icrs',unit='deg')
gal_p1 = np.array([fk5_p1.galactic.l.degree,fk5_p1.galactic.b.degree])
angle = math.degrees(np.arctan2(gal_p1[1]-gal_centre[1],gal_p1[0]-gal_centre[0]))


if profile["extension"] == "Ell" or profile["extension"] == "Seg":
	if profile["extension"] in ["GKing","King","OGKing"]:
		rcb = math.degrees(MAP[5]/distance)
		rta = math.degrees(MAP[4]/distance)
		rtb = math.degrees(MAP[6]/distance)
		ell  = Ellipse(gal_centre,width=2*rta,height=2*rtb,angle=angle,clip_box=ax.bbox,
			edgecolor="black",facecolor=None,fill=False,linewidth=2)
		ax.add_artist(ell)

	else:
		rcb = math.degrees(MAP[4]/distance)

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
	figure = corner.corner(samples, labels=profile["parameter_names"],truths=MAP,truth_color="red",range=profile["parameter_interval"],
		reverse=False,plot_datapoints=False,fill_contours=True,
		quantiles=np.array(percentiles)/100,
        show_titles=True, title_kwargs={"fontsize": 12})
else:
	figure = corner.corner(samples, labels=namepar,truths=profile["initial_values"],truth_color="blue",range=profile["parameter_interval"],
		reverse=False,plot_datapoints=False,fill_contours=True,
		quantiles=np.array(percentiles)/100,
        show_titles=True, title_kwargs={"fontsize": 12})

pdf.savefig(bbox_inches='tight')
plt.close()

if profile["extension"] == "Seg":
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
	plt.ylim(lim_dens)
	plt.yscale("log")
	pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
	plt.close()

#######################  ELLIPTICITIES ##############################
if profile["extension"] == "Ell" or profile["extension"] == "Seg":
	#--------- computes ellipticity of core radius
	eps_rc     = np.array(Epsilon(samples[:,profile["id_rc"]].T))
	samples    = np.column_stack((samples,eps_rc))
	epsrc_mode = st.mode(eps_rc)[0]
	quanta     = np.percentile(eps_rc,percentiles)
	nerc,bins,_= plt.hist(eps_rc,30, density=True,
					ec="black",histtype='step', linestyle='solid',label="$\epsilon_{rc}$")
	plt.annotate('$\epsilon_{rc}$=[%0.2f,%0.2f,%0.2f]' % ( tuple(quanta) ),[0.1,0.95],xycoords="axes fraction")
	epsilons = np.array([epsrc_mode])

	if profile["extension"] in ["GKing","King","OGKing"]:
		eps_rt      = np.array(Epsilon(samples[:,profile["id_rt"]].T))
		samples     = np.column_stack((samples,eps_rt))
		epsrt_mode  = st.mode(eps_rt)[0]
		quanta     = np.percentile(eps_rt,percentiles)
		nert,bins,_ = plt.hist(eps_rt,30, density=True, range=[0,1], 
						ec="black",histtype='step', linestyle='dashed',label="$\epsilon_{rt}$")
		plt.annotate('$\epsilon_{rt}$=[%0.2f,%0.2f,%0.2f]' % ( tuple(quanta) ),[0.7,0.95],xycoords="axes fraction")
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

#-------------- Number of systems ----------------------------------
if profile["extension"] in ["GKing","King","OGKing"]:
	Nrt = np.empty(len(samples))
	for s,par in enumerate(samples):
		Nrt[s] = mod.Number(par[profile["id_rt"][0]],0,par,Rmax,np.max(Nr))
	quanta     = np.percentile(Nrt,percentiles)
	plt.hist(Nrt,50, density=True, ec="black",histtype='step')
	plt.ylabel("Density")
	plt.xlabel("Number of stars within $r_t$")
	plt.annotate('Number of systems = [%3.0f,%3.0f,%3.0f]' % ( tuple(quanta) ),(0.5,0.8),xycoords="axes fraction")
	# plt.xlim(quanta[1]-3*quanta[0],quanta[1]+3*quanta[2])
	pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
	plt.close()

#--------- Total mass -----
	
	LogMassrj  = np.log10(MassRj(samples[:,profile["id_rt"][0]]))
	quanta  = np.percentile(LogMassrj,percentiles)
	plt.hist(LogMassrj,50, density=True, ec="black",histtype='step')
	plt.ylabel("Density")
	plt.xlabel("Log Mass [$M_\odot$]")
	plt.annotate('Log Mass = [%3.2f,%3.2f,%3.2f]' % ( tuple(quanta) ),(0.5,0.8),xycoords="axes fraction")
	# plt.yscale("log")
	pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
	plt.close()

pdf.close()


#------------ END --------------------------------

print("Take a look at the pdf files in "+dir_out) 








 
