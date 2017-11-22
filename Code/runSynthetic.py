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
from scipy.optimize import brentq
import scipy.stats as st

import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.mlab
import seaborn as sns
import emcee
import corner

# np.random.seed(seed=100)
from astroML.decorators import pickle_results

dir_  = os.path.expanduser('~') +"/PyAspidistra/"
dir_graph = dir_ + "Analysis/Synthetic/King/Truncated/"
real  = False

model = "King"
iters = 2000
walkers_ratio = 10
init  = int(0.5*iters) # iterations to be discarded as burnin
##################### DATA ##############################################################
# ------ number of stars and maxium radius (pc)
Ns    = np.array([1000,2000,3000])
Rs    = np.array([5,10,15,20])
Ss    = 10

#+====================== Chose model ================
if model == "King":
	from Models.KingSynthetic  import LogPosterior,Number,cdfKing,King
	# ------ true values ------
	tpars = np.array([2.0,20.0])
	# ---- hyperparametros ---
	hyper  = np.array([1e2,1e2]) 
	#--------- Initial parameters --------------
	namepar = ["$r_c$","$r_t$"]

ndim  = len(tpars)
nwalkers = ndim*walkers_ratio

# r = np.array([1,2,5,7,19.9])
# print 2*r*King(r,2,20)
# sys.exit()
############# Loop over stars and truncation values #################
@pickle_results("SyntheticKing.pkl")
def Loop(Ss,Ns,Rs,ndim,nwalkers):
	asmp = np.empty((Ss,len(Ns),len(Rs),(iters-init)*nwalkers,ndim))
	amap = np.empty((Ss,len(Ns),len(Rs),ndim))
	for i,N in enumerate(Ns):
		for j,R in enumerate(Rs):
			for h in range(Ss):
				print "Stars and Truncation radius: ",N," ",R
				strs = np.random.uniform(low=0.0,high=1.0,size=N)
				radii= np.empty(N)
				for k,st in enumerate(strs):
					radii[k]= brentq(lambda x:cdfKing(x,tpars[0],tpars[1],R)-st,a=0.0,b=float(R))
				print "Maximum radius: ",np.max(radii)
				
				################## SAMPLER ##################################################
				sampler = emcee.EnsembleSampler(nwalkers, ndim, LogPosterior, args=(radii,R,hyper))
				sampler.a = 4.0
				#---------  inital positions -------------------------
				init_pos = tpars*(1.0 + 0.5*np.random.uniform(low=0,high=1,size=(nwalkers,2)))
				# print init_pos
				# sys.exit()
				#=====================INFerence===================================
				sampler.run_mcmc(init_pos, iters)

				#------ analysis -----
				iat   = int(max(sampler.acor))
				print 'Mean Acceptance Fraction: ', np.mean(sampler.acceptance_fraction)
				print 'Autocorrelation Time: ', sampler.acor

				#--------- MAP ----------------------------------------
				samples  = sampler.chain[:, init:,:]
				id_map   = np.where(sampler.lnprobability[:,init:] == np.max(sampler.lnprobability[:,init:]))
				MAP      = samples[id_map][0]
				print "MAP values: ",MAP
				#------------------------------------------------------
				asmp[h,i,j,:,0] = samples[:,:,0].flatten()
				asmp[h,i,j,:,1] = samples[:,:,1].flatten()
				amap[h,i,j,:]   = MAP
				#----------------------------------
				srad     = np.sort(radii)
				Nr       = np.arange(len(radii))+1
				x        = np.linspace(0.0,R,num=100)

				#---------- Graficas de valores verdaderos --------
				pdf = PdfPages(dir_graph+'Synthetic-'+str(model)+'-Rcut='+str(int(R))+"-Stars="+str(int(N))+'-Iter='+str(h)+'.pdf')

				# ---------------- Trace plots --------------------
				fig = plt.figure()
				fig.suptitle('Walkers', fontsize=14, fontweight='bold')

				ax = fig.add_subplot(211)
				ax.plot(samples[:,:,0].T)
				ax.set_xticks([])
				ax.set_ylabel(namepar[0])

				ax = fig.add_subplot(212)
				ax.plot(samples[:,:,1].T)
				ax.set_xlabel('Iterations')
				ax.set_ylabel(namepar[1])

				pdf.savefig(bbox_inches='tight')
				plt.close()

				# ------------- Corner plot ---------------------

				fig = corner.corner(samples.reshape((-1, ndim)),bins=10, labels=namepar,truths=tpars)
				pdf.savefig(bbox_inches='tight')
				plt.close

				#------------- Fit plot -----------------------
				fig = plt.figure()
				plt.scatter(srad,Nr,s=1,color="black")
				plt.plot(x,Number(x,MAP[0],MAP[1],R,N), linewidth=1,color="red")
				plt.plot(x,Number(x,tpars[0],tpars[1],R,N), linewidth=1,color="blue")
				plt.ylim((0,1.1*max(Nr)))
				plt.xlim((0,1.1*R))
				plt.xlabel("Radius [$pc$]")
				plt.ylabel("Number of stars")
				pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
				plt.close()
				pdf.close()
				# sys.exit()

	return asmp,amap

asmp,amap = Loop(Ss,Ns,Rs,ndim,nwalkers)

dmap = amap.copy()
dmap[:,:,:,0] = (amap[:,:,:,0] - tpars[0])/tpars[0]
dmap[:,:,:,1] = (amap[:,:,:,1] - tpars[1])/tpars[1]

mmap = np.empty((len(Ns),len(Rs),ndim,2))
mmap[:,:,:,0] = np.mean(dmap,axis=0)
mmap[:,:,:,1] = np.std(dmap,axis=0)

print amap[:,:,0,0]
# print dmap[:,0,0,1]
# print np.mean(dmap[:,0,0,1]),np.std(dmap[:,0,0,1])
# print mmap[0,0,1,0],mmap[0,0,1,1]
# sys.exit()

jit = np.array([-0.25,0,0.25])
lty = ["dotted", "dashdot","dashed" ]
lss = [None,None,None]
cls = [None,None,None,None]

clr = mpl.cm.get_cmap(name='rainbow')([240,190,100,30])
bws = "scott"

pdf = PdfPages(dir_graph+'Synthetic-'+str(model)+'-densities.pdf')
fig = plt.figure()
plt.suptitle('Posterior distributions', fontsize=14, fontweight='bold')

p1 = fig.add_subplot(211)
for i,N in enumerate(Ns):
	lss[i] = mpl.lines.Line2D([], [], ls=lty[i], color="black",label=str(N))
	for j,R in enumerate(Rs):
		sns.kdeplot(asmp[:,i,j,:,0].flatten(),color=clr[j],ls=lty[i],shade=False,clip=(0,5),bw=bws)
p1.vlines(tpars[0], [1e-5], 10.0,colors="grey")
p1.set_xlabel(namepar[0]+" [$pc$]")
p1.set_xlim(1,4)
p1.set_yscale("log", nonposy='clip')
p1.set_ylim(0.01,6.0)
p1.legend(title="Stars",handles=lss,bbox_to_anchor=(0,0,1.2,0.2), borderaxespad=0.)

p2 = fig.add_subplot(212)
for j,R in enumerate(Rs):
	cls[j] = mpl.lines.Line2D([], [], color=clr[j], label=str(R))
	for i,N in enumerate(Ns):
		sns.kdeplot(asmp[:,i,j,:,1].flatten(),color=clr[j],ls=lty[i],shade=False,clip=(5,50),bw=bws)
p2.vlines(tpars[1], [1e-5], 10.0,colors="grey")
p2.set_xlabel(namepar[1] +" [$pc$]")
p2.set_yscale("log", nonposy='clip')
p2.set_ylim(0.01,1)
p2.set_xlim(5,40)
p2.legend(title="$R_{max}\ \ [pc]$",handles=cls,bbox_to_anchor=(0,0,1.2,0.9), borderaxespad=0.)

plt.subplots_adjust(left=0.125, bottom=0.2, right=0.9, top=None, wspace=0.2, hspace=0.4)

pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
plt.close()
################ Absolute Relative Error ######################
fig = plt.figure()
plt.suptitle('Mean Relative Error', fontsize=14, fontweight='bold')

p1 = fig.add_subplot(211)
for i,N in enumerate(Ns):
	p1.errorbar(Rs+jit[i],mmap[i,:,0,0],yerr=mmap[i,:,0,1],ls=lty[i],label=str(N))
# p1.set_yticks([0.0,0.1,0.2])
p1.set_xticks([])
p1.set_ylabel(namepar[0])

p2 = fig.add_subplot(212)
for i,N in enumerate(Ns):
	p2.errorbar(Rs+jit[i],mmap[i,:,1,0],yerr=mmap[i,:,1,1],ls=lty[i],label=str(N))
p2.set_xlabel("$R_{max}\,[pc]$")
p2.set_ylabel(namepar[1])
p2.legend(title="Stars",loc=0,ncol=1,bbox_to_anchor=(0,0,1.2,1.4), borderaxespad=0.)

plt.subplots_adjust(left=0.125, bottom=0.2, right=0.9, top=None, wspace=0.2, hspace=0.0)

pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
pdf.close()
