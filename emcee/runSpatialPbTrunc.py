import sys
import numpy as np
import scipy.stats as st
from pandas import read_csv

import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.mlab

from pyhmc import integrated_autocorr1 as iact
from mpltools import special

from statsmodels.tsa.stattools import acf
import seaborn as sns

import emcee
import corner

dir_  = "/pcdisk/boneym5/jolivares/Aspidistra/"
real  = True


model = str(sys.argv[1])
rcut  = float(sys.argv[2])
iters = 400
walkers_ratio = 6
init  = int(0.2*iters) # iterations to be discarded as burnin
##################### DATA ##############################################################
if real :
	# fdata = dir_+'Data/Tycho2-DANCe-Complete.txt'
	fdata = dir_+'Data/Members.csv'
	#--------- Transforms data --------
	Dist  = 136
	D2R   = np.pi/180
	cntr  = [56.65,24.13]
	data  = np.array(read_csv(fdata,header=0,sep=','))
	cdts  = np.array(data[:,10:12],dtype=np.float32)
	# cdts  = np.array(read_csv(fdata,header=0,dtype=np.float32, sep=" "))[:,0:2]
	radii = np.arccos(np.sin(cntr[1]*D2R)*np.sin(cdts[:,1]*D2R)+
	            np.cos(cntr[1]*D2R)*np.cos(cdts[:,1]*D2R)*
	            np.cos((cntr[0]-cdts[:,0])*D2R))*Dist
	pro   = data[:,8]
else :
	Ntot  = 10000
	cr    = 0.0001
	Ncls  = int(Ntot*(1-cr))
	cls   = np.random.multivariate_normal([0.0,0.0],[[4.0, 0.0],[0.0, 4.0]],Ncls)
	fld   = np.column_stack([np.random.uniform(-10,10,Ntot-Ncls),np.random.uniform(-10,10,Ntot-Ncls)])
	cdts  = np.vstack([cls,fld])
	radii = np.sqrt(cdts[:,0]**2+cdts[:,1]**2)
	pro   = np.repeat(1,len(radii))
#----------- Select Number and Max Radius ------------------------
radii      = np.sort(radii)
idx        = np.where(radii <  rcut)[0]
radii      = radii[idx]
pro        = pro[idx]
Rmax       = max(radii)
bins       = np.linspace(0,Rmax+0.1,101)
hist       = np.histogram(radii,bins=bins)[0]
bins       = bins[1:]
dr         = np.hstack([bins[0]/2,np.diff(bins)])
da         = 2*np.pi*bins*dr
densi      = hist/da
densi      = densi/sum(densi*bins*dr)
Nr         = np.arange(len(radii))+1
#########################################################################################
#+====================== Chose model ================
if model == "Gaussian":
	from Models.GaussianProTrunc import LogPosterior,Number,LikeCluster,LikeField
	# ---- hyperparametros ---
	hyper  = np.array([1e2]) 
	#--------- Initial parameters --------------
	namepar = ["$Sg$"]
	mins    = np.array(([0.1]))
	maxs    = np.array(([5.0]))
	#--------- arguments of logLike function
	args     = (radii,pro,Rmax,hyper)

if model == "Plummer":
	from Models.PlummerProTrunc import LogPosterior,Number,LikeCluster,LikeField
	# ---- hyperparametros ---
	hyper  = np.array([1e2]) 
	#--------- Initial parameters --------------
	namepar = ["$Rc$"]
	mins    = np.array(([0.1]))
	maxs    = np.array(([5.0]))
	#--------- arguments of logLike function
	args     = (radii,pro,Rmax,hyper)
if model == "King":
	from Models.KingProTrunc    import LogPosterior,Number,LikeCluster,LikeField
	# ---- hyperparametros ---
	hyper  = np.array([1e2,1e2]) 
	#--------- Initial parameters --------------
	namepar = ["$Rc$","$Rt$"]
	mins    = np.array(([0.001, 3.0]))
	maxs    = np.array(([3.0, 10]))
	#--------- arguments of logLike function
	args     = (radii,pro,Rmax,hyper)
if model == "EFF":
	from Models.EFFProTrunc    import LogPosterior,Number,LikeCluster,LikeField
	# ---- hyperparametros ---
	hyper  = np.array([1e2,10]) 
	#--------- Initial parameters --------------
	namepar = ["$Rc$","$g$"]
	mins    = np.array(([0.001, 2.01]))
	maxs    = np.array(([5.0, 4.0]))
	#--------- arguments of logLike function
	args     = (radii,pro,Rmax,hyper)
#========================================================
#--------- Dimension, walkers and inital positions -------------------------
ndim     = len(mins)
nwalkers = ndim*walkers_ratio
init_pos = [[np.random.uniform(low=mins[j],high=maxs[j],size=1)[0] for j in range(ndim)] for i in range(nwalkers)]

################## SAMPLER ##################################################
sampler = emcee.EnsembleSampler(nwalkers, ndim, LogPosterior, args=args)
sampler.a = 3.0
sampler.run_mcmc(init_pos, iters)

#------ analysis -----
iat   = int(max(sampler.acor))
print 'Mean Acceptance Fraction: ', np.mean(sampler.acceptance_fraction)
print 'Autocorrelation Time: ', sampler.acor

f=open(dir_+"Samples/"+str(model)+'-Rcut='+str(int(rcut))+'-a='+str(int(sampler.a))+'-wr='+str(walkers_ratio)+".out", "w")
for i in range(iters): 
	pos = sampler.chain[:,i,:]
	f.write("\n".join(["\t".join([str(q) for q in p]) for p in pos]))
	f.write("\n")
	f.flush()
f.close()

#--------- MAP ----------------------------------------
id_map= np.argmax(sampler.lnprobability[:,-1],axis=0)
MAP   = sampler.chain[id_map, -1, ]
#------------------------------------------------------
if model == "Gaussian":
	fit_Nr  = Number(radii,MAP[0],len(radii),Rmax)
	fit_Den = LikeCluster(radii,MAP[0],Rmax)
if model == "Plummer":
	fit_Nr  = Number(radii,MAP[0],len(radii),Rmax)
	fit_Den = LikeCluster(radii,MAP[0],Rmax)
if model == "King":
	fit_Nr  = Number(radii,MAP[0],MAP[1],len(radii),Rmax)
	fit_Den = LikeCluster(radii,MAP[0],MAP[1],Rmax)
if model == "EFF":
	fit_Nr  = Number(radii,MAP[0],MAP[1],len(radii),Rmax)
	fit_Den = LikeCluster(radii,MAP[0],MAP[1],Rmax)

mean_pop = np.zeros((iters,ndim))

for i in range(ndim):
	mean_pop[:,i] = (np.mean(sampler.chain[:,:, i],axis=0).T-np.mean(sampler.chain[:,:,i]))/(np.std(sampler.chain[:,:,i])/np.sqrt(nwalkers))
#-----------
x = sampler.chain
width= 2*iat
wds = int((iters-init)/width)
tau = np.zeros((2,wds))
wdw = np.arange(1,wds+1)*width
print "the band width is ",width
samples = sampler.chain[:, -init::, :].reshape((-1, ndim))

wsd = iat/2
wdsd = int((iters-init)/wsd)
print wdsd
std_pop  = np.zeros((wdsd,ndim))
for j in range(ndim):
	for i in range(wdsd):
		std_pop[i,j] = np.std(mean_pop[(wsd*i):(wsd*(i+1)-1),j])

#---------- Graficas de valores verdaderos --------
with PdfPages(dir_+'Analysis/'+str(model)+'-Rcut='+str(int(rcut))+'-a='+str(int(sampler.a))+'-wr='+str(walkers_ratio)+'.pdf') as pdf:

	plt.scatter(radii,Nr,s=1,color="black")
	plt.plot(radii,fit_Nr, linewidth=1,color="red")
	plt.ylim((0,1.1*max(Nr)))
	plt.xlim((0,1.1*rcut))
	pdf.savefig()  # saves the current figure into a pdf page
	plt.close()

	plt.scatter(bins,densi,s=1,color="black")
	plt.plot(radii,fit_Den, linewidth=1,color="red")
	pdf.savefig()  # saves the current figure into a pdf page
	plt.close()


	plt.figure()
	plt.plot(sampler.lnprobability.T,color='black')
	plt.ylim((np.mean(sampler.lnprobability[:,-100::])-2*np.std(sampler.lnprobability[:,-100::]),
			  np.mean(sampler.lnprobability[:,-100::])+2*np.std(sampler.lnprobability[:,-100::])))
	pdf.savefig()
	plt.close()

	for i in range(ndim):
		plt.plot(sampler.chain[:,:, i].T,color="black")
		plt.ylabel(namepar[i])
		plt.xlabel('Iterations')
		pdf.savefig()
		plt.close()

	for i in range(ndim):
		plt.plot(mean_pop[:,i],color="black")
	plt.xlabel('Iterations')
	plt.ylabel('Normalised mean')
	plt.axhline(y=-2,color = 'red')
	plt.axhline(y=2,color = 'red')
	plt.ylim((-10,10))
	pdf.savefig()
	plt.close()

	for i in range(ndim):
		plt.plot(std_pop[:,i],color="black")
	plt.xlabel('Window time')
	plt.ylabel('Std Normalised mean')
	pdf.savefig()
	plt.close()

	corner.corner(samples, labels=namepar)
	pdf.savefig()
	plt.close()

	# histdist = matplotlib.pyplot.hist(X, 100, normed=False,color="gray")
	# plt.plot(histdist[1],    2*N_f*(histdist[1][1]-histdist[1][0])*matplotlib.mlab.normpdf(histdist[1],t_mu_f,t_sd_f), linewidth=1,color='blue')
	# plt.plot(histdist[1],    N_c*(histdist[1][1]-histdist[1][0])*matplotlib.mlab.normpdf(histdist[1],t_mu_c,t_sd_c), linewidth=1,color='blue')
	# plt.plot(histdist[1],    2*N_s*(histdist[1][1]-histdist[1][0])*m_pi*matplotlib.mlab.normpdf(histdist[1],t_mu_f,t_sd_f), linewidth=1,color='red')
	# plt.plot(histdist[1],    N_s*(histdist[1][1]-histdist[1][0])*(1-m_pi)*matplotlib.mlab.normpdf(histdist[1],m_mu_c,m_sd_c), linewidth=1,color='red')
	# pdf.savefig()  # saves the current figure into a pdf page
	# plt.close()

	plt.figure(figsize=(6, 6))
	plt.xlim([init-10,iters+10])
	plt.xlabel('Window time')
	plt.ylabel('Autocorrelation Time')
	for i in range(ndim):
		tw  = np.zeros((nwalkers,wds))
		for k in range(wds):
			for j in range(nwalkers):
				tw[j,k] = iact(x[j,init:(init+wdw[k]),i])
			tau[0,k]=np.mean(tw[:,k])
			tau[1,k]=np.std(tw[:,k])
		special.errorfill(wdw, tau[0,:], tau[1,:],alpha_fill=0.1)
	pdf.savefig()
	plt.close()

	for i in range(ndim):
		y = (np.mean(sampler.chain[:,:, i],axis=0).T)#-np.mean(sampler.chain[:,:-1,i]))/(np.std(sampler.chain[:,:-1,i])/np.sqrt(nwalkers))
		plt.plot(acf(y, nlags=200, unbiased=False, fft=True))
		# plt.errorbar(wdw, tau[0,:], fmt='ro',yerr=tau[1,:], ecolor='black')
	pdf.savefig() 
	plt.close()
