
import sys
import os
import numpy as np
import pandas as pn
from numpy.core.defchararray import add as adds

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def my_format(x):
	if x > 999:
		return ">999"
	if x < 1e-2 and x>0:
		return "<1e-2"
	return "{0:.2f}".format(x)

def my_format1(x):
	if np.isnan(x):
		return " "
	return "{0:.2f}".format(x)

#########################################################################################
dir_  = os.path.expanduser('~') +"/PyAspidistra/"

# exte  = str(sys.argv[1])

models = np.array(["EFF","GDP","GKing","King","OGKing","RGDP"])
Rcuts  = [11]
extes  = ["Ctr","Ell","Seg"]

dir_out = dir_+'Analysis/BayesFactors/'
if not os.path.exists(dir_out): os.mkdir(dir_out)

evs    = np.nan*np.ones((2,len(extes),len(models)))
Nrts   = np.empty((len(extes),len(Rcuts),3,50,2))
Nrtm   = np.empty((len(extes),len(Rcuts),3))
for e,exte in enumerate(extes):

	if exte == "None":
		dir_ext = "NoCentre"
	elif exte == "Ctr":
		dir_ext = "Centre"

		#---- parameters ac,dc,rc,rt,alpha,beta,gamma ----------
		pnames      = np.array(["$\\alpha_c$ [$^\circ$]","$\delta_c$ [$^\circ$]","$r_c$ [pc]","$r_t$ [pc]",
			"$\\alpha$","$\\beta$","$\gamma$"])
		idx_EFF    =    [1, 1 ,1,0  ,0    ,0   ,1   ]
		idx_GDP    =    [1, 1 ,1,0  ,1    ,1   ,1   ]
		idx_GKing  =    [1, 1 ,1,1  ,1    ,1   ,0   ]
		idx_King   =    [1, 1 ,1,1  ,0    ,0   ,0   ]
		idx_OGKing =    [1, 1 ,1,1  ,0    ,0   ,0   ]
		idx_RGDP   =    [1, 1 ,1,0  ,1    ,1   ,0   ]
		idx_MODS   = np.vstack([idx_EFF,idx_GDP,idx_GKing,idx_King,idx_OGKing,idx_RGDP])

	elif exte == "Ell":
		dir_ext = "Elliptic"

		#---- parameters ac,dc,rc,rt,alpha,beta,gamma ----------
		pnames     = np.array(["$\\alpha_c$ [$^\circ$]","$\delta_c$ [$^\circ$]","$\phi$ [rad]",
					"$r_{ca}$ [pc]","$r_{ta}$ [pc]","$r_{cb}$ [pc]","$r_{tb}$ [pc]","$\\alpha$",
					"$\\beta$","$\gamma$","$\epsilon_{rc}$","$\epsilon_{rt}$"])
		idx_EFF    =    [1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0]
		idx_GDP    =    [1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0]
		idx_GKing  =    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]
		idx_King   =    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1]
		idx_OGKing =    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1]
		idx_RGDP   =    [1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0]
		idx_MODS   = np.vstack([idx_EFF,idx_GDP,idx_GKing,idx_King,idx_OGKing,idx_RGDP])



	elif exte == "Seg":
		dir_ext = "Segregated"

		#---- parameters ac,dc,rc,rt,alpha,beta,gamma ----------
		pnames     = np.array(["$\\alpha_c$ [$^\circ$]","$\delta_c$ [$^\circ$]","$\phi$ [rad]",
						"$r_{ca}$ [pc]","$r_{ta}$ [pc]","$r_{cb}$ [pc]","$r_{tb}$ [pc]",
						"$\\alpha$","$\\beta$","$\\gamma$","$\kappa$ [pc mag$^{-1}$]","$\epsilon_{rc}$","$\epsilon_{rt}$"])
		idx_EFF    =    [1, 1 ,1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0]
		idx_GDP    =    [1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]
		idx_GKing  =    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1]
		idx_King   =    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1]
		idx_OGKing =    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1]
		idx_RGDP   =    [1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0]
		idx_MODS   = np.vstack([idx_EFF,idx_GDP,idx_GKing,idx_King,idx_OGKing,idx_RGDP])


	else :
		print("Extention not recognised!")
		print("Available ones are: None, Centre (Ctr), Elliptic (Ell), and Segregated (Seg")
		sys.exit()

	for r,Rcut in enumerate(Rcuts):

		############## Read evidences #########################
		zs  = np.empty((len(models),2))
		for i,model in enumerate(models):
			fevid  = dir_+'Analysis/'+dir_ext+"/"+model+'_'+str(Rcut)+"/0-stats.dat"
			zs[i]  = np.array(pn.read_csv(fevid,nrows=1,delim_whitespace=True,header=None,usecols=[5,7]))

		evs[r,e,:] = zs[:,0]

		############ Compute bayes factors ####################
		bf = np.empty((len(models),len(models)))
		print zs[:,0]
		for i,z in enumerate(zs[:,0]):
			bf[i] = np.exp(z-zs[:,0])

		np.fill_diagonal(bf,zs[:,0])

		df = pn.DataFrame(bf,columns=models)
		df = df.rename(lambda x:models[x])

		# -------- Writes Bayes factors -------------------------
		fout = dir_out+"BF_"+dir_ext+"_"+str(Rcut)+".txt"
		df.to_latex(fout,index_names=True,float_format=my_format)

		############### READ MAPS ##############################
		MAPs  = np.nan*np.ones((len(models),len(pnames)))
		for i,model in enumerate(models):
			fevid  = dir_+'Analysis/'+dir_ext+"/"+model+'_'+str(Rcut)+"/"+model+"_map.txt"
			MAPs[i,np.where(idx_MODS[i]==1)[0]] = np.array(pn.read_csv(fevid,
				nrows=1,delim_whitespace=True,header=None))

			#------ reads numbers ---------
			if model in ["GKing","King","OGKing"]:
				fnums  = dir_+'Analysis/'+dir_ext+"/"+model+'_'+str(Rcut)+"/"+model+"_numbers.txt"
				Nrtm[e,r,i-2] = np.array(pn.read_csv(fnums,
					nrows=1,delim_whitespace=True,header=None,skiprows=0))
				Nrts[e,r,i-2,:,:] = np.array(pn.read_csv(fnums,
					nrows=50,delim_whitespace=True,header=None,skiprows=1))

		maps = pn.DataFrame(MAPs,columns=pnames)
		maps = maps.rename(lambda x:models[x])

		# -------- Writes MAPS to Latex-------------------------
		fout = dir_out+"MAPs_"+dir_ext+"_"+str(Rcut)+".txt"
		maps.to_latex(fout,index_names=True,float_format=my_format1,escape=False)


evs = evs.reshape((2,-1))
name_all = np.array([models for ext in extes],dtype="S10").flatten()
for r,Rcut in enumerate(Rcuts):
	############ Compute bayes factors ####################
	bf = np.empty((len(evs[r]),len(evs[r])))
	for i,z in enumerate(evs[r]):
		bf[i] = np.exp(z-evs[r])

	np.fill_diagonal(bf,evs[r])

	df = pn.DataFrame(bf,columns=name_all)
	df = df.rename(lambda x:name_all[x])

	# -------- Writes Bayes factors -------------------------
	fout = dir_out+"BF_All_"+str(Rcut)+".txt"
	df.to_latex(fout,index_names=True,float_format=my_format,multirow=True)

for i,model in enumerate(["GKing","King","OGKing"]):
	pdf = PdfPages(dir_out+"Nsys_"+model+'.pdf')
	plt.figure()
	for e,ext in enumerate(extes):
		for r,Rcut in enumerate(Rcuts):
			plt.hist(Nrts[e,r,i,:,0],50,weights=Nrts[e,r,i,:,1],align="mid",
					histtype='step',alpha=0.5,label=ext+" "+str(Rcut))
	plt.title(model)
	plt.yscale("log")
	plt.legend()
	pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
	plt.close()
	pdf.close()

df = pn.DataFrame(Nrtm.reshape(len(extes),3),columns=["GKing","King","OGKing"])
df = df.rename(lambda x:extes[x])

# -------- Writes Bayes factors -------------------------
fout = dir_out+"Nsys_"+str(Rcut)+".txt"
df.to_latex(fout,index_names=True,float_format=my_format1,multirow=True)





 
