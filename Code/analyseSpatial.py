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
import pandas as pn
from numpy.core.defchararray import add as adds

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


from chainconsumer import ChainConsumer

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

def my_format2(x):
	if np.isnan(x):
		return " "
	return "{0:.0f}".format(x)

#########################################################################################
dir_  = os.getcwd().replace("Code","")



models = np.array(["EFF","GDP","GKing","King","OGKing","RGDP"])
Rcuts  = [33]
extes  = ["Ctr","Ell","Seg"]
lstys  = ['-','--',':','-.']


dir_out = dir_+'Analysis/BayesFactors'
if not os.path.exists(dir_out): os.mkdir(dir_out)

evs    = np.nan*np.ones((2,len(extes),len(models)))
Nrts   = np.empty((len(extes),len(Rcuts),3,50,2))
Nrtm   = np.empty((len(extes),len(Rcuts),3))
Mrts   = np.empty((len(extes),len(Rcuts),3,100,2))
Mrtm   = np.empty((len(extes),len(Rcuts),3))
Meps   = np.empty((len(extes)-1,len(Rcuts),3,100,2))
Mepm   = np.empty((len(extes)-1,len(Rcuts),3))
for e,exte in enumerate(extes):

	if exte == "None":
		dir_ext = "NoCentre"
	elif exte == "Ctr":
		dir_ext = "Centre"

		#---- parameters ac,dc,rc,rt,alpha,beta,gamma ----------
		pnames     = ["$\\alpha_c$","$\delta_c$","$r_c$","$r_t$","$\\alpha$","$\\beta$","$\gamma$"]
		units      = ["[$^\circ$]"," [$^\circ$]"," [pc]"," [pc]","$\\alpha$","$\\beta$","$\gamma$"]
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
		pnames     = ["$\\alpha_c$","$\delta_c$","$\phi$",
					"$r_{ca}$","$r_{ta}$","$r_{cb}$","$r_{tb}$","$\\alpha$",
					"$\\beta$","$\gamma$","$\epsilon_{rc}$","$\epsilon_{rt}$"]

		units      = ["[$^\circ$]","[$^\circ$]","[rad]",
					"[pc]","[pc]","[pc]","[pc]","",
					"","","",""]

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
		pnames     = [r"$\alpha_c$",r"$\delta_c$",r"$\phi$",
						r"$r_{ca}$",r"$r_{ta}$",r"$r_{cb}$",r"$r_{tb}$",
						r"$\alpha$",r"$\beta$",r"$\gamma$",r"$\kappa$",
						r"$\epsilon_{rc}$",r"$\epsilon_{rt}$"]

		units      = [r"[$^\circ$]",r"[$^\circ$]","[rad]",
						"[pc]","[pc]","[pc]","[pc]",
						"","","",r"[pc mag$^{-1}$]","",""]
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
		for i,z in enumerate(zs[:,0]):
			bf[i] = np.exp(z-zs[:,0])

		np.fill_diagonal(bf,zs[:,0])

		df = pn.DataFrame(bf,columns=models)
		df = df.rename(lambda x:models[x])

		# -------- Writes Bayes factors -------------------------
		fout = dir_out+"/BF_"+dir_ext+"_"+str(Rcut)+".txt"
		df.to_latex(fout,index_names=True,float_format=my_format)

		# ############### CHAIN CONSUMER ##############################
		# ------------------ Start chains -------------------------------------
		c = ChainConsumer()
		for i,model in enumerate(models):
			file_chain  = dir_+'Analysis/'+dir_ext+"/"+model+'_'+str(Rcut)+"/chain.csv"
			chain  = np.array(pn.read_csv(file_chain))
			idx_names = np.where(idx_MODS[i]==1)[0]
			names = [pnames[i] for i in idx_names]
			c.add_chain(chain,parameters=names,name=model)

		c.configure(statistics="cumulative")

		#--------------- Plot --------------------------------------
		plt.figure()
		c.plotter.plot_summary()

		plt.savefig(dir_out+"/Models_"+str(Rcut)+".pdf", bbox_inches='tight')  # saves the current figure into a pdf page
		plt.close()
		#------------------------------------------------------------------------------

		#------------------------ Table ----------------------------------------------------------------
		# print(c.analysis.get_summary())
		table = c.analysis.get_latex_table(caption="Summary", label="tab:summary")

		#---------------- Writes table to file-------------------------
		fout = dir_out+"/Parameters_"+dir_ext+"_"+str(Rcut)+".tex"
		
		with open(fout, "w") as text_file:
			text_file.write(table)
		#-------------------------------------------------------------------------------------------------
		##################################################################################################3


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
	fout = dir_out+"/BF_All_"+str(Rcut)+".txt"
	df.to_latex(fout,index_names=True,float_format=my_format,multirow=True)


##################### NUMBER OF SYSTEMS ###################################################

plt.figure()
for i,model in enumerate(["GKing","King","OGKing"]):
	for e,ext in enumerate(extes):
		for r,Rcut in enumerate(Rcuts):
			plt.hist(Nrts[e,r,i,:,0],50,weights=Nrts[e,r,i,:,1],align="mid",
					histtype='step',alpha=0.7,label=model+"+"+ext)
plt.ylabel("Density")
plt.xlabel("Total number of systems")
plt.yscale("log")
# plt.ylim(ymin=1e-5)
plt.legend()
plt.savefig(dir_out+"/Nsys_"+str(Rcut)+".pdf",bbox_inches='tight')  # saves the current figure into a pdf page
plt.close()


df = pn.DataFrame(Nrtm.reshape(len(extes),3),columns=["GKing","King","OGKing"])
df = df.rename(lambda x:extes[x])

# -------- Writes numbers -------------------------
fout = dir_out+"/Nsys_"+str(Rcut)+".txt"
df.to_latex(fout,index_names=True,float_format=my_format2,multirow=True)

################################## MASS ##############################################3

pdf = PdfPages(dir_out+"/Msys_"+str(Rcut)+".pdf")
plt.figure()
for e,ext in enumerate(extes):
	for i,model in enumerate(["GKing","King","OGKing"]):
			for r,Rcut in enumerate(Rcuts):
				plt.hist(Mrts[e,r,i,:,0],50,weights=Mrts[e,r,i,:,1],align="mid",
						alpha=0.7,histtype='step',label=model,ls=lstys[i])
	plt.ylabel("Density")
	plt.xlabel("Total mass [$M_\odot$]")
	plt.yscale("log")
	plt.ylim(ymin=1e-5)
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
	       ncol=3, mode="expand", borderaxespad=0.)
	pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
	plt.close()
pdf.close()


df = pn.DataFrame(Mrtm.reshape(len(extes),3),columns=["GKing","King","OGKing"])
df = df.rename(lambda x:extes[x])

# -------- Writes masses -------------------------
fout = dir_out+"/Msys_"+str(Rcut)+".txt"
df.to_latex(fout,index_names=True,float_format=my_format2,multirow=True)


print("Done!")





 
