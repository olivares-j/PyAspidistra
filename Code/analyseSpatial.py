
import sys
import os
import numpy as np
import pandas as pn
#########################################################################################
dir_  = os.path.expanduser('~') +"/PyAspidistra/"

exte  = str(sys.argv[1])
Rcut  = int(sys.argv[2])


models =np.array(["EFF","GDP","GKing","King","OGKing","RGDP"])

if exte == "None":
	dir_ext = "NoCentre"
elif exte == "Ctr":
	dir_ext = "Centre"

	#---- parameters ac,dc,rc,rt,alpha,beta,gamma ----------
	pnames      = np.array([r"$\alpha_c$",r"$\delta_c$",r"$r_c$",r"$r_t$",r"$\alpha$",r"$\beta$",r"$\gamma$"])
	idx_EFF    =    [1, 1 ,1,0  ,0    ,0   ,1   ]
	idx_GDP    =    [1, 1 ,1,0  ,1    ,1   ,1   ]
	idx_GKing  =    [1, 1 ,1,1  ,1    ,1   ,0   ]
	idx_King   =    [1, 1 ,1,1  ,0    ,0   ,0   ]
	idx_OGKing =    [1, 1 ,1,1  ,0    ,0   ,0   ]
	idx_RGDP   =    [1, 1 ,1,0  ,1    ,1   ,0   ]
	idx_MODS   = np.vstack([idx_EFF,idx_GDP,idx_GKing,idx_King,idx_OGKing,idx_RGDP])

elif exte == "Ell":
	dir_ext = "Elliptic"
	models  = models[1:] 

	#---- parameters ac,dc,rc,rt,alpha,beta,gamma ----------
	pnames     = np.array([r"$\alpha_c$",r"$\delta_c$",r"$\phi$",r"$r_{ca}$",r"$r_{ta}$",
					r"$r_{cb}$",r"$r_{tb}$",r"$\alpha$",r"$\beta$",r"$\gamma$"])
	idx_GDP    =    [1, 1, 1, 1, 0, 1, 0, 1, 1, 1]
	idx_GKing  =    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
	idx_King   =    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
	idx_OGKing =    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
	idx_RGDP   =    [1, 1, 1, 1, 0, 1, 0, 1, 1, 0]
	idx_MODS   = np.vstack([idx_GDP,idx_GKing,idx_King,idx_OGKing,idx_RGDP])


elif exte == "Seg":
	dir_ext = "Segregated"
	models  = models[1:]

	#---- parameters ac,dc,rc,rt,alpha,beta,gamma ----------
	pnames     = np.array([r"$\alpha_c$",r"$\delta_c$",r"$\phi$",r"$r_{ca}$",r"$r_{ta}$",
					r"$r_{cb}$",r"$r_{tb}$",r"$\alpha$",r"$\beta$",r"$\gamma$","Slope"])
	idx_GDP    =    [1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1]
	idx_GKing  =    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]
	idx_King   =    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1]
	idx_OGKing =    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1]
	idx_RGDP   =    [1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1]
	idx_MODS   = np.vstack([idx_GDP,idx_GKing,idx_King,idx_OGKing,idx_RGDP])


else :
	print("Extention not recognised!")
	print("Available ones are: None, Centre (Ctr), Elliptic (Ell), and Segregated (Seg")
	sys.exit()



############## Read evidences #########################
zs  = np.empty((len(models),2))
for i,model in enumerate(models):
	fevid  = dir_+'Analysis/'+dir_ext+"/"+model+'_'+str(Rcut)+"/0-stats.dat"
	zs[i]  = np.array(pn.read_csv(fevid,nrows=1,delim_whitespace=True,header=None,usecols=[5,7]))

############ Compute bayes factors ####################
bf = np.empty((len(models),len(models)))
for i,z in enumerate(zs[:,0]):
	bf[i] = np.exp(z-zs[:,0])

np.fill_diagonal(bf,zs[:,0])

df = pn.DataFrame(bf,columns=models)
df = df.rename(lambda x:models[x])
dir_out = dir_+'Analysis/BayesFactors/'
if not os.path.exists(dir_out): os.mkdir(dir_out)

def my_format(x):
	if x > 1e2:
		return ">1e2"
	if x < 1e-2 and x>0:
		return "<1e-2"
	return "{0:.2f}".format(x)

# -------- Writes Bayes factors -------------------------
fout = dir_out+"BF_"+dir_ext+"_"+str(Rcut)+".txt"
df.to_latex(fout,index_names=True,float_format=my_format)

############### READ MAPS ##############################
MAPs  = np.nan*np.ones((len(models),len(pnames)))
for i,model in enumerate(models):
	fevid  = dir_+'Analysis/'+dir_ext+"/"+model+'_'+str(Rcut)+"/"+model+"_map.txt"
	temp   = np.array(pn.read_csv(fevid,nrows=1,delim_whitespace=True,header=None))[0]
	MAPs[i,np.where(idx_MODS[i]==1)[0]] = temp

maps = pn.DataFrame(MAPs,columns=pnames)
maps = maps.rename(lambda x:models[x])

def my_format1(x):
	if np.isnan(x):
		return " "
	return "{0:.2f}".format(x)

# -------- Writes Bayes factors -------------------------
fout = dir_out+"MAPs_"+dir_ext+"_"+str(Rcut)+".txt"
maps.to_latex(fout,index_names=True,float_format=my_format1,escape=False)






 
