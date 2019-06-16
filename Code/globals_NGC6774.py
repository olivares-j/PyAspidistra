##################################### GLOBAL VARIABLES ######################################################
#-------------------- Paths and knobs -----------------------------------------------
dir_  = "/home/javier/Repositories/PyAspidistra/" # Specify the full path to the Aspidistra directory
real  = True   # Boolean real or synthetic data
Ntot  = 10000  # Number of stars if synthetic (real = False)
percentiles = [16,50,84] # The percentiles of the output statistics

dir_analysis = dir_ + "NGC6774/"
#---------------------------------------------------------------------------------------

#----------------------------- Data ----------------------------------------------------
file_members = dir_+'Data/'+'NGC_6774.csv' # Put here the name of the members file.
#"Make sure you have data in right order! (R.A., Dec. Probability, Band,e_band)")
list_observables = ["RAJ2000","DEJ2000","probability","G","G_error"]
mag_limit = 25.0 # Upper limit in magnitud (objects above this limit will be discarded)
pro_limit = 0.5  # Lower limit in membership probability (objects below this limit will be discarded)
#---------------------------------------------------------------------------------------

#+++++++++++++++++++++ Clusters values +++++++++++++++++++++++++
distance = 309
Rcut     = 20
#------- Initial values of Centre ------
ctr_ra   = 289.10
ctr_dec  = -16.38
centre   = [ctr_ra,ctr_dec]
#--------------------------------------
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#############################################################################################################


#=========================== Profiles ======================================================================================
#+++++++++++++ Hyper-parameters +++++++++++++++++++++++++++++++
upper_exp = 100.0 # Upper limit of exponential prior for exponent parameters
scale_exp = 1.0   # Scale of exponential prior for exponents parameters 
scale_rc  = 1.0   # Scale of the Half-Cauchy prior for the core radius
scale_rt  = 10.0  # Scale of the Half-Cauchy prior for the tidal radius
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#------------------------- EFF ----------------------------------------------------------------------------------------------
profile = {"model":"EFF",
			"extension":"None",
			"parameter_names":["$r_c$ [pc]","$\gamma$"],
			"initial_values":[2.0,3.0],
			"parameter_interval":[[0,4],[2,4]],
			"hyper-parameters":[scale_rc,upper_exp,scale_exp]}

# profile = {"model":"EFF",
# 			"extension":"Ctr",
# 			"parameter_names":["$\\alpha_c\ \ [^\circ]$","$\\delta_c\ \ [^\circ]$","$r_c$ [pc]","$\gamma$"],
# 			"initial_values":[ctr_ra,ctr_dec,2.0,3.0],
# 			"parameter_interval":[[ctr_ra-1,ctr_ra+1],[ctr_dec-1,ctr_dec+1],[0,4],[2,4]],
# 			"hyper-parameters":[1,1,scale_rc,upper_exp,scale_exp]}

# profile = {"model":"EFF",
# 			"extension":"Ell",
# 			"parameter_names":["$\\alpha_c\ \ [^\circ]$","$\\delta_c\ \ [^\circ]$","$\phi$ [radians]","$r_{ca}$ [pc]","$r_{cb}$ [pc]","$\gamma$"],
# 			"initial_values":[ctr_ra,ctr_dec,3.14/4,5.0,1.0,3.0],
# 			"parameter_interval":[[ctr_ra-1,ctr_ra+1],[ctr_dec-1,ctr_dec+1],[0,3.14],[0,10],[0,5],[2,4]],
# 			"id_rc":[3,4],
# 			"hyper-parameters":[1,1,scale_rc,upper_exp,scale_exp]}

# profile = {"model":"EFF",
# 			"extension":"Seg",
# 			"parameter_names":["$\\alpha_c\ \ [^\circ]$","$\\delta_c\ \ [^\circ]$","$\phi$ [radians]",
# 					"$r_{ca}$ [pc]","$r_{cb}$ [pc]","$\gamma$","$\kappa$ [pc$\cdot \\rm{mag}^{-1}$]"],
# 			"initial_values":[ctr_ra,ctr_dec,np.pi/4,5.0,1.0,3.0,0.0],
# 			"parameter_interval":[[ctr_ra-1,ctr_ra+1],[ctr_dec-1,ctr_dec+1],[0,np.pi],[0,10],[0,5],[2,4],[-0.6,1.2]],
# 			"id_rc":[3,4],
# 			"hyper-parameters":[1,1,scale_rc,upper_exp,scale_exp,0,0.5]}
#-----------------------------------------------------------------------------------------------------------------------------

#------------------------- GDP ----------------------------------------------------------------------------------------------
# profile = {"model":"GDP",
# 			"extension":"None",
# 			"parameter_names":["$r_c$ [pc]","$\\alpha$","$\\beta$","$\gamma$"],
# 			"initial_values":[2.0,0.5,2.0,0.1],
# 			"parameter_interval":[[0,10],[0,2],[0,10],[0,1]],
# 			"hyper-parameters":[scale_rc,upper_exp,scale_exp]}

# profile = {"model":"GDP",
# 			"extension":"Ctr",
# 			"parameter_names":["$\\alpha_c\ \ [^\circ]$","$\\delta_c\ \ [^\circ]$","$r_c$ [pc]","$\\alpha$","$\\beta$","$\gamma$"],
# 			"initial_values":[ctr_ra,ctr_dec,2.0,0.5,2.0,0.1],
# 			"parameter_interval":[[ctr_ra-1,ctr_ra+1],[ctr_dec-1,ctr_dec+1],[0,10],[0,2],[0,10],[0,1]],
# 			"hyper-parameters":[1,1,scale_rc,upper_exp,scale_exp]}

# profile = {"model":"GDP",
# 			"extension":"Ell",
# 			"parameter_names":["$\\alpha_c\ \ [^\circ]$","$\\delta_c\ \ [^\circ]$","$\phi$ [radians]","$r_{ca}$ [pc]","$r_{cb}$ [pc]",
#  					"$\\alpha$","$\\beta$","$\gamma$"],
# 			"initial_values":[ctr_ra,ctr_dec,np.pi/4,2.0,2.0,0.5,2.0,0.0],
# 			"parameter_interval":[[ctr_ra-1,ctr_ra+1],[ctr_dec-1,ctr_dec+1],[0,np.pi],[0,10],[0,10],[0,2],[0,10],[0,2]],
# 			"id_rc":[3,4],
# 			"hyper-parameters":[1,1,scale_rc,upper_exp,scale_exp]}

# profile = {"model":"GDP",
# 			"extension":"Seg",
# 			"parameter_names":["$\\alpha_c\ \ [^\circ]$","$\\delta_c\ \ [^\circ]$","$\phi$ [radians]","$r_{ca}$ [pc]","$r_{cb}$ [pc]",
# 					"$\\alpha$","$\\beta$","$\gamma$","$\kappa$ [pc$\cdot \\rm{mag}^{-1}$]"],
# 			"initial_values":[ctr_ra,ctr_dec,3.14/4,2.0,2.0,0.5,2.0,0.0,0.0],
# 			"parameter_interval":[[ctr_ra-1,ctr_ra+1],[ctr_dec-1,ctr_dec+1],[0,3.14],[0,10],[0,10],[0,2],[0,10],[0,2],[-0.6,1.2]],
# 			"id_rc":[3,4],
# 			"hyper-parameters":[1,1,scale_rc,upper_exp,scale_exp,0,0.5]}
#-----------------------------------------------------------------------------------------------------------------------------


#------------------------- GKing ----------------------------------------------------------------------------------------------
# profile = {"model":"GKing",
# 			"extension":"None",
# 			"parameter_names":["$r_c$ [pc]","$r_t$ [pc]","$\\alpha$","$\\beta$"],
# 			"initial_values":[2.0,30.0,0.5,2.0],
# 			"parameter_interval":[[0,5],[10,100],[0,2],[0,5]],
# 			"id_rt":[3],
# 			"hyper-parameters":[scale_rc,scale_rt,upper_exp,scale_exp]}

# profile = {"model":"GKing",
# 			"extension":"Ctr",
# 			"parameter_names":["$\\alpha_c\ \ [^\circ]$","$\\delta_c\ \ [^\circ]$",
# 								"$r_c$ [pc]","$r_t$ [pc]","$\\alpha$","$\\beta$"],
# 			"initial_values":[ctr_ra,ctr_dec,2.0,30.0,0.5,2.0],
# 			"parameter_interval":[[ctr_ra-1,ctr_ra+1],[ctr_dec-1,ctr_dec+1],[0,5],[10,100],[0,2],[0,5]],
# 			"id_rt":[3],
# 			"hyper-parameters":[1,1,scale_rc,scale_rt,upper_exp,scale_exp]}

# profile = {"model":"GKing",
# 			"extension":"Ell",
# 			"parameter_names":["$\\alpha_c\ \ [^\circ]$","$\\delta_c\ \ [^\circ]$","$\phi$ [radians]","$r_{ca}$ [pc]","$r_{ta}$ [pc]", 
# 					"$r_{cb}$ [pc]","$r_{tb}$ [pc]","$\\alpha$","$\\beta$"],
# 			"initial_values":[ctr_ra,ctr_dec,3.14/4,2.0,20.0,2.0,20.0,0.5,2.0],
# 			"parameter_interval":[[ctr_ra-1,ctr_ra+1],[ctr_dec-1,ctr_dec+1],[0,3.14],[0,5],[10,100],[0,5],[10,100],[0,2],[0,5]],
# 			"id_rc": [3,5],
# 			"id_rt": [4,6],
# 			"hyper-parameters":[1,1,scale_rc,scale_rt,upper_exp,scale_exp]}

# profile = {"model":"GKing",
# 			"extension":"Seg",
# 			"parameter_names":["$\\alpha_c\ \ [^\circ]$","$\\delta_c\ \ [^\circ]$","$\phi$ [radians]","$r_{ca}$ [pc]","$r_{ta}$ [pc]", 
# 					"$r_{cb}$ [pc]","$r_{tb}$ [pc]","$\\alpha$","$\\beta$","$\kappa$ [pc$\cdot \\rm{mag}^{-1}$]"],
# 			"initial_values":[ctr_ra,ctr_dec,3.14/4,2.0,20.0,2.0,20.0,0.5,2.0,0.0],
# 			"parameter_interval":[[ctr_ra-1,ctr_ra+1],[ctr_dec-1,ctr_dec+1],[0,3.14],[0,5],[10,100],[0,5],[10,100],[0,2],[0,5],[-0.6,1.2]],
# 			"id_rc": [3,5],
# 			"id_rt": [4,6],
# 			"hyper-parameters":[1,1,scale_rc,scale_rt,upper_exp,scale_exp,0,0.5]}
#-----------------------------------------------------------------------------------------------------------------------------

#------------------------- King ----------------------------------------------------------------------------------------------
# profile = {"model":"King",
# 			"extension":"None",
# 			"parameter_names":["$r_c$ [pc]","$r_t$ [pc]"],
# 			"initial_values":[2.0,30.0],
# 			"parameter_interval":[[0,5],[10,100]],
# 			"id_rt":[3],
# 			"hyper-parameters":[scale_rc,scale_rt]}

# profile = {"model":"King",
# 			"extension":"Ctr",
# 			"parameter_names":["$\\alpha_c\ \ [^\circ]$","$\\delta_c\ \ [^\circ]$","$r_c$ [pc]","$r_t$ [pc]"],
# 			"initial_values":[ctr_ra,ctr_dec,2.0,30.0],
# 			"parameter_interval":[[ctr_ra-1,ctr_ra+1],[ctr_dec-1,ctr_dec+1],[0,5],[10,100]],
# 			"id_rt":[3],
# 			"hyper-parameters":[1,1,scale_rc,scale_rt]}

# profile = {"model":"King",
# 			"extension":"Ell",
# 			"parameter_names":["$\\alpha_c\ \ [^\circ]$","$\\delta_c\ \ [^\circ]$","$\phi$ [radians]","$r_{ca}$ [pc]","$r_{ta}$ [pc]", 
# 					"$r_{cb}$ [pc]","$r_{tb}$ [pc]"],
# 			"initial_values":[ctr_ra,ctr_dec,3.14/4,2.0,20.0,2.0,20.0],
# 			"parameter_interval":[[ctr_ra-1,ctr_ra+1],[ctr_dec-1,ctr_dec+1],[0,3.14],[0,5],[10,100],[0,5],[10,100]],
# 			"id_rc": [3,5],
# 			"id_rt": [4,6],
# 			"hyper-parameters":[1,1,scale_rc,scale_rt]}

# profile = {"model":"King",
# 			"extension":"Seg",
# 			"parameter_names":["$\\alpha_c\ \ [^\circ]$","$\\delta_c\ \ [^\circ]$","$\phi$ [radians]","$r_{ca}$ [pc]","$r_{ta}$ [pc]", 
# 					"$r_{cb}$ [pc]","$r_{tb}$ [pc]""$\kappa$ [pc$\cdot \\rm{mag}^{-1}$]"],
# 			"initial_values":[ctr_ra,ctr_dec,3.14/4,2.0,20.0,2.0,20.0,0.0],
# 			"parameter_interval":[[ctr_ra-1,ctr_ra+1],[ctr_dec-1,ctr_dec+1],[0,3.14],[0,5],[10,100],[0,5],[10,100],[-0.6,1.2]],
# 			"id_rc": [3,5],
# 			"id_rt": [4,6],
# 			"hyper-parameters":[1,1,scale_rc,scale_rt,0,0.5]}
#-----------------------------------------------------------------------------------------------------------------------------

#------------------------- OGKing ----------------------------------------------------------------------------------------------
# profile = {"model":"OGKing",
# 			"extension":"None",
# 			"parameter_names":["$r_c$ [pc]","$r_t$ [pc]"],
# 			"initial_values":[2.0,30.0],
# 			"parameter_interval":[[0,5],[10,100]],
# 			"id_rt":[3],
# 			"hyper-parameters":[scale_rc,scale_rt]}

# profile = {"model":"OGKing",
# 			"extension":"Ctr",
# 			"parameter_names":["$\\alpha_c\ \ [^\circ]$","$\\delta_c\ \ [^\circ]$","$r_c$ [pc]","$r_t$ [pc]"],
# 			"initial_values":[ctr_ra,ctr_dec,2.0,30.0],
# 			"parameter_interval":[[ctr_ra-1,ctr_ra+1],[ctr_dec-1,ctr_dec+1],[0,5],[10,100]],
# 			"id_rt":[3],
# 			"hyper-parameters":[1,1,scale_rc,scale_rt]}

# profile = {"model":"OGKing",
# 			"extension":"Ell",
# 			"parameter_names":["$\\alpha_c\ \ [^\circ]$","$\\delta_c\ \ [^\circ]$","$\phi$ [radians]","$r_{ca}$ [pc]","$r_{ta}$ [pc]", 
# 					"$r_{cb}$ [pc]","$r_{tb}$ [pc]"],
# 			"initial_values":[ctr_ra,ctr_dec,3.14/4,2.0,30.0,2.0,30.0],
# 			"parameter_interval":[[ctr_ra-1,ctr_ra+1],[ctr_dec-1,ctr_dec+1],[0,3.14],[0,4],[10,100],[0,4],[10,100]],
# 			"id_rc": [3,5],
# 			"id_rt": [4,6],
# 			"hyper-parameters":[1,1,scale_rc,scale_rt]}

# profile = {"model":"OGKing",
# 			"extension":"Seg",
# 			"parameter_names":["$\\alpha_c\ \ [^\circ]$","$\\delta_c\ \ [^\circ]$","$\phi$ [radians]","$r_{ca}$ [pc]","$r_{ta}$ [pc]", 
# 					"$r_{cb}$ [pc]","$r_{tb}$ [pc]""$\kappa$ [pc$\cdot \\rm{mag}^{-1}$]"],
# 			"initial_values":[ctr_ra,ctr_dec,3.14/4,2.0,30.0,2.0,30.0,0.0],
# 			"parameter_interval":[[ctr_ra-1,ctr_ra+1],[ctr_dec-1,ctr_dec+1],[0,3.14],[0,5],[10,100],[0,5],[10,100],[-0.6,1.2]],
# 			"id_rc": [3,5],
# 			"id_rt": [4,6],
# 			"hyper-parameters":[1,1,scale_rc,scale_rt,0,0.5]}
#-----------------------------------------------------------------------------------------------------------------------------

#------------------------- RGDP ----------------------------------------------------------------------------------------------
# profile = {"model":"RGDP",
# 			"extension":"None",
# 			"parameter_names":["$r_c$ [pc]","$\\alpha$","$\\beta$"],
# 			"initial_values":[2.0,0.5,2.0],
# 			"parameter_interval":[[0,10],[0,2],[0,10]],
# 			"hyper-parameters":[scale_rc,upper_exp,scale_exp]}

# profile = {"model":"RGDP",
# 			"extension":"Ctr",
# 			"parameter_names":["$\\alpha_c\ \ [^\circ]$","$\\delta_c\ \ [^\circ]$","$r_c$ [pc]","$\\alpha$","$\\beta$"],
# 			"initial_values":[ctr_ra,ctr_dec,2.0,0.5,2.0],
# 			"parameter_interval":[[ctr_ra-1,ctr_ra+1],[ctr_dec-1,ctr_dec+1],[0,10],[0,2],[0,10]],
# 			"hyper-parameters":[1,1,scale_rc,upper_exp,scale_exp]}

# profile = {"model":"RGDP",
# 			"extension":"Ell",
# 			"parameter_names":["$\\alpha_c\ \ [^\circ]$","$\\delta_c\ \ [^\circ]$","$\phi$ [radians]","$r_{ca}$ [pc]","$r_{cb}$ [pc]",
#  					"$\\alpha$","$\\beta$"],
# 			"initial_values":[ctr_ra,ctr_dec,np.pi/4,2.0,2.0,0.5,2.0],
# 			"parameter_interval":[[ctr_ra-1,ctr_ra+1],[ctr_dec-1,ctr_dec+1],[0,np.pi],[0,10],[0,10],[0,2],[0,10]],
# 			"id_rc":[3,4],
# 			"hyper-parameters":[1,1,scale_rc,upper_exp,scale_exp]}

# profile = {"model":"RGDP",
# 			"extension":"Seg",
# 			"parameter_names":["$\\alpha_c\ \ [^\circ]$","$\\delta_c\ \ [^\circ]$","$\phi$ [radians]","$r_{ca}$ [pc]","$r_{cb}$ [pc]",
#  					"$\\alpha$","$\\beta$","$\kappa$ [pc$\cdot \\rm{mag}^{-1}$]"],
# 			"initial_values":[ctr_ra,ctr_dec,3.14/4,2.0,2.0,0.5,2.0,0.0],
# 			"parameter_interval":[[ctr_ra-1,ctr_ra+1],[ctr_dec-1,ctr_dec+1],[0,3.14],[0,10],[0,10],[0,2],[0,10],[-0.6,1.2]],
# 			"id_rc":[3,4],
# 			"hyper-parameters":[1,1,scale_rc,upper_exp,scale_exp,0,0.5]}
#-----------------------------------------------------------------------------------------------------------------------------