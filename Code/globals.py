############ GLOBAL VARIABLES #################################################
dir_  = "/home/jromero/Repos/PyAspidistra/"  # Specify the full path to the Aspidistra path
real  = False
Ntot  = 10000  # Number of stars if synthetic (real = False)

Dist    = 309
centre  = [289.10,-16.38]

dir_analysis = dir_ + "Analysis/"

file_members = dir_+'Data/'+'members_ALL.csv' # Put here the name of the members file.
#"Make sure you have data in right order! (R.A., Dec. Probability, Band,e_band)")
list_observables = ["RAJ2000","DEJ2000","probability","G","e_G"]

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