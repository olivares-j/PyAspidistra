# Condor submit file 

DIR     = /pcdisk/boneym5/jolivares/PyAspidistra/Code/
DIR_LOG = $(DIR)/Log

RCUT = 11.5


notification            = Error
notify_user 	        = avijer@gmail.com
getenv                  = True

universe                = vanilla
executable              = runSpatial.py
initialdir              = $(DIR)
should_transfer_files   = yes
when_to_transfer_output = on_exit

requirements            = (Machine == "kool.cab.inta-csic.es")


output                  = $(DIR_LOG)/$(cluster)_$(process).out
error                   = $(DIR_LOG)/$(cluster)_$(process).err
log                     = $(DIR_LOG)/$(cluster)_$(process).log

EXT  = Ctr

# arguments               = EFF $(RCUT) $(EXT) 
# queue
# arguments               = GDP $(RCUT) $(EXT) 
# queue
# arguments               = GKing $(RCUT) $(EXT) 
# queue
# arguments               = King $(RCUT) $(EXT) 
# queue
# arguments               = OGKing $(RCUT) $(EXT) 
# queue
# arguments               = RGDP $(RCUT) $(EXT) 
# queue


EXT  = Ell
arguments               = EFF $(RCUT) $(EXT) 
queue
arguments               = GDP $(RCUT) $(EXT) 
queue
arguments               = GKing $(RCUT) $(EXT) 
queue
arguments               = King $(RCUT) $(EXT) 
queue
arguments               = OGKing $(RCUT) $(EXT) 
queue
arguments               = RGDP $(RCUT) $(EXT) 
queue


EXT  = Seg

arguments               = EFF $(RCUT) $(EXT) 
queue
arguments               = GDP $(RCUT) $(EXT) 
queue
arguments               = GKing $(RCUT) $(EXT) 
queue
arguments               = King $(RCUT) $(EXT) 
queue
arguments               = OGKing $(RCUT) $(EXT) 
queue
arguments               = RGDP $(RCUT) $(EXT) 
queue

