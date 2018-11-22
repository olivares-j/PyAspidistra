# Copyright 2017 Javier Olivares Romero
# Condor submit file 

DIR = /home/jromero/Repos/PyAspidistra/Code/
DIR_LOG = $(DIR)/Log

RCUT = 30


notification            = Error
notify_user 	        = avijer@gmail.com
getenv                  = True

universe                = vanilla
executable              = test
# executable              = runSpatial.py
initialdir              = $(DIR)

copy_to_spool           = FALSE
transfer_executable     = YES
should_transfer_files   = yes
when_to_transfer_output = ON_EXIT_OR_EVICT

requirements            = (TARGET.machine != "dance-headnode")
# && (TARGET.machine == "node08")


output                  = $(DIR_LOG)/$(cluster)_$(process).out
error                   = $(DIR_LOG)/$(cluster)_$(process).err
log                     = $(DIR_LOG)/$(cluster)_$(process).log

EXT = Ctr

arguments               = EFF $(RCUT) $(EXT) 
queue
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


# EXT  = Ell
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


# EXT  = Seg

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

