# PyAspidistra
Infer parameters of projected spatial density profiles and select best model using pyMultiNest

This code was used to obtain the results of Olivares et al. (2017b, submitted to A&A).

The data sets of Olivares et al. (2017a, submitted to A&A) and Bouy et al. (2015) are in the folder "Data".

The code is executed using the following syntaxis:

python runSpatial.py arg1 arg2 [arg3]

with arg1 being the model (profile), arg2 the maximum radius coverage (in pc), and arg3 the extension to the model.

Possible models are:
- EFF : the Elson et al. 1987
- GDP : similar to that of Lauer et al. (1995)
- King: classical King 1962 profile
- GKing : introduced in Olivares et al. (2017b)
- OGKing : the optimised GKing 
- RGDP : the restricted GDP

Possible extensions are:
- Ctr : the centre of the cluster is inferred as parameter
- Ell : in addition to the centre, the models have ellipticity
- Seg : in addition to the centre and ellipticity the models are luminosity segregated.


## Requirements
In order to run this code you need the following libraries.
-pandas
-numpy
-scipy
-pymultinest
-importlib
-corner
-matplotlib

##Disclaimer
This code has not been yet tested. If you experience crashes, please let us know at javier.olivares@univ-grenoble-alpes.fr
