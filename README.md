# Kilonova-analysis

This folder contains the spectral modelling of the kilonova (KN) AT2017gfo presented in the paper, Sneppen et al (2023), "Spherical symmetry in the kilonova AT2017gfo/GW170817", Nature, 614.

Relatively simple models have proven quite succesfull in describing the observed properties of AT2017gfo. The continuum is well-modelled as a blackbody (see observational constraints in Sneppen et al (2023) "On the Blackbody Spectrum of Kilonovae", ApJ, 1955). Spectral perturbation from the blackbody (e.g. Absorption below 390 nm, P Cygni at 760\,nm and 1$\mu$m) follow from the strongest transitions of a light r-process atmosphere. These features are produced by abundant elements (r-process peak elements) positioned at leftmost part of the periodic table (due to their low partition functions - analogous to the Na, Mg, Ca lines in solar spectra or Ca lines in SNe). 



#### The parameters of the fit: 
##### Continuum Emission 
N, Overall normalisation describes ratio between intrinsic luminosity of blackbody and observed luminosity

T, Blackbody Temperature

##### P-cygni parameters
𝑣_𝑚𝑎𝑥,𝑣_𝑝ℎ𝑜𝑡: Describes overall velocity, sets wavelength-scale

𝑣_𝑒, 𝜏: Velocity stratification and optical depth of line 𝜏(𝑣)=𝜏𝑒^(−(𝑣_𝑟𝑒𝑓−𝑣)/𝑣_𝑒 )

𝑂_𝑜𝑐𝑐: sets asymmetry between absorption and emission. 

ratio_vel: eccentricity of photosphere, if fitting an ellipsoidal photosphere

Theta_inc: Inclination angle, if fitting an ellipsoidal photosphere
