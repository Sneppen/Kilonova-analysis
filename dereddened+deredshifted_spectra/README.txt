
A uniform reduction and calibration of X-shooter data for AT2017gfo
and comparison with HST grism spectra 
========================================================================

J. Selsing (DARK) - reduced all 10 epochs of X-shooter data
J. Gillanders (QUB) - corrected to photometry, merged, smoothed and redshift/extinction corrected
S. Smartt (QUB) - checked and tested products, photometry selection 
E. Pian (INAF-Bologna), P D'Avanzo (INAF-Brera) - data provision and oversight
N. Tanvir - - data provision and oversight

Use of these data are contigent on citing both the following papers as
the orignal data source. 

Pian et al. 2017, Nature, 551, 67
Smartt et al. 2017, Nature 551, 75

And if the HST spectra are used, then this paper must be cited 
Tanvir et al. 2017, ApJ, 848, L27

You can acknowledge ENGRAVE for provision of the data
(www.engrave-eso.org), but citation of these two papers is obligatory
and proper scientific practice. 

Edit (J. Gillanders - 06/05/2020)
======================================
Re-calibrated the +1.5d UVB arm to new photometry point, as the
GROND point used appeared to be discrepant with the other available
photometry at that epoch. Spectrum was shifted by a constant to match
the weighted-average DECam and Sinistro g-band photometry point taken
at the epoch of the first spectrum. All affected spectra on ERDA have
been updated to reflect this update.
Photometry value used: 18.66+-0.07

X-shooter data sources and reductions
======================================

X-shooter observed AT2017gfo for 10 consecutive nights from 1.5 days
after discovery of GW170817 to 11.5 after. Two papers presented these
data in 2017 (Pian et al. 2017, Nature, 551, 67 ; Smartt et al. 2017,
Nature 551, 75). We have re-reduced all these spectra in a single
consistent fashion and provided a uniform calibration to observed
photometry. The observed photometry at each epoch was chosen carefully
from all of the published values (see below) and the package SMS
(https://github.com/cinserra/S3) was used to Measure synthetic
photometry of the X-shooter spectra ("Synthetic Magnitudes from
Spectra" SMS is the algorithm that was used, which was checked with
IRAF/sbands).

J. Selsing re-reduced all the spectra from Pian et al. (2017) and
Smartt et al. (2017), providing wavelength, flux calibrated, telluric
corrected 1D spectra at a dispersion of 0.2 Angs per pixel.  Each arm
UVB, VIS and NIR were reduced separately, and the following procedures
were implemented to correct to photometry, stitch the arms together and 
finally deredden and correct to restframe 

Photometric calibrations 
=======================================
J. Gillanders calibrated these to the selected grizJHK photometry
(checked by S. Smartt). A python program was written to apply a linear
scaling function to the spectra to fit them to the photometry.  We
chose a linear scaling function to avoid tampering too much with the
Selsing spectral flux calibration.

Typical percentage corrections range across the spectra, but almost all
required scaling  less than about -25% to +25%. A couple of the spectra
end regions required higher corrections (up to a maximum of +70% to the blue end
of +10.40d NIR arm - extreme case).

After correction, the synthetic photometry was measured again
to check for consistency. As can be seen below (and in the Figures
attached), the program was largely successful with agreement mostly to
better than 5% (0.05 mag). There are a few outliers, but we chose not
to increase the order of the Polynomial and to preserve the shape of
the spectra and their features from the spectrophotometric calibration.
The equations of the linear scaling functions applied to the spectra are
documented below. The flux calibrated spectra are labelled as

UVB_AT2017gfo_XSHOOTER_MJD-57983.969_Phase-1.43.dat 

or similar, and are located in the folder flux_corrected_unsmoothed_spectra.

UVB spectra:

+1.43d, +4.40d, +5.40d, +6.40d, +9.40d and +10.4d were all just shifted
by a constant to match the g-band photometry. By doing this, the spectra
fit the photometry, and agreed well with the overlapping part of the VIS
arm for the respective epochs. For +2.42d, +3.41d and +7.40d, the
spectra were scaled linearly to agree with both the g-band photometry and
the overlap with the VIS arm. +8.40d could not be reconciled with photometry,
or the overlap of the VIS arm, and so was discarded.

VIS spectra:

+1.43d - +9.40d were all corrected to the photometry using the J. Gillanders
python script. This scales the spectrum to the photometry, based on
how the (r-i) or (i-z) value differs for the photometry and for the
spectrum. Depending on the epoch, either (r-i) or (i-z) was used
(whichever produced the best fit). +10.40d didn't work using this method,
and so a simple linear scaling was created manually, and applied to
the spectrum. Listed below are the photometry values for each epoch,
and the corresponding synthetic values calculated for the flux
corrected spectra using SMS. The equations of the scaling functions
are also listed for each epoch.

The colour that was used to constrain the linear fit is listed before the
Corrected_Flux equation. For example, in Epoch 1 (r-i) was used. 

Phase:	Filter:	Photometric:	Synthetic:
+1.43d	r	17.99±0.01	17.97
	i	17.85±0.05	17.85
	z	17.72±0.03	17.72
	]
r-i:	 Corrected_Flux = (0.488 * (( 7.97e-05 * Wavelength) + 1)) * Flux)

Phase:	Filter:	Photometric:	Synthetic:
+2.42d	r	19.13±0.17	19.13
	i	18.58±0.04	18.58
	z	18.33±0.06	18.33

i-z:	 Corrected_Flux = ( 0.261 * (( 2.33e-4 * Wavelength) + 1)) * Flux)

Phase:	Filter:	Photometric:	Synthetic:
+3.41d	r	19.81±0.02	19.89
	i	19.03±0.01	19.03
	z	18.74±0.02	18.74

i-z:	 Corrected_Flux = ( 0.119 * (( 6.60e-4 * Wavelength) + 1)) *
Flux)

Phase:	Filter:	Photometric:	Synthetic:
+4.40d	r	20.53±0.05	20.52
	i	19.51±0.04	19.52
	z	19.07±0.06	19.12
	
r-i:	 Corrected_Flux = ( 0.015 * (( 6.39e-3 * Wavelength) + 1)) * Flux)

Phase:	Filter:	Photometric:	Synthetic:
+5.40d	r	20.79±0.24	20.43
	i	19.55±0.18	19.55
	z	19.17±0.11	19.17

i-z:	 Corrected_Flux = ( 0.861 * (( -3.91e-06 * Wavelength) + 1)) * Flux)

Phase:	Filter:	Photometric:	Synthetic:
+6.40d	r	20.95±0.35	20.95
	i*	20.05±0.18	20.05
	z*	19.53±0.11	19.74

r-i:	 Corrected_Flux = ( 1.263 * (( -3.48e-05 * Wavelength) + 1)) * Flux)

Phase:	Filter:	Photometric:	Synthetic:
+7.40d	r	21.23±0.11	21.23
	i	20.54±0.05	20.54
	z	19.89±0.05	20.21

r-i:	 Corrected_Flux = ( 1.254 * (( -3.86e-05 * Wavelength) + 1)) * Flux)

Phase:	Filter:	Photometric:	Synthetic:
+8.40d	r	21.95±0.18	21.18
	i	20.72±0.06	20.72
	z	20.40±0.06	20.40	

i-z:	 Corrected_Flux = ( 1.649 * (( -4.38e-05 * Wavelength) + 1)) *Flux)

Phase:	Filter:	Photometric:	Synthetic:
+9.40d	r	22.20±0.04	21.80
	i	21.37±0.06	21.37
	z	21.19±0.07	21.19

i-z:	 Corrected_Flux = ( 1.201 * (( -3.44e-05 * Wavelength) + 1)) * Flux)

Phase:	Filter:	Photometric:	Synthetic:
+10.40d	r	22.45±0.07	22.74
	i	22.38±0.10	22.25
	z	22.06±0.13	22.54

Linear correction:	Corrected_Flux = 2.96e-23 * Wavelength + 0.55e-17

* - interpolated photometry value. There was no photometry exactly at this epoch.
Therefore values were obtained from interpolation between measured photometry
before and after the spectral epoch. 

NIR spectra:

+1.43d - +3.41d and +7.40d were all corrected to the photometry using
J. Gillanders method.  +4.40d - +6.40d and +8.40d - +10.40d did not produce
satisfactory results using this method (the program struggled to fit
the K-band photometry). Instead, a simple linear scaling was
applied. Listed below are the photometry values for each epoch, and
the corresponding synthetic values calculated for the flux corrected
spectra using SMS. The equations of the scaling functions are also
listed for each epoch.

For the K-band we report some multiple measurements to check for
validity (from GROND and other instruments) in the tables below.

Phase:	Filter:	Observed:	Synthetic:
+1.43d	J	17.58±0.07	17.58
	H	17.64±0.08	17.64
	K	17.87±0.12	17.87

J-H:	 Corrected_Flux = ( 0.667 * (( 3.26e-05 * Wavelength) + 1)) * Flux)

Phase:	Filter:	Observed:	Synthetic: 
+2.42d	J	17.73±0.09	17.73
	H	17.64±0.08	17.64
	K	17.73±0.09	17.76

J-H:	 Corrected_Flux = ( 0.696 * (( 2.45e-05 * Wavelength) + 1)) * Flux)

Phase:	Filter:	Observed:	Synthetic:
+3.41d	J	17.95±0.07	17.95
	H	17.72±0.07	17.72
	K	17.69±0.08	17.60

J-H:	 Corrected_Flux = ( 0.573 * (( 2.63e-05 * Wavelength) + 1)) * Flux)

Phase:	Filter:	Observed:	Synthetic:
+4.40d	J	18.17±0.07	18.21
	H	18.02±0.10	17.87
	K	17.74±0.11	17.90
	GROND_K	17.67±0.08	17.90

Linear Correction:	Corrected Flux = -2.17e-21 * Wavelength + 6.40e-17

Phase:	Filter:	Observed:	Synthetic:
+5.40d	J	18.46±0.07	18.44
	H	18.26±0.12	18.18
	K	17.79±0.11	17.89

Linear Correction:	Corrected Flux = 2.11e-21 * Wavelength + 7.00e-17

Phase:	Filter:	Observed:	Synthetic:
+6.40d	J	18.74±0.04	18.72
	H	18.50±0.12	18.29
	K	17.84±0.03	17.93

Linear Correction:	Corrected Flux = -9.95e-22 * Wavelength + 3.20e-17

Phase:	Filter:	Observed:	Synthetic:
+7.40d	J	19.26±0.28	19.26
	H	18.74±0.06	18.74
	K	18.04±0.12	18.09
	GROND_K	18.27±0.39	18.09

J-H:	 Corrected_Flux = ( 0.514 * (( 4.22e-05 * Wavelength) + 1)) * Flux)

Phase:	Filter:	Observed:	Synthetic:
+8.40d	J	19.64±0.11	19.73
	H	19.26±0.26	19.17
	K	18.40±0.20	18.53
	GROND_K	18.43±0.15	18.53

Linear Correction:	Corrected Flux = -2.93e-22 * Wavelength + 1.75e-17

Phase:	Filter:	Observed:	Synthetic:
+9.40d	J	20.23±0.10	20.34
	H	19.66±0.14	19.61
	K	18.50±0.20	18.66

Linear Correction:	Corrected Flux = 7.02e-23 * Wavelength + 0.10e-17

Phase:	Filter:	Observed:	Synthetic:
+10.40d	J	21.02±0.22	20.98
	H	20.17±0.34	19.91
	K	18.75±0.25	18.76
	GROND_K	18.51±0.15	18.76

Linear Correction:	Corrected Flux= -7.41e-23 * Wavelength +0.85e-17

Smoothing and combining the X-Shooter spectra
==================================================

J. Gillanders merged the flux calibrated spectra together. There was
good agreement between the UVB and VIS arms for all epochs (except
+8.40d - mentioned above), and so they were joined at 5500
Angstroms. The join between VIS and NIR did not tend to overlap
perfectly, and so between 9942 and 10201 Angstroms, an average of the
spectra was taken. The full spectrum was then taken and a smoothing
function was applied (python inbuilt function -
scipy.signal.medfilt. A kernel size of 51 was used). Around the
UVB-VIS and VIS-NIR joins the spectrum still retained some noise
(spectra were noisiest at the ends of the arms), and so in the ranges
5500-5600 & 9940-10200 Angstroms, the spectrum was smoothed even
further.

The ends of the spectra were also removed as they were noisy and
didn't contain any useful information (<3300 & >24000 Angstroms). The
NIR ends of +1.43d and +4.40d were snipped at 22500 Angstroms as they
were messy beyond this point, and din't contain any useful
information. The UVB end of +8.40d was snipped at 6000 Angstroms as
the spectra could not be reconciled with the photometry (g-band
photometry for +8.40d is in disagreement with the spectrum). The flux
calibrated, smoothed and joined spectra are labelled as

AT2017gfo_XSHOOTER_MJD-57983.969_Phase-1.43.dat

or similar, and are located in the folder

flux_corrected_smoothed_spectra.

The DECAM y-band photometry fits well at all phases, except at
+1.43d. The photometry value is too bright, and so another y-band
point was interpolated, from PanSTARRS photometry, taken 0.7d before
and 0.3d after the spectrum. This interpolated photometric point is
plotted in the pdf provided as a dark grey point. It agrees well with
the spectrum and so further corrections to bring the spectrum in line
with the DECAM photometric point were not considered. There may be a
calibration issue with that DECam Y-band point.

Reddening and redshift corrected spectra 
================================================

All the above corrections were applied to the observed spectra before
correction for Milky Way extinction and correction to restframe.
The smoothed, reddening corrected, and redshift corrected spectra are
labelled as

AT2017gfo_XSHOOTER_MJD-57983.969_Phase-1.43_deredz.dat

or similar, and are located in the folder 

dereddened+deredshifted_spectra. 

Also located in this folder are 3 pdfs which illustrate the series of
spectra, with the dereddened photometry overlaid for comparison. The
telluric regions are highlighted grey, along with the regions where
the spectra were stitched together, to highlight that the features at
these wavelengths (if there are any) are likely not physical. Some HST
spectra are also plotted and are labelled as such. They are plotted
for comparison with the X-Shooter spectra.


HST spectra from Tanvir et al. 2017
=====================================

Some HST spectra were obtained from Tanvir et al. 2017, ApJ, 848, L27 are
labelled as AT2017gfo_HST_G102_G141_Phase-4.9.dat or similar, and are
located in the folder HST. They have been shifted by a constant to
match photometry and/or the X-Shooter spectra. They don't agree
perfectly with the X-Shooter spectra, or the photometry, but they
offer the ability to see the shape of the spectrum in one of the
telluric regions (13100 - 14360 Angstroms). They have been dereddened
and deredshifted for the purposes of comparing with the X-Shooter
spectra in these plots, but the HST data files have not been corrected
for reddening or redshift effects.



Photometry sources reference 
=====================================

In order to correct the spectra to the most reliable photometry,
S. Smartt compiled his own selected set of photometry from all the
published work, as described in Coughlin et al. 2018, MNRAS 480, 3871.
To quote and paraphrase from that paper : 

We began with the photometry from the UV to K band from (Andreoni et
al. 2017; Arcavi et al. 2017; Chornock et al. 2017; Cowperthwaite et
al. 2017; Drout et al. 2017; Evans et al. 2017; Kasliwal et al. 2017;
Tanvir et al. 2017; Pian et al. 2017a; Troja et al. 2017; Smartt et
al. 2017; Utsumi et al. 2017; Valenti et al. 2017) from phases +0.467
to +25.19 d after GW170817 and at each epoch created the broadest
spectral energy distribution possible.

We began with the photometry of Smartt et al. (2017) as the core data
set. This employed difference imaging at all epochs of PESSTO (Public
ESO Spectroscopic Survey of Transient Objects; Smartt et al. 2015),
GROND, and Pan-STARRS imaging. Our approach was to (i) complement this
photometry only when this was necessary - either due to insufficient
temporal or wavelength coverage, (ii) primarily use only grizyJHKS AB
mag photometry from sources that used image subtraction (mostly DECam
and Skymapper Cow - perthwaite et al. 2017; Andreoni et al. 2017), or
from HST where host contamination is not important (Tanvir et
al. 2017), and (iii) when this was not possible, focus on a small
number of independent sources such as Gemini South (Kasliwal et
al. 2017), VISTA (Tanvir et al. 2017), and Sirius (Utsumi et
al. 2017). We verified consistency between the data sets through
direct comparison. In this way, we compiled grizyJHK SEDs, or as
broad a subset as the data allowed.

The photometry is available in the file : AT2017gfo_phot_compiled_sjs.dat
