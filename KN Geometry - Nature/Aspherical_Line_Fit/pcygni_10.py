##P Cygni 9 == Inclination attempt 2 with variable inclination
##P Cygni 8 == Fixed Inclination with variable ellipticity
##P Cygni 7 == Elliposidal along l.o.s (preliminary and wrong) but we can develop. 
##P Cygni 5 == Spherical

#!/usr/bin/env python
# MIT License
#
# Copyright (c) 2017 Ulrich Noebauer
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EveNT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
References
----------
 * Jeffery & Branch 1990: "Analysis of Supernova Spectra"
   ADS link:http://adsabs.harvard.edu/abs/1990sjws.conf..149J
 * Thomas et al 2011: "SYNAPPS: Data-Driven Analysis for Supernova
   Spectroscopy"
   ADS link:http://adsabs.harvard.edu/abs/2011PASP..123..237T
"""
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integ
import scipy
import astropy.units as units
import astropy.constants as csts
from numba.experimental import jitclass
import numba 
from multiprocessing import Pool
import multiprocessing

spec = [
    ('rmax', numba.float32), 
    ('rmin', numba.float32), 
    ('Ip', numba.float32), 
    ('t', numba.float32), 
    ('vdet_min', numba.float32), 
    ('vdet_max', numba.float32), 
    ('tauref', numba.float32), 
    ('vref', numba.float32), 
    ('ve', numba.float32), 
]

def proxy(f):
    return lambda x: f(x)


import numpy as np
import numba

@numba.njit
def _calc_p(r, z, a, b):
    """
    Calculate p-coordinate of location (r,z) in ellipsoidal ejecta.

    Parameters
    ----------
    r : float
        Ellipsoidal radial coordinate of location.
    z : float
        z-coordinate (along the line-of-sight).

    Returns
    -------
    p : float
        p-coordinate (perpendicular to z).
    """
    assert np.abs(r) > np.abs(z)
    return np.sqrt(r**2 - z**2)


@numba.njit
def _calc_r_g(x, y, z, a, b):
    """
    Calculate the ellipsoidal radius of a location (x, y, z).

    Parameters
    ----------
    x, y : float
        Coordinates perpendicular to the line-of-sight.
    z : float
        Coordinate along the line-of-sight.
    a : float
        Semi-major axis of the ellipsoid.
    b : float
        Semi-minor axis of the ellipsoid.

    Returns
    -------
    r : float
        Ellipsoidal radius of the location.
    """
    g = np.sqrt((x/a)**2 + (y/b)**2 + (z/a)**2)  # Ellipsoidal scaling factor
    return g * b  # Convert back to physical distance

@numba.njit
def _calc_r( x, y, z, a, b):
    """
    Calculate radius of location (z, p) in ejecta;

    Parameters
    ----------
    p : float
        p-coordinate (perpendicular to line-of-sight to observer)
    z : float
        z-coordinate (along line-of-sight to observer)

    Returns
    -------
    r : float
        radius of location
    """
    return np.sqrt(x**2 + y**2 + z**2)

@numba.njit
def _calc_W_old(r, vphot, t):
    """
    Calculate geometric dilution factor.

    Parameters
    ----------
    r : float
        Ellipsoidal radius of location.

    Returns
    -------
    W : float
        Geometric dilution factor.
    """
    #return (1. - np.sqrt(1. - (vphot * t / r)**2)) / 2

    corr = min(1,vphot*t / r )
    
    return (np.float32(1) - np.sqrt(np.float32(1) - (corr)**2)) / 2


@numba.njit
def _calc_W(r, vphot, t, g):
    """
    Calculate geometric dilution factor for an ellipsoidal ejecta.

    Parameters
    ----------
    x, y, z : float
        Spatial coordinates.
    vphot : float
        Reference photospheric velocity.
    t : float
        Time since explosion.
    a : float
        Semi-major axis of ellipsoid (longer axis).
    b : float
        Semi-minor axis of ellipsoid (shorter axis).

    Returns
    -------
    W : float
        Geometric dilution factor.
    """
    # Compute the elliptical radius factor g(x, y, z)
    #g_phot = np.sqrt((1/a)**2 + (1/b)**2 + (1/a)**2)  # Reference g at photosphere
    
    # Adjusted vphot for the ellipsoidal shape
    #vphot_local = vphot * (g_phot / g)

    # Compute W using the modified vphot
    return (1. - np.sqrt(1. - (1/g)**2)) / 2


@numba.njit
def _calc_tau_old(x, y, z, r, vmax, vphot, t, vdet_min, vdet_max, tauref, ve, a, b):
    """
    Calculate the line optical depth at radius r in an ellipsoidal ejecta.

    Arguments
    ---------
    r : float
        Ellipsoidal radius of location.

    Returns
    -------
    tau : float
        Line optical depth.
    """
    v = r / t
    #v = np.sqrt( (x/a)**2 + (y/b)**2 + (z/a)**2 )
    #v_ph = np.sqrt( (x/a * vmax/vphot)**2 + (y/b * vmax/vphot)**2 + (z/a * vmax/vphot)**2 )

    if vdet_min <= v <= vdet_max:
        #g = r / a  # Normalized ellipsoidal scaling factor
        return tauref * np.exp( - (v-vphot) / ve)  # Scaling tau by (b/a)
        #return tauref * np.exp( - (v-v_ph) / ve )  # Scaling tau by (b/a)
    else:
        return 1e-20

@numba.njit
def _calc_tau_new(x, y, z, vmax, vphot, t, vdet_min, vdet_max, tauref, ve, a, b):
    """
    Calculate the line optical depth at location (x, y, z) in an ellipsoidal ejecta.

    Arguments
    ---------
    x, y, z : float
        Coordinates of the location in the ellipsoid.
    vmax : float
        Maximum velocity (for scaling).
    vphot : float
        Photospheric velocity.
    t : float
        Time parameter.
    vdet_min, vdet_max : float
        Minimum and maximum detection velocities.
    tauref : float
        Reference optical depth.
    ve : float
        Velocity scale for the exponential decay.
    a, b : float
        Ellipsoidal axes dimensions (major and minor).

    Returns
    -------
    tau : float
        Line optical depth at the location.
    """
    # Calculate the normalized ellipsoidal distance factor
    g = np.sqrt((x / a)**2 + (y / b)**2 + (z / a)**2 )  # Scaled distance in ellipsoidal geometry
    #print(g)
    # Calculate the velocity at the location
    v = g / t  # Velocity scaling by the ellipsoidal distance factor

    # Apply the optical depth formula, adjusting for the ellipsoidal scaling
    if vdet_min <= v <= vdet_max:
        # Rescale tau according to the ellipsoidal distance
        return tauref * np.exp(-(v - vphot) / ve)  # Apply exponential decay with rescaling
    else:
        return 1e-20  # If the velocity is outside the detection range, return effectively zero

@numba.njit
def _calc_tau(x, y, z, r, vmax, vphot, t, vdet_min, vdet_max, tauref, ve, a, b):
    """
    Calculate the line optical depth at location (x, y, z) in an ellipsoidal ejecta using a power-law
    relationship for optical depth as a function of the generalized distance, normalized at the photosphere.

    Arguments
    ---------
    x, y, z : float
        Coordinates of the location in the ellipsoid.
    vmax : float
        Maximum velocity (for scaling).
    vphot : float
        Photospheric velocity.
    t : float
        Time parameter.
    vdet_min, vdet_max : float
        Minimum and maximum detection velocities.
    tauref : float
        Reference optical depth at the photosphere.
    ve : float
        Velocity scale for the exponential decay.
    a, b : float
        Ellipsoidal axes dimensions (major and minor).
    p : float
        Power-law index for optical depth scaling.

    Returns
    -------
    tau : float
        Line optical depth.
    """
    # Calculate the generalized radius g for the location (x, y, z)
    g = np.sqrt( (x / (a * t))**2 + (y / (b * t) )**2 + (z / (a * t) )**2 )
    
    # Calculate the generalized radius at the photosphere, where the velocity is vphot
    #g_phot = np.sqrt( (vphot*t / a)**2 + (y / b * vphot/vmax)**2 + (z / a * vphot/vmax)**2 )

    # Apply the power-law scaling for optical depth, normalized at the photosphere
    v = r / t

    if vdet_min <= v <= vdet_max:
        # Power-law dependence on g, normalized at the photosphere
        #tau = tauref * np.exp( -(g-g_phot)/(ve*t) )  # Scaling by g_phot
        tau = tauref * (1 / g)**4.5  # Scaling by g_phot
        return tau
    else:
        return 1e-20  # If the velocity is outside the detection range, return effectively zero






@numba.njit
def _S(x, y, z, vmax, vphot, t, vdet_min, vdet_max, tauref, ve, a, b):
    """
    Calculate source function at location (x, y, z) in ejecta.

    Parameters
    ----------
    x, y : float
        Coordinates perpendicular to line-of-sight.
    z : float
        Coordinate along line-of-sight.

    Returns
    -------
    S : float
        Source function at location (x, y, z).
    """
    r = _calc_r(x, y, z, a, b)
    g = np.sqrt( (x/(a*t))**2 + (y/(b*t))**2 + (z/(a*t))**2 )
    
    if r > vmax * t or g < 1:
        return 0  # Outside ejecta or inside photosphere
    elif z < 0 and np.sqrt( (x/(a*t))**2 + (y/(b*t))**2 ) < 1:
        return 0  # Occulted region
    else:
        W = _calc_W(r, vphot, t, g)
        return W * 1  # Source function
        

@numba.njit
def _I(x, y, z, vphot, t, a, b):
    """
    Determine the initial specific intensity for a ray passing through (x, y, z).

    Parameters
    ----------
    x, y : float
        Coordinates of location.
    z : float
        Coordinate along line-of-sight.

    Returns
    -------
    I : float
        Initial specific intensity.
    """
    if np.sqrt( (x/(a*t))**2 + (y/(b*t))**2 ) < 1: # np.sqrt(x**2 + y**2) < vphot * t:
        return 1  # Inside photosphere
    else:
        return 0  # Above the photosphere

@numba.njit
def _tau(x, y, z, vmax, vphot, t, vdet_min, vdet_max, tauref, ve, a, b):
    """
    Determine the line optical depth along the line-of-sight towards the observer.

    Parameters
    ----------
    x, y : float
        Coordinates of location.
    z : float
        Coordinate along line-of-sight.

    Returns
    -------
    tau : float
        Optical depth at the location.
    """
    r = _calc_r(x, y, z, a, b)

    if r > vmax * t or np.sqrt( (x/(a*t))**2 + (y/(b*t))**2 + (z/(a*t))**2 ) < 1:
        return 0  # Outside ejecta or inside photosphere
    elif z < 0 and np.sqrt( (x/(a*t))**2 + (y/(b*t))**2) < 1:
        return 0  # Occulted region
    else:
        #return _calc_tau(r, vmax, vphot, t, vdet_min, vdet_max, tauref, ve, a, b)
        return _calc_tau(x, y, z, r, vmax, vphot, t, vdet_min, vdet_max, tauref, ve, a, b)

@numba.njit(fastmath=True, nogil=True)
def _Iemit(x, y, z, vmax, vphot, t, vdet_min, vdet_max, tauref, ve, a, b):
    """
    Determine the total specific intensity reaching the observer from (x, y, z).

    Parameters
    ----------
    x, y : float
        Coordinates of location.
    z : float
        Coordinate along line-of-sight.

    Returns
    -------
    Iemit : float
        Total specific intensity emitted towards the observer.
    """
    tau = _tau(x, y, z, vmax, vphot, t, vdet_min, vdet_max, tauref, ve, a, b)

    return (_I(x, y, z, vphot, t, a, b) * np.exp(-tau) + 
            _S(x, y, z, vmax, vphot, t, vdet_min, vdet_max, tauref, ve, a, b) * 
            (1. - np.exp(-tau))) #* np.sqrt( x**2 + y**2)

@numba.cfunc(numba.f8(numba.i4, numba.types.CPointer(numba.f8)))
def get_Iemit(n, ptr):
    temp = numba.carray(ptr, n)

    vmax     = np.float32(temp[3])
    vphot    = np.float32(temp[4])
    t        = np.float32(temp[5])
    vdet_min = np.float32(temp[6])
    vdet_max = np.float32(temp[7])
    tauref   = np.float32(temp[8])
    ve = np.float32(temp[9])
    a = np.float32(temp[10])
    b = np.float32(temp[11])

    return _Iemit(np.float32(temp[0]), np.float32(temp[1]), np.float32(temp[2]),
                  vmax, vphot, t, vdet_min, vdet_max, tauref, ve, a, b)

    
    
    
    
class PcygniCalculator(object):
    """
    Calculator for P-Cygni line profiles emerging from homologously expanding
    supernova ejecta flows using the Elementary Supernova model and
    prescriptions outlined in Jeffery & Branch 1990.

    This calculator heavily relies on the impact geometry (see Fig. 1 in
    Jeffery & Branch 1990) in which the z-axis points towards the observer and
    the impact parameters p is defined perpendicular to that. The connection to
    spherical symmetry is achieved by mu=z/r, r=sqrt(z**2 + p**2).

    The only routines which should be used by the user are the ones without an
    underscore, i.e.:
    * calc_profile_Fnu
    * calc_profile_Flam
    * show_line_profile
    """
    def __init__(self, t=3000 * units.s, vmax=0.01 * csts.c,
                 vphot=0.001 * csts.c, tauref=1, vref=5e7 * units.cm/units.s,
                 ve=5e7 * units.cm/units.s, lam0=1215.7 * units.AA, ratio_vel = 1, theta_inc=21, 
                 vdet_min=None, vdet_max=None):
        """
        Parameters
        ----------
        t : scalar astropy.units.Quantity
            time since explosion; together with the photospheric and maximum
            velocity this sets the length scale of the ejecta (default 3000 s)
        vmax : scalar astropy.units.Quantity
            maximum ejecta velocity; with the time since explosion, this sets
            the outer radius of the ejecta (default 1 per cent speed of light)
        vphot : scalar astropy.units.Quantity
            photospheric velocity; with the time since explosion, this sets the
            radius of the photosphere, i.e. of the inner boundary (default 0.1
            per cent of speed of light)
        tauref : float
            line optical depth at a reference velocity (vref) in the ejecta;
            this sets the strength of the line transition (default 1)
        vref : scalar astropy.units.Quantity
            reference velocity; needed in the assumed density stratification
            and sets the ejecta location where the reference line optical depth
            is measured (default 5e7 cm/s)
        ve : scalar astropy.units.Quantity
            second parameter used in the assumed density stratification
            (defautl 5e7 cm/s)
        lam0 : scalar astropy.units.Quantity
            rest frame wavelength of the line transition (default 1215.7 A)
        vdet_min : None or scalar astropy.units.Quantity
            lower/inner location of the line formation region; enables
            detachment of line formation region; if None, will be set to vphot
            (default None)
        vdet_max : None or scalar astropy.units.Quantity
            upper/outer location of the line formation region; enables
            detachment of line formation region; if None, will be set to vmax
            (default None)
        ratio_vel : the velocity of l.o.s. relative to perpendicular (ie the eccentricity of the ellipse)    
        """

        # ensure that the calculator works with the correct units
        self._t = t.to("s").value
        self._vmax = vmax.to("cm/s").value
        self._vphot = vphot.to("cm/s").value
        self._ve = ve.to("cm/s").value
        self._vref = vref.to("cm/s").value

        # spatial extent of the ejecta
        self._rmax = self._t * self._vmax
        self._rmin = self._t * self._vphot
        self._zmax = self._rmax

        # CMF natural wavelength and frequency of the line
        self._lam0 = lam0.to("cm").value
        self._nu0 = csts.c.cgs.value / self._lam0

        # determine the maximum width of the profile
        dlambda = self._lam0 / self._t * self._zmax / csts.c.cgs.value

        # determine the wavelength/frequency range over which the profile will
        # be calculated (5% more than maximum Doppler shift on both ends)
        self._lam_min = self._lam0 - 1.05 * dlambda
        self._lam_max = self._lam0 + 1.05 * dlambda
        self._nu_min = csts.c.cgs.value / self._lam_max
        self._nu_max = csts.c.cgs.value / self._lam_min

        self._tauref = tauref
        self._Ip = 1
        
        self._ratio_vel = ratio_vel

        if vdet_min is None:
            vdet_min = self.vphot
        else:
            vdet_min = vdet_min.to("cm/s").value
        if vdet_max is None:
            vdet_max = self.vmax
        else:
            vdet_max = vdet_max.to("cm/s").value

        self._vdet_min = vdet_min
        self._vdet_max = vdet_max

        t0 = t.to("s").value
        ve0 = ve.to("cm/s").value
        vmax0 = vmax.to("cm/s").value
        vphot0 = vphot.to("cm/s").value
        #vdm0 = vmax.to("cm/s").value
        self.args = (vmax0, vphot0, t0, vdet_min, vdet_max, tauref, ve0, vphot0/ratio_vel, vphot0)
        self._Iemit = scipy.LowLevelCallable(get_Iemit.ctypes)
    # Using properties allows the parameters to be crudely "hidden" from the
    # user; thus he is less likely to change them after initialization
    @property
    def t(self):
        """time since explosion in s"""
        return self._t

    @property
    def vmax(self):
        """maximum ejecta velocity in cm/s"""
        return self._vmax

    @property
    def vphot(self):
        """photospheric velocity in cm/s"""
        return self._vphot

    @property
    def ve(self):
        """velocity scale in density profile in cm/s"""
        return self._ve

    @property
    def vref(self):
        """reference velocity in cm/s"""
        return self._vref

    @property
    def vdet_min(self):
        """inner location of line-formation region in cm/s"""
        return self._vdet_min

    @property
    def vdet_max(self):
        """outer location of line-formation region in cm/s"""
        return self._vdet_max

    @property
    def rmax(self):
        """outer ejecta radius in cm"""
        return self._rmax

    @property
    def rmin(self):
        """photospheric radius in cm"""
        return self._rmin

    @property
    def zmax(self):
        """maximum z-coordinate in ejecta in cm (corresponds to rmax)"""
        return self._zmax

    @property
    def lam0(self):
        """CMF natural wavelength of line transition in cm"""
        return self._lam0

    @property
    def nu0(self):
        """CMF natural frequency of line transition in Hz"""
        return self._nu0

    @property
    def lam_min(self):
        """minimum wavelength for line profile calculation in cm"""
        return self._lam_min

    @property
    def lam_max(self):
        """maximum wavelength for line profile calculation in cm"""
        return self._lam_max

    @property
    def nu_min(self):
        """minimum frequency for line profile calculation in Hz"""
        return self._nu_min

    @property
    def nu_max(self):
        """maximum frequency for line profile calculation in Hz"""
        return self._nu_max

    @property
    def Ip(self):
        """photospheric continuum specific intensity in arbitrary units"""
        return self._Ip

    @property
    def tauref(self):
        """reference line optical depth"""
        return self._tauref

    @property
    def ratio_vel(self):
        """reference line optical depth"""
        return self._ratio_vel
    
    def _calc_z(self, nu):
        """
        Calculate location (in terms of z) of resonance plane for photon
        emitted by the photosphere with frequency nu

        Parameters
        ----------
        nu : float
            photospheric photon frequency

        Returns
        -------
        z : float
            z coordinate of resonance plane
        """

        return csts.c.cgs.value * self.t * (1. - self.nu0 / nu)


    def _calc_line_flux(self, nu, mode="both"):
        """
        Calculate the emergent flux at LF frequency nu

        Parameters
        ----------
        nu : float
            lab frame frequency at which the line flux is to be calculated
        mode : str
            identifies the included interaction channels; see self.Iemit
            (default 'both')

        Returns
        -------
        Fnu : float
            emergent flux F_nu
        """

        z = self._calc_z(nu)
        pmax = self.rmax

        #print( np.max())
        # integration over impact parameter p
        #Fnu = 2. * np.pi * integ.quad(self._Iemit, 0, pmax, args=(z, *self.args), epsabs=1)[0]
        
        #if self.ratio_vel>1: 
        #    pmax = self.rmax*self.ratio_vel
        
        #double integral
        #Fnu  = 4*integ.dblquad(self._Iemit, 0, pmax, lambda x: 0, lambda x: (pmax**2-x**2)**(1/2), args=(z, *self.args), epsabs=1)[0]
        #Fnu  = 2*integ.dblquad(self._Iemit, -pmax, pmax, lambda x: 0, lambda x: pmax, args=(z, *self.args), epsabs=1e25)[0] 
        
        a = pmax #/ self.ratio_vel
        b = pmax
        
        Fnu = integ.dblquad(self._Iemit, 
                           -a, a,  # y-limits
                           lambda y: -b * (1 - (y/a)**2)**0.5,  # x-min
                           lambda y:  b * (1 - (y/a)**2)**0.5,  # x-max
                           args=(z, *self.args), epsabs=1e-1)[0]
        
        
        #1e25 , epsabs=1e25
        return Fnu

    def _calc_line_profile_base(self, nu_min, nu_max, npoints=100,
                                mode="both"):
        """
        Calculate the full line profile between the limits nu_min and nu_max in
        terms of F_nu.

        Parameters
        ----------
        nu_min : float
            lower frequency limit
        nu_max : float
            upper frequency limit
        npoints : int
            number of points of the equidistant frequency grid (default 100)
        mode : str
            identifier setting the interaction mode, see self.Iemit
            (default 'both')

        Returns
        -------
        nu : np.ndarray
            frequency grid
        Fnu : np.ndarray
            emitted flux F_nu
        """

        nu = np.linspace(nu_min, nu_max, npoints)

        Fnu = []
        for nui in nu:
            Fnu.append(self._calc_line_flux(nui, mode=mode))

        #with Pool(4) as p:     
        #    Fnu = p.map(self._calc_line_flux, nu)
        
        return nu * units.Hz, np.array(Fnu)

    def calc_profile_Fnu(self, npoints=100, mode="both"):
        """Calculate normalized line profile in terms of F_nu

        Parameters
        ----------
        npoints : int
            number of points of the equidistant frequency grid (default 100)
        mode : str
            identifier setting the interaction mode, see self.Iemit
            (default 'both')

        Returns
        -------
        nu : np.ndarray
            frequency grid
        Fnu_normed : np.ndarray
            emitted flux F_nu, normalized to the emitted photospheric continuum
            flux
        """

        nu, Fnu = self._calc_line_profile_base(self.nu_min, self.nu_max,
                                               npoints=npoints, mode=mode)

        Fnu_normed = Fnu / Fnu[0]
        return nu, Fnu_normed

    def calc_profile_Flam(self, npoints=100, mode="both"):
        """Calculate normalized line profile in terms of F_lambda

        Parameters
        ----------
        npoints : int
            number of points in the wavelength grid. NOTE even though a
            F_lam(lam) is calculated the underlying wavelength grid is chosen
            such that it is equidistant in nu-space (since the actual
            integration happens in terms of F_nu(nu))
            (default 100)
        mode : str
            identifier setting the interaction mode, see self.Iemit
            (default 'both')

        Returns
        -------
        lam : np.ndarray
            wavelength grid
        Flambda_normed : np.ndarray
            emitted flux F_lambda, normalized to the emitted photospheric
            continuum flux
        """

        nu, Fnu = self._calc_line_profile_base(self.nu_min, self.nu_max,
                                               npoints=npoints, mode=mode)
        lam = nu.to("AA", equivalencies=units.spectral())[::-1]
        cont = (Fnu[0] * np.ones(len(Fnu)) * nu.to("Hz").value**2 /
                csts.c.cgs.value)
        F_lambda_normed = (Fnu * nu.to("Hz").value**2 /
                           csts.c.cgs.value / cont)[::-1]

        return lam, F_lambda_normed

    def show_line_profile(self, npoints=100, include_abs=True,
                          include_emit=True, vs_nu=False):
        """
        Visualise Line Profile

        The P-Cygni line profile will always be displayed. The pure absorption
        and emission components can be included in the plot as well. The flux
        (will always be be F_nu) may be plotted against frequency or
        wavelength.

        Arguments:
        nu_min  -- lower frequency limit
        nu_max  -- upper frequency limit

        Keyword arguments:
        npoints -- number of points of the frequency grid (default 100)
        include_abs  -- if True, the pure absorption flux will be included and
                        shown as a separate line (default True)
        include_emit -- if True, the pure emission flux will be included and
                        shown as a separate line (default True)
        vs_nu -- if True the quantities will be shown against frequency,
                 otherwise against wavelength (default True)

        Returns:
        fig -- figure instance containing plot
        """

        if vs_nu:
            x, y = self.calc_profile_Fnu(npoints=npoints, mode="both")
            x = x.to("Hz")
            if include_abs:
                yabs = self.calc_profile_Fnu(npoints=npoints, mode="abs")[-1]
            if include_emit:
                yemit = self.calc_profile_Fnu(npoints=npoints, mode="emit")[-1]
        else:
            x, y = self.calc_profile_Flam(npoints=npoints, mode="both")
            x = x.to("AA")
            if include_abs:
                yabs = self.calc_profile_Flam(npoints=npoints, mode="abs")[-1]
            if include_emit:
                yemit = self.calc_profile_Flam(
                    npoints=npoints, mode="emit")[-1]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.subplots_adjust(top=0.8)

        if include_abs:
            ax.plot(x, yabs, color="grey", ls="dashed",
                    label="absorption component")
        if include_emit:
            ax.plot(x, yemit, color="grey", ls="dotted",
                    label="emission component")

        ax.plot(x, y, ls="solid",
                label="emergent line profile")
        ax.legend(bbox_to_anchor=(0., 1.05, 1., .102), loc=3, ncol=2,
                  mode="expand", borderaxespad=0.)

        if vs_nu:
            ax.set_xlabel(r"$\nu$ [Hz]")
            ax.set_ylabel(r"$F_{\nu}/F_{\nu}^{\mathrm{phot}}$")
        else:
            ax.set_xlabel(r"$\lambda$ [$\AA$]")
            ax.set_ylabel(r"$F_{\lambda}/F_{\lambda}^{\mathrm{phot}}$")

        ax.set_xlim([np.min(x.value), np.max(x.value)])

        return fig


def example():
    """a simple example illustrating the use of the line profile calculator"""

    prof_calc = PcygniCalculator(t=3000 * units.s, vmax=0.01 * csts.c,
                                 vphot=0.001 * csts.c, tauref=1, vref=5e7 *
                                 units.cm/units.s, ve=5e7 * units.cm/units.s,
                                 lam0=1215.7 * units.AA)

    prof_calc.show_line_profile(npoints=100)
    plt.show()

if __name__ == "__main__":

    example()
