import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
import numba 
import fast_integrator
import multiprocessing

import pickle
from pcygni_5 import PcygniCalculator
import astropy.units as units
import astropy.constants as csts
from scipy.interpolate import interp1d
import lmfit
import time

path = './spectra_filtered' #\AT2017gfo\AT2017gfo\dereddened+deredshifted_spectra' 
files = os.listdir(path)
x = np.loadtxt(path+'/'+files[0]).T #path+'/'+files[0]).T
exc_reg_2 = (~((x[:,0] > 13100) & (x[:,0] < 14400))) & (~((x[:,0] > 17550) & (x[:,0] < 19200))) & \
          (~((x[:,0] > 5330) & (x[:,0] < 5740))) & (~((x[:,0] > 9840) & (x[:,0] < 10300))) & \
          (x[:,0] > 3600) & (x[:,0] < 22500)
wl, flux, error = x[:,0][exc_reg_2], x[:,1][exc_reg_2], x[:,2][exc_reg_2]

c_cgs = csts.c.cgs.value
AA_in_cgs = units.AA.cgs.scale
#pi = np.pi; h = 6.626e-27; c = 3.0e+10; k = 1.38e-16 #cgs
pi = np.pi; h = 6.626e-34; c = 3.0e+8; k = 1.38e-23   #SI

# works in cgs
def calc_profile_Flam(npoints, t, vmax, vphot, tauref, vref, ve, lam0):
    zmax = t * vmax
    dlambda = lam0 / t * zmax / c_cgs
    lam_min = lam0 - 1.05 * dlambda
    lam_max = lam0 + 1.05 * dlambda
    nu_min = c_cgs / lam_max

    nu_max = c_cgs / lam_min
    vdet_min = vphot
    vdet_max = vmax
    nu, Fnu = fast_integrator.calc_line_profile_base(npoints, nu_min, nu_max, vmax, vphot, t, vdet_min, vdet_max, tauref, ve, lam0)
    
    lam = c_cgs/nu[::-1] 
    cont = (Fnu[0] * np.ones(len(Fnu)) * nu**2 / c_cgs)
    F_lambda_normed = (Fnu * nu**2 / c_cgs / cont)[::-1]

    return lam, F_lambda_normed


#new = False
p_cygni_line_corr = lambda *x: p_cygni_line_corr_new(*x) if new else p_cygni_line_corr_old(*x)
def p_cygni_line_corr_new(wl, v_out, v_phot, tau, lam, vref, ve, t0): 
    cyg = calc_profile_Flam(npoints=50, t=t0, vmax=v_out * c_cgs,
                                 vphot=v_phot * c_cgs, tauref=tau, vref=vref *
                                 c_cgs, ve=ve * c_cgs,
                                 lam0=lam * AA_in_cgs)#, _lam_min=7000, _lam_max=13000)
    return np.interp(wl, cyg[0]*1e-2, cyg[1])    

def p_cygni_line_corr_old(wl, v_out, v_phot, tau, lam, vref, ve, t0): 
    calc = PcygniCalculator(t=t0 * units.s, vmax=v_out * csts.c,
                                 vphot=v_phot * csts.c, tauref=tau, vref=vref *
                                 csts.c, ve=ve * csts.c,
                                 lam0=lam * units.AA)#, _lam_min=7000, _lam_max=13000)
    cyg = calc.calc_profile_Flam(npoints=50)
    inter = interp1d(cyg[0]*1e-10, cyg[1], bounds_error=False, fill_value=1)    
    return inter(wl)



@numba.njit(fastmath=True)
def temp(wav,T):
    a = 2.0*h*c**2
    b = h*c/(wav*k*T)
    return a/ ( (wav**5)*(np.exp(b) - 1.0) )
    
@numba.njit(fastmath=True)
def gaussian(wav, amp, mu, sigma): 
    return amp*np.exp(-(wav-mu)**2/(2*sigma**2))
    
def planck_with_mod(wav, T, N, v_out, v_phot, tau=0.55, occul = 1, ve=0.32,
                    amp1 = 2, cen1=15500, sig1=250,amp2 = 2, cen2=15500, sig2=250, 
                    vref=0., t0=120960, lam=10506.3653): #blackbody
    #a = 2.0*h*pi*c**2
    #b = h*c/(wav*k*T)
    #intensity = a/ ( (wav**5)*(np.exp(b) - 1.0) )
    intensity = temp(wav, T)
    #intensity = 1
    
    pcyg_prof3 = p_cygni_line_corr(wav, v_out, v_phot, 1/13.8*tau   , 10036.65,  vref, ve , t0)
    pcyg_prof4 = p_cygni_line_corr(wav, v_out, v_phot, 8.1/13.8*tau , 10327.311, vref, ve , t0)
    pcyg_prof5 = p_cygni_line_corr(wav, v_out, v_phot, 4.7/13.8*tau , 10914.887, vref, ve , t0)
    correction = pcyg_prof3*pcyg_prof4*pcyg_prof5
    correction[correction>1] = (correction[correction>1]-1)*occul + 1
    
    #correction = p_cygni_line_corr(wav, v_out, v_phot, tau, lam, vref=vref, ve=ve, t0=t0)
    #correction[correction>1] = (correction[correction>1]-1)*occul + 1
    
    # Gaussians
    gau1 = gaussian(wav, 1e-17*amp1, cen1, sig1)
    gau2 = gaussian(wav, 1e-17*amp2, cen2, sig2)
    #gau1 = models.Gaussian1D.evaluate(wav, 1e-17*amp1, cen1, sig1)
    #gau2 = models.Gaussian1D.evaluate(wav, 1e-17*amp2, cen2, sig2)
    
    return N*intensity*correction+gau1+gau2


def residual(pars, wav, data=None, error=None): 
    v = pars.valuesdict()
    T, N, vphot = v["T"], v["N"], v["vphot"]
    t0, vmax, tau, vref, ve, occult2 = v["t0"], v["vmax"], v["tau"], v["vref"], v["ve"], v["occult"]
    amp1, amp2, cen1, cen2, sig1, sig2= v["amp1"], v["amp2"], v["cen1"], v["cen2"], v["sig1"], v["sig2"]

    model = planck_with_mod(wav, T, N, vmax, vphot, tau=tau, occul=occult2, ve = ve, vref = vref, 
                           amp1 = amp1, cen1=cen1, sig1=sig1, amp2 = amp2, cen2=cen2, sig2=sig2)
    
    if data is None:
        return model
    return (model - data)/error

@numba.njit(fastmath=True)
def lnprob_inner(model, flux, error):
    return -0.5 * np.sum(((model - flux) / error)**2 + np.log(2 * np.pi * error**2))

def lnprob(pars):
    #print("inner", multiprocessing.current_process().name, pars["T"].value)

    """This is the log-likelihood probability for the sampling."""  
    model = residual(pars, wl*1e-10)
    return lnprob_inner(model, flux, error)

def construct_p(vals): 
    p = p0_glob.copy()
    idx = 0;
    for x in p:
        if p[x].vary:
            p[x].value = vals[idx]
            idx += 1
    return p

p0_glob = None
def _lnprob(vals):
    p = construct_p(vals)
    prob = lnprob(p)
    return 1e100 if np.isnan(prob) else prob
    
def optimize(p0, use_new, nwalkers, steps):
    global new, p0_glob
    p0_glob = p0
    new = use_new
    import emcee
    mini = lmfit.Minimizer(lnprob, p0) 
    
    with multiprocessing.Pool(2) as pool:
        var_arr = [p0[x].value for x in p0 if p0[x].vary]
        nvarys = len(var_arr)
        sampler = emcee.EnsembleSampler(nwalkers, nvarys, _lnprob, pool=pool)
        rng = np.random.RandomState(12345)
        sampler.random_state = rng.get_state()
        
        p0_vals = (1 + rng.randn(nwalkers, nvarys) * 1.e-4) * var_arr
        sampler.run_mcmc(p0_vals, steps, progress=True)
    return sampler#, construct_p(p0_vals)
