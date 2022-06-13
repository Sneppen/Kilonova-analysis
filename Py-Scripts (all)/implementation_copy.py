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

#path = './spectra_filtered' #\AT2017gfo\AT2017gfo\dereddened+deredshifted_spectra' 
#files = os.listdir(path)

x = np.loadtxt('spectra_filtered/epoch1.txt').T#path+'/'+files[0]).T
#x= np.array([wl, flux, error,error]).T
exc_reg_2 = (~((x[:,0] > 13100) & (x[:,0] < 14400))) & (~((x[:,0] > 17550) & (x[:,0] < 19200))) & \
          (~((x[:,0] > 5330) & (x[:,0] < 5740))) & (~((x[:,0] > 9840) & (x[:,0] < 10300))) & \
          (x[:,0] > 3600) & (x[:,0] < 22500)

wl, flux, error = x[:,0][exc_reg_2], x[:,1][exc_reg_2], x[:,2][exc_reg_2]
bins = 100
binned_wl = wl# np.empty(bins)
binned_flux = flux # np.empty(bins)
binned_error = error #np.empty(bins)

#binned_wl = np.empty(bins)
#binned_flux = np.empty(bins)
#binned_error = np.empty(bins)
#for i, data in enumerate(np.array_split(np.vstack([wl, flux, error]).T, bins)):
#    binned_wl[i] = data[:,0].mean()
#    binned_flux[i] = np.average(data[:,1], weights = 1/data[:,2])
#    binned_error[i] = np.average(data[:,2], weights = 1/data[:,2])

c_cgs = csts.c.cgs.value
AA_in_cgs = units.AA.cgs.scale

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


use_new = False
p_cygni_line_corr = lambda *x: p_cygni_line_corr_new(*x) if new else p_cygni_line_corr_old(*x)
def p_cygni_line_corr_new(wl, v_out, v_phot, tau, lam, vref, ve, t0): 
    cyg = fast_integrator.calc_profile_Flam(50, t0, v_out * c_cgs,
                                v_phot * c_cgs, tau, vref *
                                 c_cgs, ve * c_cgs,
                                 lam * AA_in_cgs)#, _lam_min=7000, _lam_max=13000)
    return np.interp(wl, cyg[0]*1e-2, cyg[1])    

def p_cygni_line_corr_old(wl, v_out, v_phot, tau, lam, vref, ve, t0): 
    calc = PcygniCalculator(t=t0 * units.s, vmax=v_out * csts.c,
                                 vphot=v_phot * csts.c, tauref=tau, vref=vref *
                                 csts.c, ve=ve * csts.c,
                                 lam0=lam * units.AA)#, _lam_min=7000, _lam_max=13000)
    cyg = calc.calc_profile_Flam(npoints=50)
    inter = interp1d(cyg[0]*1e-10, cyg[1], bounds_error=False, fill_value=1)    
    return inter(wl)


pi = np.pi; h = 6.626e-34; c = 3.0e+8; k = 1.38e-23

@numba.njit(fastmath=True)
def temp(wav,T):
    a = 2.0*h*c**2
    b = h*c/(wav*k*T)
    return a/ ( (wav**5)*(np.exp(b) - 1.0) )
    
@numba.njit(fastmath=True)
def gaussian(wav, amp, mu, sigma): 
    return amp*np.exp(-(wav-mu)**2/(2*sigma**2))

#area
x_arr = np.linspace(0, np.pi/2, 10)
x_center = (x_arr[1:]+x_arr[:-1])/2
areas = 3*np.sin(x_center)*np.cos(x_center)**2 * (x_arr[1]-x_arr[0])


def planck_with_mod(wav, T, N, vmax, vphot, tau=0.55, occult = 1, ve=0.32,
                    amp1 = 2, cen1=15500, sig1=250,amp2 = 2, cen2=15500, sig2=250, 
                    v_bb = 0.2, T_power = 0.54,
                    vref=0., t0=120960, lam=10506.3653): #blackbody
    
    #vbb=0.28
    ##this part correctly weighs the areas
    corr = (1-v_bb**2)**(1/2)*1/(1-v_bb*np.cos(x_center))
    
    #This includes the cooling of temperatures using Drout2017
    t = v_bb*t0/(1-v_bb) - np.cos(x_center) * (v_bb*t0)/(1-v_bb*np.cos(x_center))
    #print('corr=',corr)
    T_n = T*((t0-t)/t0)**(-T_power)
    #print('Tn=',T_n)
    
    #intensity = np.average([temp2(wav, T_i, i)*i**5 for i,T_i in zip(corr,T_n)], axis=0, weights=areas)
    intensity = np.average([temp(wav, T_i*i) for i,T_i in zip(corr,T_n)], axis=0, weights=areas)
    
    #intensity = temp(wav, 2*T)

    pcyg_prof3 = p_cygni_line_corr(wav, vmax, vphot, 1/13.8*tau   , 10036.65, vref, ve , t0)
    pcyg_prof4 = p_cygni_line_corr(wav, vmax, vphot, 8.1/13.8*tau , 10327.311,vref, ve , t0)
    pcyg_prof5 = p_cygni_line_corr(wav, vmax, vphot, 4.7/13.8*tau , 10914.887,vref, ve , t0)
    correction = pcyg_prof3*pcyg_prof4*pcyg_prof5
    correction[correction>1] = (correction[correction>1]-1)*occult + 1
    
    # Gaussians
    gau1 = gaussian(wav, 1e-17*amp1, cen1, sig1)
    gau2 = gaussian(wav, 1e-17*amp2, cen2, sig2)

    intensity = intensity*correction
    return N*intensity+gau1+gau2


def residual(pars, wav, data=None, error=None): 
    model = planck_with_mod(wav, **pars)
    
    if data is None:
        return model
    return (model - data)/error


@numba.njit(fastmath=True)
def lnprob_inner(model, flux, error):
    return -0.5 * np.sum(((model - flux) / error)**2 + np.log(2 * np.pi * error**2))

def lnprob(pars):
    #print("inner", multiprocessing.current_process().name, pars["T"].value)

    """This is the log-likelihood probability for the sampling."""  
    model = residual(pars, binned_wl*1e-10)
    return lnprob_inner(model, binned_flux, binned_error)

def _lnprob(variable_pars, **static_pars):
    prob = lnprob({**variable_pars, **static_pars})
    return 1e100 if np.isnan(prob) else prob
    
def optimize(p0, use_new, nwalkers, steps):
    global new
    new = use_new
    import emcee
    mini = lmfit.Minimizer(lnprob, p0) 
    
    with multiprocessing.Pool(4) as pool:
        #pool = None
        var_dict = {x: p0[x].value for x in p0 if p0[x].vary}
        static_dict = {x: p0[x].value for x in p0 if not p0[x].vary}
        nvarys = len(var_dict)
        sampler = emcee.EnsembleSampler(nwalkers, nvarys, _lnprob, pool=pool, 
                                        parameter_names=list(var_dict.keys()), kwargs=static_dict)
        rng = np.random.RandomState(12345)
        sampler.random_state = rng.get_state()
        
        p0_vals = (1 + rng.randn(nwalkers, nvarys) * 1.e-4) * np.array(list(var_dict.values()))
        sampler.run_mcmc(p0_vals, steps, progress=True)
    return sampler
