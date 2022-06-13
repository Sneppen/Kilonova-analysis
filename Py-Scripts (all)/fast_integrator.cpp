#include <cassert>
#include <math.h>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>


namespace py = pybind11;

double calc_r(double p, double z){
    /*
    Calculate radius of location (z, p) in ejecta;

    Parameters
    ----------
    p : double
        p-coordinate (perpendicular to line-of-sight to observer)
    z : double
        z-coordinate (along line-of-sight to observer)

    Returns
    -------
    r : double
        radius of location
    */
    return sqrt(p*p + z*z);
}

double calc_W(double r, double vphot, double t){
    /*
    Calculate geometric dilution factor

    Parameters
    ----------
    r : double
        radius of location

    Returns
    -------
    W : double
        geometric dilution factor
    */

    return (1 - sqrt(1 - pow(vphot*t / r,2))) / 2;
}

double calc_tau(double r, double vmax, double vphot, double t, double vdet_min, double vdet_max, double tauref, double ve){
    /*
    Calculate line optical depth at radius r, according to density profile.

    We assume an exponential density and thus optical depth profile as
    presented in Thomas et al. 2011.

    Arguments
    ---------
    r : double
        radius of location

    Returns
    -------
    tau : double
        line optical depth
    */

    double v = r / t;

    if (v >= vdet_min && v <= vdet_max)
        return tauref * exp( - v / ve);
    else
        return 1e-20;
}

double S(double p, double z, double vmax, double vphot, double t){
    /*
    Calculate source function at location (p, z) in ejecta.

    In case only the pure absorption component of the line profile is
    considered, the source function is of course 0. Otherwise, it follows
    from eq. 33 of Jeffery & Branch 1990.

    Parameters
    ----------
    p : double
        p-coordinate of location
    z : double
        z-coordinate of location
    mode : str
        flag setting the interaction mode: 'both' for full line profile,
        'abs' for pure absorption (default 'both')

    Returns
    -------
    S : double
        source function at location (p, z)
    */

    auto r = calc_r(p, z);

    if (r > vmax*t || r < vphot*t) // outside ejecta or inside photosphere
        return 0;
    else if (z < 0 && p < vphot*t)// occulted region
        return 0;
    else // emission region
        return calc_W(r, vphot, t);
}

double I(double p, double vphot, double t){
    /*
    Determine the initial specific intensity for a ray passing through (p,
    z) towards the observer.

    Used in eq. 71 of Jeffery & Branch 1990. Only if the line of sight
    going through (p, z) and towards the observer intersects the
    photosphere, a non-zero initial specific intensity is found.

    Parameters
    ----------
    p : double
        p-coordinate of location of interest
    z : double
        z-coordinate of location of interest

    Returns
    -------
    I : double
        initial specific intensity
    */

    if (p < vphot*t) // in the photosphere plane
        return 1;
    else
        return 0;    // above the photosphere plane
}

double tau(double p, double z, double vmax, double vphot, double t, double vdet_min, double vdet_max, double tauref, double ve){
    /*
    Determine the line optical on the line-of-sight towards the observer,
    at location (p, z).

    Used in eq.  of Jeffery & Branch 1990. Only locations in the emission
    region outside of the occulted zone may attenuated the radiation field.
    Thus, only there a non-zero optical depth is returned.

    Parameters
    ----------
    p : double
        p-coordinate of the location of interest
    z : double
        z-coordinate of the location of interest

    Returns
    -------
    tau : double
        optical depth at the location of interest
    */

    auto r = calc_r(p, z);

    if (r > vmax*t || r < vphot*t) // outside ejecta or inside photosphere
        return 0;
    else if (z < 0 && p < vphot*t) // occulted region
        return 0;
    else                           // emission region
        return calc_tau(r, vmax, vphot, t, vdet_min, vdet_max, tauref, ve); 
}

double Iemit(double p, double z, double vmax, double vphot, double t, double vdet_min, double vdet_max, double tauref, double ve){
    /*
    Determine the total specific intensity eventually reaching the observer
    from (p, z).

    The absorption or emission-only cases may be treated, or both effects
    may be included to calculate the full line profile. Used in eq. 71 of
    Jeffery & Branch 1990.

    Parameters
    ----------
    p : double
        p-coordinate of location of interest
    z : double
        z-coordinate of location of interest
    mode : str
        flag determining the line profile calculation mode: 'abs' for pure
        absorption, 'emit' for pure emission, 'both' for the full line
        profile calculation (default 'both')

    Returns
    -------
    Iemit : double
        total specific intensity emitted towards the observer from
        location (p, z)
    */
    auto tau_ = tau(p, z, vmax, vphot, t, vdet_min, vdet_max, tauref, ve);

    return (I(p, vphot, t) * exp(-tau_) + S(p, z, vmax, vphot, t) * (1 - exp(-tau_))) * p;
}

class IntegrandParams{
    public:
        double z, vmax, vphot, t, vdet_min, vdet_max, tauref, ve;
};
extern "C" {
double get_Iemit(double p, void * params_ptr){
    auto params = *static_cast<IntegrandParams*>(params_ptr);
    return Iemit(p, params.z, params.vmax, params.vphot, params.t,
        params.vdet_min, params.vdet_max, params.tauref, params.ve);
}
}

constexpr double c_cgs = 29979245800;
double calc_z(double nu, double t, double lam0){
    /*
    Calculate location (in terms of z) of resonance plane for photon
    emitted by the photosphere with frequency nu

    Parameters
    ----------
    nu : double
        photospheric photon frequency

    Returns
    -------
    z : double
        z coordinate of resonance plane
    */
    auto nu0 = c_cgs/lam0;
    return c_cgs * t * (1 - nu0 / nu);
}

double calc_line_flux(double nu, double vmax, double vphot, double t, double vdet_min, double vdet_max, double tauref, double ve, double lam0, gsl_integration_workspace * workspace){
    /*
    Calculate the emergent flux at LF frequency nu

    Parameters
    ----------
    nu : double
        lab frame frequency at which the line flux is to be calculated
    mode : str
        identifies the included interaction channels; see self.Iemit
        (default 'both')

    Returns
    -------
    Fnu : double
        emergent flux F_nu
    */

    double z = calc_z(nu, t, lam0);

    IntegrandParams params {z, vmax, vphot, t, vdet_min, vdet_max, tauref, ve};


    gsl_function iE;
    iE.function = &get_Iemit;
    iE.params = &params;

    double res, error;
    size_t n_eval;
    //double disc[] {t*vphot, sqrt(pow(vmax*t,2)-z*z), sqrt(pow(vphot*t,2)-z*z), sqrt(pow(vdet_min*t,2)-z*z), sqrt(pow(vdet_max*t,2)-z*z)};
    //gsl_integration_qagp (&iE, disc, (sizeof(disc)/sizeof(*disc)), 0, t*vmax, 1, 1.49e-06, 50, workspace, &res, &error);
    gsl_integration_qng (&iE, 0, t*vmax, 1, 1.49e-06, &res, &error, &n_eval);
//    if (gsl_integration_qng (&iE, 0, t*vmax, 1, 1.49e-06, &res, &error, &n_eval)){
//        py::print("Convergence failed on integral with parameters:", nu, ", ", vmax/c_cgs, ", ", vphot/c_cgs, ", ", t, ", ", vdet_min, ", ", vdet_max, ", ", tauref, ", ", ve, ", ", lam0);
//        py::print("\n reached error of ", error );
//    }
    return 2 * M_PI * res;
}
   

py::tuple calc_line_profile_base(int npoints, double nu_min, double nu_max, double vmax, double vphot, double t, double vdet_min, double vdet_max, double tauref, double ve, double lam0){
    /*
    Calculate the full line profile between the limits nu_min and nu_max in
    terms of F_nu.

    Parameters
    ----------
    nu_min : double
        lower frequency limit
    nu_max : double
        upper frequency limit
    npoints : int
        number of points of the equidistant frequency grid

    Returns
    -------
    nu : np.ndarray
        frequency grid
    Fnu : np.ndarray
        emitted flux F_nu
    */
    //py::gil_scoped_release release;
    auto nu_array = py::array_t<double>(npoints);
    auto bright_array = py::array_t<double>(npoints);
    auto nu_ptr = static_cast<double *>(nu_array.request().ptr);
    auto Fnu_ptr = static_cast<double *>(bright_array.request().ptr);

    auto workspace = gsl_integration_workspace_alloc (50);
    gsl_set_error_handler_off();
    double interval = (nu_max - nu_min) / (npoints-1);
    for (int idx = 0; idx < npoints; idx++){
        double nu = nu_min + interval * idx;
        Fnu_ptr[idx] = calc_line_flux(nu, vmax, vphot, t, vdet_min, vdet_max, tauref, ve, lam0, workspace);
        nu_ptr[idx] = nu;
    }
    gsl_integration_workspace_free (workspace);
    return py::make_tuple(nu_array, bright_array);
}


py::tuple calc_profile_Flam(int npoints, double t, double vmax, double vphot, double tauref, double vref, double ve, double lam0){
    double zmax = t * vmax;
    double dlambda = lam0 / t * zmax / c_cgs;
    double lam_min = lam0 - 1.05 * dlambda;
    double lam_max = lam0 + 1.05 * dlambda;

    auto lambda_array = py::array_t<double>(npoints);
    auto lambda_ptr = static_cast<double *>(lambda_array.request().ptr);
    auto Flambda_array = py::array_t<double>(npoints);
    auto Flambda_ptr = static_cast<double *>(Flambda_array.request().ptr);
    //py::gil_scoped_release release;

    auto workspace = gsl_integration_workspace_alloc (50);
    gsl_set_error_handler_off();

    double interval = (lam_max - lam_min) / (npoints-1);
    double F_nu0;
    for (int idx = 0; idx < npoints; idx++){
        double lam = lam_min + interval * idx;
        double nu = c_cgs/lam;
        double F_lambda = calc_line_flux(nu, vmax, vphot, t, vphot, vmax, tauref, ve, lam0, workspace);
        if (idx == 0)
            F_nu0 = F_lambda;//strange normalization
        double cont = F_nu0 * nu*nu / c_cgs;
        double F_lambda_normed = F_lambda*nu*nu/(c_cgs * cont);
        lambda_ptr[idx]  = F_lambda_normed;
        Flambda_ptr[idx]  = lam;
    }
    gsl_integration_workspace_free (workspace);

    return py::make_tuple(lambda_array, Flambda_array);
}

PYBIND11_MODULE(fast_integrator, m) {
    m.def("calc_line_profile_base", &calc_line_profile_base, "calculates the profile base");
    m.def("calc_profile_Flam", &calc_profile_Flam, "calculates the profile Flam");
}