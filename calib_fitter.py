"""
Author: Sergey E. Koposov
Email: skoposov __AT__ ed __DOT__ ac __DOT__ uk

"""
import numpy as np
import scipy.stats
import dynesty
from contextlib import nullcontext
import multiprocessing as mp
"""
-xlim<X<xlim


x1, e1, x2, e2

floor, multiplier

We model the d=x1-x2 as

f_outl P(d | outl) + ( 1-f_outl) P(d|good)

P(d|outl)= N_*(d|0,sigma_outl)
P(d|good)= N_*(d|off,sigma_X)
where sigma_x = np.sqrt((mult*e1)**2 + (mult*e2)**2 + floor**2)



"""


def gau_cdf(x, cen, sig):
    """ Fast N(x|cen,sig) gaussian cdf """
    return scipy.special.ndtr((x - cen) / sig)


def gau_lpdf(x, cen, sig):
    """ Log PDF of N(x|cen,sig) """
    return np.log(1 / np.sqrt(2 * np.pi) / sig) - 0.5 * ((x - cen) / sig)**2


def trunc_gau_lpdf(x, cen, sig, left, right):
    """ Log PDF of N(x|cen,sig) truncated at left, right """
    # N = scipy.stats.norm(cen, sig)
    norm = gau_cdf(right, cen, sig) - gau_cdf(left, cen, sig)
    eps = 1e-30
    norm = np.maximum(eps, norm)
    return gau_lpdf(x, cen, sig) - np.log(norm)


def like(p, data):
    dx, xlim, e1, e2 = data
    outl_frac, off, mult, floor, outl_sig = p

    logl_good = trunc_gau_lpdf(dx, off,
                               np.sqrt((e1**2 + e2**2) * mult**2 + floor**2),
                               -xlim, xlim)
    logl_outl = trunc_gau_lpdf(dx, 0, outl_sig, -xlim, xlim)
    ret = np.logaddexp(
        np.log(outl_frac) + logl_outl,
        np.log1p(-outl_frac) + logl_good)
    ret = np.sum(ret)
    if not np.isfinite(ret):
        print('Warning, not finite likelihood value!')
        ret = -1e100
    # print(ret, p)
    return ret


class LogUniPrior:

    def __init__(self, x1, x2):
        self.x1 = x1
        self.x2 = x2

    def __call__(self, x):
        return np.exp(np.log(self.x1) + x * np.log(self.x2 / self.x1))


class UniPrior:

    def __init__(self, x1, x2):
        self.x1 = x1
        self.x2 = x2

    def __call__(self, x):
        return self.x1 + x * (self.x2 - self.x1)


class Prior:

    def __init__(self,
                 max_outl_frac=0.3,
                 min_outl_sig=10,
                 max_outl_sig=1000,
                 min_off=-5,
                 max_off=5,
                 min_floor=0.001,
                 max_floor=10,
                 min_mult=0.5,
                 max_mult=2):
        self.outl_frac_prior = UniPrior(0, max_outl_frac)
        self.off_prior = UniPrior(min_off, max_off)
        self.mult_prior = LogUniPrior(min_mult, max_mult)
        self.floor_prior = LogUniPrior(min_floor, max_floor)
        self.outl_sig_prior = LogUniPrior(min_outl_sig, max_outl_sig)

    def __call__(self, p):
        outl_frac0, off0, mult0, floor0, outl_sig0 = p
        pret = np.array([
            self.outl_frac_prior(outl_frac0),
            self.off_prior(off0),
            self.mult_prior(mult0),
            self.floor_prior(floor0),
            self.outl_sig_prior(outl_sig0)
        ])
        return pret


class DataCache:
    data = None


def cache_wrapper(p):
    return like(p, DataCache.data)


def cache_init(data):
    DataCache.data = data


def fit(v1,
        e1,
        v2,
        e2,
        vlim=10,
        max_outl_frac=0.3,
        min_outl_sig=10,
        max_outl_sig=1000,
        min_off=-5,
        max_off=5,
        min_floor=0.001,
        max_floor=10,
        min_mult=0.5,
        max_mult=2,
        nthreads=1):
    """
    Fit pairwise differences to determine error calibration and floor

    Parameters:
    -----------
    v1: ndarray
       Measurement array from instrument 1
    ev1: ndarray
       Measurement uncertainty array from instrument 1
    v2: ndarray
       Measurement array from instrument 2 (must have the same length
       as array v1)
    ev2: ndarray
       Measurement uncertainty array from instrument 2
    vlim: float
       The differences will be modelled only in the -vlim...vlim interval

    vlim the interval of differences  that will be modeled
    I.e. the modelling will be from -vlim to vlim
    
    """
    dv = v1 - v2
    xind = np.abs(dv) < vlim
    data = dv[xind], vlim, e1[xind], e2[xind]
    prior = Prior(max_outl_frac=0.3,
                  min_outl_sig=min_outl_sig,
                  max_outl_sig=max_outl_sig,
                  min_off=min_off,
                  max_off=max_off,
                  min_floor=min_floor,
                  max_floor=max_floor,
                  min_mult=min_mult,
                  max_mult=max_mult)

    ndim = 5

    DataCache.data = data
    rstate = np.random.default_rng(24573343)
    with (mp.Pool(nthreads, initializer=cache_init, init_args=data)
          if nthreads > 1 else nullcontext()) as poo:
        dns = dynesty.DynamicNestedSampler(
            cache_wrapper,
            prior,
            ndim,
            sample='rwalk',
            #                                           logl_args=(data, ),
            pool=poo,
            queue_size=nthreads,
            rstate=rstate)
        dns.run_nested()
    res = dns.results
    samp = res.samples_equal()

    names = ['outlierfrac', 'off', 'mult', 'floor', 'outlsig']
    return dict(zip(names, samp.T))


if __name__ == '__main__':
    import corner
    N = 1000
    rng = np.random.default_rng(3232)

    e1 = rng.uniform(0.1, 3, N)
    e2 = rng.uniform(0.5, 5, N)

    cens = rng.normal(0, 100, N)

    add_floor = 1
    x1 = np.random.normal(cens, (e1**2 + add_floor**2)**.5, N)
    x2 = np.random.normal(cens, e2, N)
    samp = fit(x1, e1, x2, e2)

    corner.corner(samp,
                  labels=['outlierfrac', 'off', 'mult', 'floor', 'outlsig'])
