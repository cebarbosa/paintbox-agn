import os
import shutil

import numpy as np
from scipy.optimize import least_squares
from scipy import stats
import astropy.units as u
import astropy.constants as const
from astropy.modeling.models import Gaussian1D
from astropy.table import Table, hstack, vstack
from specutils import Spectrum1D
import emcee
import multiprocessing as mp
from tqdm import tqdm

import paintbox as pb
from paintbox.utils import CvD18, disp2vel
import matplotlib.pyplot as plt

import context
from der_snr import DER_SNR

def load_data(galaxy):
    specname = f"{galaxy.lower()}_osiris.fits"
    filename = os.path.join(context.data_dir, galaxy, specname)
    spec1d = Spectrum1D.read(filename)
    wave = spec1d.spectral_axis.value
    mask = np.full(len(wave), True)
    windows = [[19881, 20156], [20450, 20610]]
    for window in windows:
        idx = np.where((wave >= window[0]) & (wave <= window[1]))[0]
        mask[idx] = False
    flux = spec1d.flux
    return wave, flux, mask

def make_paintbox_model(wave, sigma=100, dlam=100):
    """ Build a paintbox model with stars, gas, polynomials terms, etc. """
    # Using velocity of galaxy to set wavelength of models
    # c = const.c.to("km/s").value
    # Vmin = V - 2000
    # wmin = wave[0] * np.sqrt((1 + Vmin/c)/(1 - Vmin/c)) - 100
    # wmax = wave[-1] + 100
    wrange = [19000, 24000]
    wtemp = disp2vel(wrange, 50)
    # # Preparing stellar population models
    #
    # templates_dir = os.path.join(context.home_dir, "templates")
    # if not os.path.exists(templates_dir):
    #     os.mkdir(templates_dir)
    # store = os.path.join(templates_dir, "CvD18_osiris")
    # elements = []
    # cvd = CvD18(wtemp, store=store, libpath=context.cvd_dir, sigma=sigma,
    #             elements=elements)
    # # Fixing slope of IMF to a Kroupa IMF
    # krpa_imf1 = 1.3
    # krpa_imf2 = 2.3
    # ssp_krpa = pb.FixParams(cvd, {"x1": krpa_imf1, "x2": krpa_imf2})
    # ssp_kin = pb.LOSVDConv(ssp_krpa, losvdpars=["V_star", "sigma_star"])
    # sed = pb.Resample(wave, ssp_kin)
    limits = {}
    # Include polynomial
    degree = np.ceil((wave.max() - wave.min()) / dlam).astype(int)
    sed = pb.Polynomial(wave, degree=degree)
    # Emission lines
    linenames = ["SiVI", "HBrgama1", "HBrgama2", "CaVIII"]
    linewaves = np.array([1.964, 2.166, 2.166, 2.321]) * u.micrometer
    linewaves = linewaves.to(u.Angstrom).value
    velscale_em = 30
    wave_em = disp2vel(wrange, velscale_em)
    for linename, linewave in zip(linenames, linewaves):

        emission = Gaussian1D(amplitude=1, mean=linewave,
                       stddev=sigma/velscale_em)(wave_em)
        em = pb.NonParametricModel(wave_em, emission, names=[f"em_{linename}"])
        em_kin = pb.LOSVDConv(em, losvdpars=[f"V_{linename}", f"sigma_"
                                                              f"{linename}"])
        sed += pb.Resample(wave, em_kin)
    return sed, limits, degree

def set_priors(parnames, limits):
    """ Defining prior distributions for the model. """
    priors = {}
    for parname in parnames:
        name = parname.split("_")[0]
        if name in limits:
            vmin, vmax = limits[name]
            delta = vmax - vmin
            priors[parname] = stats.uniform(loc=vmin, scale=delta)
        elif name == "V":
            priors[parname] = stats.norm(loc=0, scale=300)
        elif parname == "eta":
            priors["eta"] = stats.uniform(loc=1., scale=19)
        elif parname == "nu":
            priors["nu"] = stats.uniform(loc=2, scale=20)
        elif name == "sigma":
            priors[parname] = stats.uniform(loc=50, scale=300)
        elif name == "em":
            priors[parname] = stats.uniform(loc=0, scale=5)
        elif name == "p":
            porder = int(parname.split("_")[1])
            if porder == 0:
                mu, sd = 1, 0.2
                a, b = (0 - mu) / sd, (np.infty - mu) / sd
                priors[parname] = stats.truncnorm(a, b, mu, sd)
                # priors[parname] = stats.uniform(0., 1.)
            elif porder == 1:
                priors[parname] = stats.norm(.1, .5)
            else:
                priors[parname] = stats.norm(0, .1)
        else:
            raise ValueError(f"parameter without prior: {parname}")
    return priors

def log_probability(theta):
    """ Calculates the probability of a model."""
    global priors
    global logp
    lp = np.sum([priors[p].logpdf(x) for p, x in zip(logp.parnames, theta)])
    if not np.isfinite(lp) or np.isnan(lp):
        return -np.inf
    ll = logp(theta)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll

def run_sampler(outdb, nsteps=5000):
    global logp
    global priors
    ndim = len(logp.parnames)
    nwalkers = 2 * ndim
    pos = np.zeros((nwalkers, ndim))
    logpdf = []
    for i, param in enumerate(logp.parnames):
        logpdf.append(priors[param].logpdf)
        pos[:, i] = priors[param].rvs(nwalkers)
    # for n in range(nwalkers):
    #     plt.plot(logp.wave, logp.model(pos[n,:-1]), c="0.8")
    # plt.plot(logp.wave, logp.observed)
    plt.show()
    backend = emcee.backends.HDFBackend(outdb)
    backend.reset(nwalkers, ndim)
    try:
        pool_size = context.mp_pool_size
    except:
        pool_size = 1
    pool = mp.Pool(pool_size)
    with pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                         backend=backend, pool=pool)
        sampler.run_mcmc(pos, nsteps, progress=True)
    return

def make_table(trace, outtab):
    data = np.array([trace[p].data for p in trace.colnames]).T
    v = np.percentile(data, 50, axis=0)
    vmax = np.percentile(data, 84, axis=0)
    vmin = np.percentile(data, 16, axis=0)
    vuerr = vmax - v
    vlerr = v - vmin
    tab = []
    for i, param in enumerate(trace.colnames):
        t = Table()
        t["param"] = [param]
        t["median"] = [round(v[i], 5)]
        t["lerr".format(param)] = [round(vlerr[i], 5)]
        t["uerr".format(param)] = [round(vuerr[i], 5)]
        tab.append(t)
    tab = vstack(tab)
    tab.write(outtab, overwrite=True)
    return tab

def plot_fitting(wave, flux, mask, sed, trace, output, fluxerr=None):
    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]},
                            figsize=(2 * context.fig_width, 3))
    t = np.array([trace[p].data for p in sed.parnames]).T
    n = len(t)
    models = np.zeros((n, len(wave)))
    y = np.percentile(models, 50, axis=(0,))
    for j in tqdm(range(len(trace)), desc="Generating models "
                                                     "for trace"):
        models[j] = sed(t[j])
    y = np.percentile(models, 50, axis=(0,))
    ax0 = fig.add_subplot(axs[0])
    if fluxerr is not None:
        ax0.plot(wave, flux, fmt="-", ecolor="0.8", c="tab:blue")
    else:
        ax0.errorbar(wave, flux, yerr=fluxerr, fmt="-",
                     ecolor="0.8", c="tab:blue")
    ax0.plot(wave, y, c="tab:orange")
    ax0.xaxis.set_ticklabels([])
    ax0.set_ylabel("Flux")
    ax1 = fig.add_subplot(axs[1])
    if fluxerr is None:
        ax1.plot(wave, 100 * (flux - y) / flux, "-", c="tab:blue")
    else:
        ax1.errorbar(wave, 100 * (flux - y) / flux,
                     yerr=100 * fluxerr, fmt="-", ecolor="0.8", c="tab:blue")
    ax1.plot(wave, 100 * (flux - y) / flux, c="tab:orange")
    ax1.set_ylabel("Res. (\%)")
    ax1.set_xlabel("$\lambda$ (Angstrom)")
    ax1.set_ylim(-5, 5)
    ax1.axhline(y=0, ls="--", c="k")
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.show()
    plt.clf()
    plt.show()
    plt.close(fig)
    return

def run_paintbox(galaxy, nsteps=5000, loglike="normal", sigma=100, z=0):
    """ Run paintbox. """
    global logp, priors
    wdir = os.path.join(context.data_dir, galaxy)
    # Defining velocity of galaxy from redshift estimate
    c = const.c.to("km/s").value
    V0 = c * ((z+1)**2 - 1) / ((z+1)**2 + 1)
    # Loading data
    wave, flux, mask = load_data(galaxy)
    flux /= np.median(flux) # Normalize input spectrum
    sn = DER_SNR(flux)
    fluxerr = flux / sn
    # Make paintbox model
    sed, limits, porder = make_paintbox_model(wave, sigma=sigma)
    if loglike == "normal2":
        logp = pb.Normal2LogLike(flux, sed, mask=mask, obserr=fluxerr)
    elif loglike == "normal":
        logp = pb.NormalLogLike(flux, sed, mask=mask)
    elif loglike == "studt2":
        logp = pb.StudT2LogLike(flux, sed, mask=mask, obserr=fluxerr)
    # Making priors
    priors = set_priors(logp.parnames, limits)

    # Perform fitting
    dbname = f"{galaxy}_{loglike}_nsteps{nsteps}.h5"
    # Run in any directory outside Dropbox to avoid problems
    tmp_db = os.path.join(os.getcwd(), dbname)
    outdb = os.path.join(wdir, dbname)
    if not os.path.exists(outdb):
        run_sampler(tmp_db, nsteps=nsteps)
        shutil.move(tmp_db, outdb)
    # Post processing of data
    if context.node in context.lai_machines: #not allowing post-processing @LAI
        return
    reader = emcee.backends.HDFBackend(outdb)
    tracedata = reader.get_chain(discard=int(nsteps * 0.9), flat=True, thin=50)
    trace = Table(tracedata, names=logp.parnames)
    outtab = os.path.join(outdb.replace(".h5", "_results.fits"))
    make_table(trace, outtab)
    outimg = outdb.replace(".h5", ".png")
    plot_fitting(wave, flux, mask, sed, trace, outimg)

if __name__ == "__main__":
    sample = os.listdir(context.data_dir)
    z = {"NGC4151": 0}
    for galaxy in sample:
        print(f"Processing galaxy {galaxy}")
        run_paintbox(galaxy, z=z[galaxy])