import numpy as np
import pymc3 as pm
import pandas as pd
import theano as tt
from matplotlib import pyplot as plt
from scipy import interpolate


def fitFunc(x, lam0, eta):
    return lam0 * (x/1e12)**eta


dFrame = pd.read_csv('../../../DataScience/github/Gaussian_mixture/galaxyData.csv', index_col=0)
print(dFrame.columns)

x = dFrame.LIR.values
y = dFrame.LPeak.values
errX = dFrame.errLIR.values
errY = dFrame.errLPeak.values

with pm.Model() as model:
    # define uninformative priors for parameters in the fit
    lam0 = pm.Bound(pm.Flat,lower=15, upper=1000)('lam0')
    eta = pm.Bound(pm.Flat,lower=-10, upper=10)('eta')

    # expected value
    mu = fitFunc(x, lam0, eta)
    
    # likelihood
    Y_obs = pm.Normal('Y_obs', mu=mu, sd=errY, observed=y)

with model:
    step = pm.NUTS(target_accept=0.9)
    trwithOut = pm.sample(draws=2000, step=step, tune=1000, discard_tuned_samples=True)

print('\nSummary of trace: ')
print(pm.summary(trwithOut).round(2))

plt.rcParams['font.family']='serif'
plt.rcParams['font.size'] = 13
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['xtick.major.size'] = 8.#5.
plt.rcParams['xtick.minor.width'] = 1.5 #2.
plt.rcParams['xtick.minor.size'] = 4.
plt.rcParams['ytick.minor.size'] = 4.
plt.rcParams['ytick.major.width'] = 1.5 #0.9
plt.rcParams['ytick.major.size'] = 8.#5.
plt.rcParams['ytick.minor.width'] = 1.5 #2.
plt.rcParams['axes.linewidth'] = 2.
plt.rcParams['axes.labelweight'] = 'light'
plt.rcParams['axes.labelsize'] = 16.
plt.rcParams['axes.grid'] = False
plt.rcParams['ytick.labelsize'] = 12.
plt.rcParams['xtick.labelsize'] = 12.
plt.rcParams['figure.figsize'] = 6.25,6
plt.rcParams['figure.subplot.left'] = 0.17
plt.rcParams['figure.subplot.bottom'] = 0.17
plt.rcParams['figure.subplot.top'] = 0.89
plt.rcParams['figure.subplot.right'] = 0.89

xfit = np.logspace(np.log10(np.min(x)), np.log10(np.max(x)))  # x values for plotting best fit
yfit_withOutliers = fitFunc(xfit, np.median(trwithOut['lam0']),
               np.median(trwithOut['eta']))  # y values for plotting best fit


fig, ax = plt.subplots()
ax.axes.tick_params(right='on',top='on',direction='in',which='both')
plt.plot(xfit, yfit_withOutliers, linewidth=2.0, color='black', zorder=99, label='best fit with outliers')
plt.errorbar(x, y, xerr=errX, yerr=errY, fmt='o', color='red', alpha=0.5)
plt.xlabel('Luminosity (solar luminosities)')
plt.ylabel('Peak Wavelength (microns)')
plt.xscale('log')
plt.yscale('log')
plt.legend(fontsize='x-small')
plt.savefig('../../../DataScience/github/Gaussian_mixture/outlier-fit.png', dpi=75, bbox_inches='tight')
plt.show()

yInterp = interpolate.interp1d(xfit, yfit_withOutliers, bounds_error=False)
sortXInd = np.argsort(x)
yBestFit = yInterp(x[sortXInd])

fig, ax = plt.subplots()
ax.axes.tick_params(right='on',top='on',direction='in',which='both')
plt.plot(x, (y-yBestFit)/errY, 'r.')
# plt.axhline(np.nanmedian((y-yBestFit)/errY), label='median distance', linewidth=2.0, color='red')
plt.axhline(0, color='black', linewidth=2.0)
plt.xscale('log')
plt.ylabel('Residual / Standard Deviation')
plt.xlabel('Luminosity (solar luminosities)')
# plt.legend(fontsize='x-small')
plt.savefig('../../../DataScience/github/Gaussian_mixture/outlier-residual.png', dpi=75, bbox_inches='tight')
plt.show()

with pm.Model() as model:
    # Define bounded flat priors for our fit parameters
    lam0 = pm.Bound(pm.Flat,lower=15, upper=1000)('lam0')
    eta = pm.Bound(pm.Flat,lower=-10, upper=10)('eta')

    # Define linear model for the inliers
    y_in = fitFunc(x, lam0, eta)

    # Define weakly informative priors for the mean and variance of outliers
    y_out = pm.Normal('yest_out', mu=100, sigma=50)
    sigma_y_out = pm.HalfNormal('sigma_y_out', sigma=20)

    # Use Bernoulli distribution to sample whether a datapoint is an inlier or outlier
    frac_outliers = pm.Uniform('frac_outliers', lower=0.0, upper=0.5)
    is_outlier = pm.Bernoulli('is_outlier', p=frac_outliers, shape=x.shape[0],
                              testval=np.random.rand(x.shape[0]) < 0.2)

    # Extract observed y and sigma_y from dataset, encode as theano objects
    yobs = tt.shared(np.asarray(y, dtype=tt.config.floatX))
    sigma_y_in = np.asarray(errY, dtype=tt.config.floatX)

    # Set up normal distributions that give us the logp for both distributions
    inliers = pm.Normal.dist(mu=y_in, sigma=sigma_y_in).logp(yobs)
    outliers = pm.Normal.dist(mu=y_out, sigma=sigma_y_in + sigma_y_out).logp(yobs)
    # Build custom likelihood, a potential will just be added to the logp and can thus function
    # like a likelihood that we would add with the observed kwarg.
    pm.Potential('obs', ((1 - is_outlier) * inliers).sum() + (is_outlier * outliers).sum())

with model:
    step = pm.NUTS(target_accept=0.9)
    tr = pm.sample(draws=2000, step=step, tune=1000, discard_tuned_samples=True)

print('\nSummary of trace: ')
print(pm.summary(tr).round(2))

cutoffPercentile = 0.3  # Make a cut at 3 sigma
dFrame['actual_outlier'] = np.percentile(tr['is_outlier'], cutoffPercentile, axis=0)

print('n_outliers:')
print(sum(dFrame.actual_outlier == 1))
print('\nn_inliers:')
print(sum(dFrame.actual_outlier == 0))

xfit = np.logspace(np.log10(np.min(x)), np.log10(np.max(x)))  # x values for plotting best fit
yfit = fitFunc(xfit, np.median(tr['lam0']), np.median(tr['eta']))  # y values for plotting best fit

fig, ax = plt.subplots()
ax.axes.tick_params(right='on',top='on',direction='in',which='both')

plt.plot(xfit, yfit, linewidth=2.0, color='black', zorder=99, label='best fit no outlier')
plt.errorbar(dFrame.loc[dFrame.actual_outlier==0, 'LIR'],
             dFrame.loc[dFrame.actual_outlier==0, 'LPeak'],
             xerr=dFrame.loc[dFrame.actual_outlier==0, 'errLIR'],
             yerr=dFrame.loc[dFrame.actual_outlier==0, 'errLPeak'], fmt='o', alpha=0.5, label='inlier')
plt.errorbar(dFrame.loc[dFrame.actual_outlier==1, 'LIR'],
             dFrame.loc[dFrame.actual_outlier==1, 'LPeak'],
             xerr=dFrame.loc[dFrame.actual_outlier==1, 'errLIR'],
             yerr=dFrame.loc[dFrame.actual_outlier==1, 'errLPeak'], fmt='o', alpha=0.5, label='outlier')

plt.xlabel('LIR')
plt.ylabel(r'$\lambda_{peak}$')
plt.xscale('log')
plt.yscale('log')
plt.legend(fontsize='x-small')
plt.show()
