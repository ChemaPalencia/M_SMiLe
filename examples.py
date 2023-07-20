
# coding: utf-8

# Examples for M_SMiLe

from M_SMiLe import microlenses

# We'll also import some useful libraries
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Set plot params
mpl.rcParams.update({'font.size': 18,'font.family':'serif'})
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['xtick.minor.size'] = 3
mpl.rcParams['xtick.minor.width'] = 1
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['ytick.minor.size'] = 3
mpl.rcParams['ytick.minor.width'] = 1
mpl.rcParams['lines.linewidth'] = 3

mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True

# # Example 1
# Simple usage case

# Set lens propierties
lens_properties = {'mu_t': 1000,
                   'mu_r': 2,
                   'sigma_star': 15.6,
                   'zs': 5,
                   'zd': 0.8}

# Create canvas and set properties
fig, ax = plt.subplots(figsize=(9, 6))

ax.set_yscale('log')
ax.set_ylabel(r'PDF(log$_{10}(\mu))$')
ax.set_xlabel(r'log$_{10}(\mu)$')
ax.set_ylim(bottom=1e-5)
ax.set_xlim(left=1, right=5)

cmap = mpl.cm.get_cmap('plasma')
color =[cmap(0.8), cmap(0.2)]
ls = ['-', '--']
label = [r'$\mu_{\rm m}>0$', r'$\mu_{\rm m}<0$']

# Create a PDF for the lens system for each parity case
for i, parity in enumerate([1, -1]):
    # Set parity accordingly
    lens_properties['mu_t'] *= parity

    # Create object
    microlens = microlenses(**lens_properties)

    # Get PDF
    pdf, log_mu = microlens.get_pdf()

    # Plot
    ax.plot(log_mu, pdf, c=color[i], ls=ls[i], label=label[i])
    ax.legend(loc='upper right')

# # Example 2
# Get a continuous distribution of PDF by varying the surface mass density

# Set lens propierties
lens_properties = {'mu_t': 500,
                   'mu_r': 4,
                   'sigma_star': 1,
                   'zs': 5,
                   'zd': 0.8}

# Get system sigma_crit
microlens = microlenses(**lens_properties)
sigma_crit = microlens.sigma_crit

# Create a set of log-spaced sigma_* that times mu_t give 10 to 100 sigma_crit
sigma_star = sigma_crit / microlens.mu_t * np.logspace(1, 2, base=10, num=200)

# Create canvas and set properties
fig, ax = plt.subplots(figsize=(9, 6))

ax.set_yscale('log')
ax.set_ylabel(r'PDF(log$_{10}(\mu))$')
ax.set_xlabel(r'log$_{10}(\mu)$')
ax.set_ylim(bottom=1e-5)
ax.set_xlim(left=1.5, right=5.5)

cmap = mpl.cm.get_cmap('plasma')
color_ind = (np.arange(len(sigma_star))/len(sigma_star)) * 80/100

for i, ele in enumerate(sigma_star):
    # Update sigma_star
    lens_properties['sigma_star'] = ele

    # Draw a line for each sigma_star
    microlens = microlenses(**lens_properties)
    pdf, log_mu = microlens.get_pdf()
    ax.plot(log_mu, pdf, c=cmap(color_ind[i]), alpha=0.2)

# Draw a curve indicating some mid-points
for i, ele in enumerate([10, 14, 20, 30, 50, 100]):
    # Update sigma_star
    lens_properties['sigma_star'] = ele * sigma_crit / microlens.mu_t

    # Draw a line for each sigma_star
    microlens = microlenses(**lens_properties)
    pdf, log_mu = microlens.get_pdf()
    color_ind_c = np.argmin(np.abs(ele*sigma_crit/microlens.mu_t - sigma_star))
    ax.plot(log_mu, pdf, c='k', lw=3.5)
    ax.plot(log_mu, pdf, c=cmap(color_ind[color_ind_c]),
            label=r'$\Sigma_{\rm eff}/\Sigma_{\rm crit}=$'
            + f'{microlens.sigma_ratio:.0f}', lw=2)
    ax.legend(loc='upper right')
