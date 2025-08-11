# -*- coding: utf-8 -*-
"""Created on Fri Aug  9 14:58:12 2025

@author: Jonas Guenzl"""
import os
from Stochastic_utils import size_distribution_ad
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

#define some colors for consistency
static_color = '#166F16'
cidal_color = '#C17D14'
wt_static_c = '#228B22'
wt_cidal_c = '#E1B000'
tol_static_c = '#0B3D0B'
tol_cidal_c = '#A0522D'

# Choose path to save results
path = 'path/to/save/results/'  # specify the path to save results

# Choose treatment
drug = 'cidal'
strain = 'wt'

# choose initial size
N0 = 50

# Decide if you want to show both treatment modes in one histogram
merge_cidal_static = True

# Define phenotypes
if strain == 'wt':
    g = 1
    d =0.4
    eta = 0.9
    #set limit for histograms
    n_max = 60
    
elif strain == 'tol':
    g = 0.7
    d =0.4
    eta = 0.4
    #set limit for histograms
    n_max = 80

# compute rates of the BD process according to treatment mode
if drug =='static':
    lam = g-eta
    mu = d
if drug =='cidal':
    lam = g
    mu = d + eta
    
time_points = np.linspace(0, 6, 70)  # Includes t = 0

# compute size distribution for every given timepoint
if merge_cidal_static == True:
    distributions_s = []
    distributions_c = []
    max_prob = 0
    for t in time_points:
        dist_s = size_distribution_ad(t, N0, lam = g-eta, mu =d, n_max =n_max)
        distributions_s.append(dist_s)
        dist_c = size_distribution_ad(t, N0, lam= g, mu=d+eta, n_max =n_max)
        distributions_c.append(dist_c)
        
else:
    distributions = []
    max_prob = 0
    for t in time_points:
        dist = size_distribution_ad(t, N0, lam, mu, n_max)
        distributions.append(dist)
        max_prob = max(max_prob, max(dist))

# Plot with consistent number of bars
if merge_cidal_static == True:
    name = 'n0_'+ str(N0) + '_'+str(strain)+'_static_and_cidal'
    if not os.path.exists(path):
        os.mkdir(path)     # create new folder for each analysis
    for idx, (t, dist_s, dist_c) in enumerate(zip(time_points, distributions_s, distributions_c)):
        #numerical fix
        dist_s = [x if 0 <= x <= 1 else 0 for x in dist_s]
        dist_c = [x if 0 <= x <= 1 else 0 for x in dist_c]
        plt.figure(figsize=(8, 5))
        if strain =='wt':
            plt.bar(range(n_max), dist_s, color=wt_static_c, edgecolor='black')
            plt.bar(range(n_max), dist_c, color=wt_cidal_c, edgecolor='black', alpha = 0.6)
        elif strain =='tol':
            plt.bar(range(n_max), dist_s, color=tol_static_c, edgecolor='black')
            plt.bar(range(n_max), dist_c, color=tol_cidal_c, edgecolor='black', alpha = 0.6)

        plt.title(f"Distribution at time t = {t:.2f}", fontsize=15)
        plt.xlabel("n", fontsize=15)
        plt.ylabel("P(n, t)", fontsize=15)
        plt.ylim(0, 0.3)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        plt.savefig(path+name+f"/distribution_t{idx}.pdf", transparent=True)
        plt.close()
else:
    name = 'n0_'+ str(N0) + drug +'_'+str(strain)
    if not os.path.exists(path):
        os.mkdir(path)     # create new folder for each analysis
    for idx, (t, dist_s, dist_c) in enumerate(zip(time_points, distributions_s, distributions_c)):
        #numerical fix
        dist_s = [x if 0 <= x <= 1 else 0 for x in dist_s]
        dist_c = [x if 0 <= x <= 1 else 0 for x in dist_c]
        plt.figure(figsize=(8, 5))
        if strain == 'wt' and drug == 'static':
            plt.bar(range(n_max), dist, color=wt_static_c, edgecolor='black')
        if strain == 'wt' and drug == 'cidal':
            plt.bar(range(n_max), dist, color=wt_cidal_c, edgecolor='black')
        if strain == 'tol' and drug == 'static':
            plt.bar(range(n_max), dist, color=tol_static_c, edgecolor='black')
        if strain == 'tol' and drug == 'cidal':
            plt.bar(range(n_max), dist, color=tol_cidal_c, edgecolor='black')
            
        plt.title(f"Distribution at time t = {t:.2f}", fontsize=15)
        plt.xlabel("n", fontsize=15)
        plt.ylabel("P(n, t)", fontsize=15)
        plt.ylim(0, 1.1)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        plt.savefig(path+name+f"/distribution_t{idx}.pdf", transparent=True)
        print(dist)
        plt.close()