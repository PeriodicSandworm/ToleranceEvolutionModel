# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 14:58:12 2025

@author: Jonas Guenzl
"""
import numpy as np
from math import comb
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

#define some colors for consistency
static_color = '#166F16'
cidal_color = '#C17D14'
wt_static_c = '#228B22'
wt_cidal_c = '#E1B000'
tol_static_c = '#0B3D0B'
tol_cidal_c = '#A0522D'


def size_distribution_ad(t, a, lam, mu, n_max = 100):
    if lam != mu:
        alpha = mu * (np.exp((lam - mu) * t) - 1) / (lam * np.exp((lam - mu) * t) - mu) if t > 0 else 0
        beta = lam * (np.exp((lam - mu) * t) - 1) / (lam * np.exp((lam - mu) * t) - mu) if t > 0 else 0
    else: 
        alpha = beta = (lam * t) / (1 + lam * t)

    pn_array = []
    for n in range(n_max):
        total = 0
        upper_limit = min(a, n)
        for j in range(0, upper_limit + 1):
            binom1 = comb(a, j)
            binom2 = comb(a + n - j - 1, a - 1)
            term = (
                binom1
                * binom2
                * alpha ** (a - j)
                * beta ** (n - j)
                * (1 - alpha - beta) ** j
            )
            total += term
        pn_array.append(total)
        if sum(pn_array) > 1-1e-3:
            pn_array.extend([0.0] * (n_max - len(pn_array)))
            break
    return pn_array

def plot_p0_vs_time(path, N0, growth_values, d, n_max, t_values, colors):
    '''Plot P(n=0) as a function of time for different growth rates.'''
    plt.figure(figsize=(8, 5))
    for idx, g in enumerate(growth_values):
        label = f"g = {g}, d = {d}"
        p0_values = [size_distribution_ad(t, N0, g, d, n_max)[0] for t in t_values]

        plt.plot(t_values, p0_values, label=label, linewidth=2, color=colors[idx])

    plt.xlabel("Time t", fontsize=15)
    plt.ylabel("Probability P(n=0)", fontsize=15)
    #plt.title("P(n=0) as a function of time", fontsize=15)
    #plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(path + "p0_t.pdf", transparent=True)
    plt.show()

def plot_p0_vs_N0(t, growth_values, d, n_max, N0_values, colors):
    path ='/Users/jonas/Documents/Cologne/Masterarbeit/Codes/Probability_distributions/ext_prob/'
    plt.figure(figsize=(8, 5))
    for idx, g in enumerate(growth_values):
        label = f"g= {g}, d = {d}"
        p0_values = [size_distribution_ad(t, N0, g, d, n_max)[0] for N0 in N0_values]
        plt.plot(N0_values, p0_values, label=label, linewidth=2, color=colors[idx])

    plt.xlabel(r"initial size $n_0$", fontsize=15)
    plt.ylabel("Probability P(n=0)", fontsize=15)
    #plt.title("P(n=0) as a function of a", fontsize=15)
    #plt.grid(True)
    plt.yscale('log')
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(path + "p0_n_0.pdf", transparent=True)
    plt.show()

def compute_surv_full_period(tau, t_mic, N0, g, d, eta, n_max =100, drug_type = 'biostatic'):
    '''Compute the survival probability after a full period of treatment, considering the initial size and treatment type.'''     
    # Define the probability of survival after the first phase
    def p0(t, death, growth):
        p = (death - death *np.exp((death-growth)*t))/(growth - death *np.exp((death-growth)*t))
        return(p)
    
    death_2 = d
    growth_2 = g
    
    if drug_type == 'biostatic':
        death_1 = d
        growth_1 = g - eta
        
    elif drug_type == 'biocidal':
        death_1 = d + eta
        growth_1 = g
        
    # compute size distribution after first phase
    pn_1 = size_distribution_ad(np.min([tau, t_mic]), N0, growth_1, death_1, n_max)
    
    if tau <= t_mic:
        ext = pn_1[0]
    else:
        ext_1 = pn_1[0]
        size_array = np.arange(len(pn_1))
        ext_2 = []
        
        for n in size_array[1:]:
            ext_2.append(p0(tau - t_mic, death_2, growth_2)**(n))
        
        ext = ext_1 + np.sum(np.multiply(ext_2, pn_1[1:]))
        
    surv = 1-ext
    return(surv)

# Example usage to compute the survival probability for static and cidal treatments over time 
if __name__ == "__main__":
    #Choose if you want to compare the wildtype and tolerance strain
    compare_tol = True
    # Define Parameters of the PKPD Model
    period = 6
    N0 =100
    g = 0.7
    d =0.4
    eta = 0.4
    n_max = 80    

    C=20
    alpha = 0.7
    # compute t_mic from the exponential decay of the drug concentration
    t_mic = np.log(C)/alpha

    #define time array
    time = np.linspace(0, period, 100)


    path ='path/to/save/plots/'

    plt.figure()
    
    # Compute survival probabilities for static and cidal treatments
    if compare_tol ==True:
        psurv_static = [compute_surv_full_period(t, t_mic,  N0, g,d,eta, n_max, 'biostatic') for t in time]
        psurv_cidal = [compute_surv_full_period(t, t_mic,  N0, g,d,eta, n_max, 'biocidal') for t in time]
        
        plt.plot(time, psurv_static, label=r"$wt_{static}$", linewidth=2, color=wt_static_c)
        plt.plot(time, psurv_cidal, label=r"$wt_{cidal}$", linewidth=2, color=wt_cidal_c)
        plt.vlines(t_mic, np.min(psurv_cidal), 1, color = "black", linestyle = ':')
        
    else:   
        psurv_static = [compute_surv_full_period(t, t_mic,  N0, g,d,eta, n_max, "biostatic") for t in time]
        psurv_cidal = [compute_surv_full_period(t, t_mic,  N0, g,d,eta, n_max, "biocidal") for t in time]
        
        plt.plot(time, psurv_static, label='static', linewidth=2, color=static_color)
        plt.plot(time, psurv_cidal, label="cidal", linewidth=2, color=cidal_color)
        plt.vlines(t_mic, np.min(psurv_cidal), 1, color = "black", linestyle = ':')
    
    
        

    plt.xlabel(r"Time $\tau$ [a.u.]", fontsize=15)
    plt.ylabel(r"Probability $P_{surv}^{(>)}(\tau)$", fontsize=15)
    plt.legend(fontsize=13)
    plt.tight_layout()
    plt.savefig(path + "/survival_probabilities_period.pdf", transparent=True)
    plt.show()