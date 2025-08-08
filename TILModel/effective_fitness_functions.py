import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

def effective_fitness_hill(lmax, lmin, C, nu, alphatau):
    '''This function defines the analytic solution of the effective fitness for a hill type DRC 
        together with a simple exponential decay.'''
    fitness = lmin*(1 + (1 + lmin/lmax) / (nu * alphatau) * np.log((C**nu * np.exp(-nu * alphatau) + lmin/lmax)/(C**nu + lmin/lmax)))
    return fitness

def effective_fitness_step(lmax, lmin, C, alphatau):
    fitness = lmax*(1-(1+lmin/lmax)/alphatau*np.log(C))
    return(fitness)

def fitness_landscape(ax, growth_array, death_array, C, alphatau, aspect, nu=4, DRC_type = 'hill'):
    fitness = np.zeros((len(death_array), len(growth_array)))
    for i, d in enumerate(death_array):
        for j, g in enumerate(growth_array):
            if DRC_type == 'step':
                fitness[i,j] = effective_fitness_step(g, d, C, alphatau)
            elif DRC_type == 'hill':
                fitness[i,j] = effective_fitness_hill(g, d, C, nu, alphatau)
    
    im = ax.imshow((fitness), cmap='RdYlGn', norm=TwoSlopeNorm(0), aspect = aspect, extent = (growth_array[0], growth_array[-1], death_array[-1], death_array[0]))
    ax.set_ylabel(r'$\lambda_{min}$ [time$^{-1}$]', fontsize = 15)
    ax.set_xlabel(r'$\lambda_{max}$ [time$^{-1}$]', fontsize = 15)
    ax.invert_yaxis()
    cb = plt.colorbar(im, orientation='vertical', ax = ax, location = 'right', shrink = 0.6)
    if DRC_type == 'step':
        cb.ax.set_ylabel(r'$\Lambda_{Step}$', fontsize = 15)
    elif DRC_type == 'hill':
        cb.ax.set_ylabel(r'$\Lambda_{Hill}$', fontsize = 15)

if __name__ == "__main__":
    # Example usage
    fig, ax = plt.subplots()
    growth_array = np.linspace(0.1, 1, 100)
    death_array = np.linspace(0.1, 1, 100)
    C = 4
    alphatau = 2.4
    nu =4 
    aspect = 'auto'
    
    fitness_landscape(ax, growth_array, death_array, C, alphatau, aspect, nu, DRC_type='hill')
    plt.show()