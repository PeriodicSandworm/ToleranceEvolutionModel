#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  8 12:16:14 2025

@author: jguenzl

The first part defines some functions that read and return the desired data. In the second part the user 
can choose which experiment to analyze and execute the respective function. The desired paths have to be set.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from size_distribution_after_death import size_distribution_ad

b_color = (30/255, 144/255, 255/255)
#define some colors for consistency
static_color = '#166F16'
cidal_color = '#C17D14'
wt_static_c = '#228B22'
wt_cidal_c = '#E1B000'
tol_static_c = '#0B3D0B'
tol_cidal_c = '#A0522D'

# Functions that are capable of reading the data
def load_data(file):
    data = np.loadtxt(file, skiprows = 1, delimiter = ',')
    return(data)

def find_in_metadata(to_find, metadata):
    '''Finds the value assigned to `to_find` in the metadata file.'''
    with open(metadata, 'r') as f:
        lines = f.readlines()

    a_value = None
    for line in lines:
        if line.strip().startswith(f"{to_find} ="):
            a_str = line.strip().split('=', 1)[1].strip()
            try:
                # Handle array-like syntax: e.g., [0.8 0.6] -> [0.8, 0.6]
                if a_str.startswith('[') and ' ' in a_str:
                    a_str = a_str.replace('[', '').replace(']', '')
                    a_str = '[' + ','.join(a_str.split()) + ']'
                a_value = eval(a_str, {"__builtins__": None}, {})
            except Exception as e:
                print(f"Could not evaluate value for '{to_find}':", e)
            break
    return a_value

def visualize_surv_prob(experiment, files, colors, labels, name, show_mic = False, save = False):
    '''This function takes different data and visualizes all this in one plot'''
    fig = plt.figure()
    ax = fig.add_subplot()
    
    for i, f in enumerate(files):
        if experiment == 'time_dependency/':
            data = pd.read_csv(f,  usecols=lambda col: col == '# end' or col.startswith("p0") or col.startswith(' p0'))
        elif experiment == 'size_dependency/':
            data = pd.read_csv(f,  usecols=lambda col: col == '# N0' or col.strip().startswith("p0") or col.strip().startswith('p0_theo'))
        
        data = np.array(data)
        print(data)
        ax.scatter(data[:,0], data[:,1], color = colors[i], s = 2)
        ax.plot(data[:,0], data[:,2], color = colors[i], label = labels[i], alpha = 1)
        ax.set_ylabel(r'$P_{surv}(t)$', fontsize = 15)
        legend = ax.legend(fontsize = 13)
        legend.get_frame().set_alpha(0.0)
        
        
        if experiment == 'time_dependency/':
            ax.set_xlabel('t', fontsize = 15)
        elif experiment == 'size_dependency/':
            ax.set_xlabel('N', fontsize = 15)
            
    if show_mic == True:
        path = f.removesuffix('data.txt')
        t_mic = find_in_metadata('t_mic', path + 'metadata.txt')
        ax.vlines(t_mic, 0, 1, color = 'black', ls = '--', alpha = 0.6)
        
        
    if save == True:
        ax.axis([0,6.1,0.4,1.02])
        plt.tight_layout()
        plt.savefig(name, transparent = True)

    
def visualize_population_size(files, colors, labels, name, show_theory = False, show_effitness = False, save = False):
    '''This function visualizes the population size as a function of time. 
    It is capable to compare stochastic simulations to the effective fitness and can average multiple stochastic simulations'''
    

def reshape_from_metadata(array, metadata_path='metadata.txt'):
    '''This function is ment to reshape the data from landscape files the 2D numpy array'''
    with open(metadata_path, 'r') as f:
        for line in f:
            if line.startswith("Resolution:"):
                _, res = line.strip().split(":")
                width, height = map(int, res.strip().lower().split('x'))
                break
        else:
            raise ValueError("Resolution not found in metadata file.")
    
    if array.size != width * height:
        raise ValueError(f"Array of size {array.size} cannot be reshaped to ({width}, {height}).")

    return array.reshape((height, width))  # Note: height = rows, width = columns
        
def visualize_landscape(file, exp = 'size'):
    '''This function visualizes the survival probability landscape data from the given file.'''
    fig = plt.figure()
    ax = fig.add_subplot()
    if exp == 'size':
        data = np.loadtxt(file + 'size.txt', delimiter = ',', skiprows = 2)
    elif exp ==  'surv':
        data = np.loadtxt(file + 'surv.txt', delimiter = ',', skiprows = 1)
    #data = reshape_from_metadata(np.loadtxt(file + 'data.txt'), file + 'metadata.txt')
    print(np.shape(data))

def visualize_colony_size(files, labels, name, show_effitness = False, save = False):
    '''This function visualizes the colony size data from the given files.
    It is capable to compare stochastic simulations to the deterministic effective fitness.'''
    fig = plt.figure()
    ax = fig.add_subplot()
    
    for i, f in enumerate(files):
        data = pd.read_csv(f,  usecols=lambda col: col == '# end' or col.startswith("# t") or col.startswith("N(end)") or col.startswith(' N(end)') or col.startswith(' N(t)') or col.startswith('N(t)'))
        data = np.array(data)
        print(data)
        for j in range(1, len(data[0])):
            ax.scatter(data[:,0], data[:,j], color = b_color, s = 0.5, label = labels[j-1], alpha = 0.9)
            
        ax.set_ylabel(r'$\langle N(t)\rangle$', fontsize = 15)
        ax.set_xlabel('t', fontsize = 15)
       
        
        if show_effitness == True:
            path = f.removesuffix('data.txt')
            period = find_in_metadata('period', path + 'metadata.txt')
            t_mic = find_in_metadata('t_mic', path + 'metadata.txt')
            if 'stochasticmixedcolony'.lower() in path.lower():
                growth = np.array(find_in_metadata('g', path + 'metadata.txt'))
                death = np.array(find_in_metadata('d', path + 'metadata.txt'))
                lam = np.array(find_in_metadata('lam', path + 'metadata.txt'))
                N0 = np.array(find_in_metadata('N0', path + 'metadata.txt'))
            elif 'stochasticcolony'.lower() in path.lower():
                growth = np.array([find_in_metadata('g', path + 'metadata.txt')])
                death = np.array([find_in_metadata('d', path + 'metadata.txt')])
                lam = np.array([find_in_metadata('lam', path + 'metadata.txt')])
                N0 = np.array([find_in_metadata('N0', path + 'metadata.txt')])
            else:
                print('no data loaded')
            drug = find_in_metadata('drug', path + 'metadata.txt')
            K = find_in_metadata('K', path + 'metadata.txt')
            gamma = find_in_metadata('gamma', path + 'metadata.txt')
            try:
                end = find_in_metadata('end', path + 'metadata.txt')
                time = np.linspace(0,end)
            except:
                time = np.linspace(0, np.max(data[:,0]))
            
            effitness = 1/period*((growth-death)*(period-t_mic) - (lam + death-growth) *t_mic)
            ax.plot(time, N0*np.exp(effitness * time), label = r'$N(t) = N_0 e^{\Lambda_{step} t}$' , color = b_color)
         
    if save == True:
        plt.tight_layout()
        plt.savefig(name, transparent = True)   
    
def visualize_rescue_and_survival(files, colors, labels, experiment, name, legend = True, save = False):
    '''This function visualizes the rescue and survival probabilities from the given files.
    It takes an experiment argument, which is either 'var' for standing gen. variation or 'mut' for mutation.
    The name is the path and name of the saved figure.'''

    fig = plt.figure()
    ax = fig.add_subplot()
    
    for i, f in enumerate(files):
        if 'rescue_scenarios' in f:
            data = pd.read_csv(f,  usecols=lambda col: col == '# freq' or col.startswith("# q_mut")or col.startswith("p_res") or col.startswith(' p_res'))
            data = np.array(data)
            if experiment == 'var':
                data[:,0] *= 100
            ax.scatter(data[:50,0], data[:50,1], color = colors[i], s = 30, marker = 'x')
            
        elif 'size_dependency/' in f:
            data = pd.read_csv(f,  usecols=lambda col: col == '# N0' or col.strip().startswith("p0") or col.strip().startswith('p0_theo'))
            data = np.array(data)
            ax.plot(data[:50,0], data[:50,2], color = colors[i], label = labels[i])
            
        if experiment == 'var':
            ax.set_ylabel(r'$P_{surv/res}(N_{0})$', fontsize = 15)
            ax.set_xlabel(r'$N_{0}$', fontsize = 15)

        elif experiment =='mut':
            ax.set_ylabel(r'$P_{res}(q_{mut})$', fontsize = 15)
            ax.set_xlabel(r'$q_{mut}$', fontsize = 15)
            ax.set_xscale('log')
        else:
            print('Experiment not known.')
        
    if legend == True:
        legend = ax.legend(fontsize = 13)
        legend.get_frame().set_alpha(0.0)
        
        
    if save == True:
        plt.tight_layout()
        plt.savefig(name, transparent = True)

def compare_single_phenotype_colonies():
       
    path = 'path/to/StochasticColony/Data'
    #TODO: Define experiment (time or size dependency)
    experiment = 'time_dependency/'
    
    #TODO: Set all this to your wishes
    file_names = ['file/with/data.txt', 'second/file/with/data.txt']
    labels = [r'$wt^{static}$', r'$tol^{static}$']
    colors = [wt_static_c, tol_static_c]
    
    files = [os.path.join(path, experiment, f) for f in file_names]
    
    # TODO: choose options of the function
    show_mic = True
    save = False
    name = 'path/tosave/rescue_vs_surv.pdf'
    show_mic = True  # Set to True if you want to show the effective fitness
    save = False  # Set to True if you want to save the figure

    visualize_surv_prob(experiment, files, colors, labels, name, show_mic = show_mic, save = save)

def compare_mixed_populations():
    path = 'path/to/MixedColony/'
    # choose the right experiment to visualize
    experiment = 'time_dependency/'
    
    #TODO: Set all this to your wishes
    file_names = ['file/with/data.txt', 'second/file/with/data.txt']
    labels = ['mutation static', 'mutation cidal']
    colors = [static_color, cidal_color]

    files = [os.path.join(path, experiment, f) for f in file_names]

    name = 'path/tosave/rescue_vs_surv.pdf'
    save = False  # Set to True if you want to save the figure
    visualize_surv_prob(experiment, files, colors, labels, name, save = save)


def compare_simulation_and_effitness():
    path = 'path/to/StochasticColony/Data'
    file_names = ['file/with/data.txt', 'second/file/with/data.txt']
    files = [os.path.join(path, f) for f in file_names]
    labels = [r'$\langle N_{stat}(t) \rangle$', r'$\langle N_{cid}(t) \rangle$']
    colors = [static_color, cidal_color]
    
    name = 'path/tosave/rescue_vs_surv.pdf'
    show_effitness = True  # Set to True if you want to show the effective fitness
    save = False  # Set to True if you want to save the figure
    visualize_colony_size(files, colors, labels, name, show_effitness=show_effitness, save = save)


def compare_rescue_and_survival():
    drug = 'static'  # or 'cidal'
    path_res = 'path/to/MixedColony/data/'
    path_surv = 'path/to/StochasticColony/data/'

    experiment = 'Rescue_vs_Survival/'
    files = [path_res +'data.txt',
                path_surv +'data.txt'
            ]
    
    labels = [r'rescue prob.', r'survival prob.']
    
    if drug =='static':
        colors = [static_color, static_color]
    elif drug == 'cidal':
        colors = [cidal_color, cidal_color]
    
    #TODO: set your preferences
    name = 'path/tosave/rescue_vs_surv.pdf'
    legend = True
    save = False
    visualize_rescue_and_survival(files, colors, labels, experiment, name, legend = legend, save = save)

        
#visualize_landscape(path + 'surv_prob_landscape/080525_N_100_effitness_params_biostatic/')
def visualize_p0_N():
    path = 'path/to/stochastic_colony/'
    experiment = 'size_dependency/'
    
    file_names = ['file/with/data.txt', 'second/file/with/data.txt']
    labels =['first', 'second']
    
    files = [os.path.join(path, experiment, f) for f in file_names]
    colors = [static_color, cidal_color]

    # choose your preferences
    name ='path/for/saving/p0_N.pdf'
    save = False

    visualize_surv_prob(experiment, files, colors, labels, name, save = save)

# Example usage
if __name__ == '__main__':
    compare_rescue_and_survival()
    
