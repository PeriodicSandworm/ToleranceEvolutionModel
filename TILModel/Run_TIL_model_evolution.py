# -*- coding: utf-8 -*- 
# author: Jonas Guenzl
# date: 2023-10-30
import numpy as np
import matplotlib.pyplot as plt
import os
import TIL_model_utils as TIL
from effective_fitness_functions import effective_fitness_hill, effective_fitness_step
from matplotlib.colors import TwoSlopeNorm


fig_length = plt.figure()
ax_length = fig_length.add_subplot()

#Choose which experiments to do and to save:
plot_length = True
save_length = True
plot_path = True
save_path = True

# set path for saving results
path = '/Users/jonas/Documents/tests'  # specify the path to save results
if not os.path.exists(path):
    os.mkdir(path)     # create new folder for each analysis

# set DRC and landscape parameters
L = 4
C = 4
nu =4 
alphatau= 2.4

# set limits of uniform interval
setups = 51
growth_min = 0.5
growth_max = 0.8
width = growth_max-growth_min
death_min = np.linspace(0.01, growth_min, setups)
death_max = (death_min + width)

# compute the trade-off fraction 
trade_off = (death_min + death_max) / (growth_max + growth_min)

# choose the DRC Type
DRC_type = 'hill'  # options: 'hill' or 'step'

# set analysis parameters
repeat_landscape = 20
generations = 500

#set color for plots
color = 'red'

path_length = np.zeros_like(death_min, dtype = float)

for i in range(0,len(death_min)):
    average_path_length = 0

    for ctr in range(repeat_landscape):
        # init landscape
        land = TIL.Full_Landscape(L)
        land.set_phenotypes(np.random.uniform(growth_min, growth_max, L), np.random.uniform(death_min[i], death_max[i], L), C, nu, alphatau, DRC_type)

        # compute properties of the landscape
        land.compute_adjecency_matrix()
        land.get_peaks()
        land.get_fitness_graph(draw =False)

        land.run_evolution(generations)
        average_path_length += (len(land.genetic_path)-1)/repeat_landscape #-1 essential since wt is part of land.gen_path

    path_length[i] = average_path_length
    # create a file for saving the results 
    file = open(os.path.join(path, 'results_'+str(i)+'.txt') , 'w')

    # save Metadata
    file.write("Metadata:\n")
    file.write("Number of Loci: " + str(L) + r", (C, $\nu$, $\alpha\,\tau$) = " + str(tuple((C,nu,alphatau))) + "\n")
    file.write(r"$\lambda_{min} \in$ " + str(tuple((death_min, death_max))) +"\n")
    file.write(r"$\lambda_{max} \in$ " + str(tuple((growth_min, growth_max))) +"\n")

    # save Results
    file.write("\n Results:\n")
    file.write("<l> = " + str(average_path_length) + '\n')

if plot_length == True:
    
    ax_length.scatter(trade_off, path_length, s = 10, color = color)
    ax_length.set_xlabel(r'$\bar{\phi}$', fontsize =15)
    ax_length.set_ylabel(r'$\langle l\rangle$', fontsize =15)
    ax_length.set_xticks(trade_off[::10])
    ax_length.set_xticklabels([f'{tick:.2f}' for tick in ax_length.get_xticks()])

    if save_length == True:
        plt.savefig(path+'/path_length.pdf', transparent = True)
    
    
    
if plot_path == True:
    fig_ev =plt.figure()
    ax_ev = fig_ev.add_subplot()
    land.show_genetic_path(save = save_path, path = path, name = '/gen_path.pdf')
    fig_ev_land = plt.figure()
    ax_ev_land = fig_ev_land.add_subplot()
    # TODO: add growth_array and death_array to the function call
    growth_array = np.linspace(0.1, 1, 200)
    death_array = np.linspace(0.1, 1, 200)
    aspect = 'auto'  # or 'equal' depending on your preference
    land.show_genetic_path_in_landscape(ax_ev_land, growth_array, death_array, C, nu, alphatau, aspect, color_peaks = True, save = save_path, path = path, name = '/gen_path_land.pdf', DRC_type= DRC_type)
