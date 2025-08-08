# -*- coding: utf-8 -*-
# author: Jonas Guenzl
# date: 2023-10-30

'''Run_TIL_analysis.py
This script runs an analysis of the TIL model by generating fitness landscapes with varying numbers of loci
and evaluating the number of fitness peaks and the accessibility property (AP).
It saves the results in a specified directory and generates plots to visualize the findings.'''

import numpy as np
import matplotlib.pyplot as plt
import os
import TIL_model_utils as TIL
from effective_fitness_functions import effective_fitness_hill, effective_fitness_step
from scipy.special import comb

fig_peaks = plt.figure()
ax_peaks = fig_peaks.add_subplot()

fig_AP = plt.figure()
ax_AP = fig_AP.add_subplot()

# set path for saving results
path = '/path/to/save/results'  # specify the path to save results
if not os.path.exists(path):
    os.mkdir(path)     # create new folder for each analysis

# choose the DRC Type
DRC_type = 'hill'  # options: 'hill' or 'step'


# set PKPD parameters
C = 4
nu = 4 #only relevant for hill type DRC
alphatau = 2.4

# set limits of uniform interval from which mutations on lmax and lmin are chosen
growth_min = 0.5
growth_max = 0.8
death_min = 0.3
death_max = 0.6

# set analysis parameters
repeat_landscape = 1                     # number of landscapes to be analysed
number_of_loci = np.arange(4,11, step =1)   # number of loci per genotype
normalize_peaks = True                       # normalize peaks by the number of possible peaks in the landscape

#set color for plots
color = 'red'

# initialize array for saving results
mean_number_of_peaks = np.zeros_like(number_of_loci, dtype = float)
mean_of_AP_fulfill = np.zeros_like(number_of_loci, dtype = float)
rel_number_of_peaks = np.zeros_like(number_of_loci, dtype = float)
peak_normalization = np.zeros_like(number_of_loci, dtype = float)

for l, loci in enumerate(number_of_loci):
    # set trackers 
    AP_fulfilled = 0
    abs_number_of_peaks = 0
    peak_layer = np.empty(0)
    
    for ctr in range(repeat_landscape):
        # init landscape
        land = TIL.Full_Landscape(loci)
        land.set_phenotypes(np.random.uniform(growth_min, growth_max, loci), np.random.uniform(death_min, death_max, loci), C, nu, alphatau, DRC_type = DRC_type)

        # compute properties of the landscape
        land.compute_adjecency_matrix()
        land.get_peaks()
        land.get_fitness_graph(draw =False)

        # evaluate_landscape AP
        test = land.test_AP()
        if test == True:
            AP_fulfilled += 1
        
        # evaluate landscape Peaks
        abs_number_of_peaks += len(land.peaks)
        
        # evaluate landscape relative peaks
        if normalize_peaks == True:
            peak_layer = np.append(peak_layer, np.sum(land.get_genotype(land.peaks), axis =1))

    # create a file for saving the results 
    file = open(os.path.join(path, 'results_'+str(loci)+'_loci.txt') , 'w')

    # save Metadata
    file.write("Metadata:\n")
    file.write("Number of Loci: " + str(loci) + r", (C, $\nu$, $\alpha\,\tau$) = " + str(tuple((C,nu,alphatau))) + "\n")
    file.write(r"$\lambda_{min} \in$ " + str(tuple((death_min, death_max))) +"\n")
    file.write(r"$\lambda_{max} \in$ " + str(tuple((growth_min, growth_max))) +"\n")

    # save Results
    file.write("\n Results:\n")
    file.write("<peaks> = " + str(abs_number_of_peaks / repeat_landscape) + '\n')
    file.write("AP fulfilled in " + str(100 * AP_fulfilled/repeat_landscape) + "% \n")
    
    # save in array for visualisation
    mean_number_of_peaks[l] = abs_number_of_peaks/repeat_landscape
    mean_of_AP_fulfill[l] = AP_fulfilled/repeat_landscape
    
    
    if normalize_peaks == True:
        counts, bins = np.histogram(peak_layer, bins=np.arange(0, loci + 2) - 0.5)
        try: 
            int(np.sum(counts)) == int(abs_number_of_peaks)
            
        except:
            print("Sum of counts and number of peaks does not match")
        # compute empirical probability mass function
        pmf = counts / np.sum(counts)
        # compute the normalization factor for the relative number of peaks
        layer_nodes = np.array([comb(loci, n) for n in range(loci + 1)])
        peak_normalization[l] = np.sum(pmf*layer_nodes)
        rel_number_of_peaks[l] = mean_number_of_peaks[l]/peak_normalization[l]
        file.write("Relative number of peaks " + str(rel_number_of_peaks[l]))
        
# make plots of the data
ax_peaks.scatter(number_of_loci, mean_number_of_peaks, s = 10, color = color, label = 'absolute count')

    
if normalize_peaks == True:
    ax_peaks.scatter(number_of_loci, rel_number_of_peaks, s=10, color = 'black', label = 'relative count')

    # uncomment if theory shpould be shown
    #peak_number_theory = mean_number_of_peaks[0]/peak_normalization[0] * peak_normalization
    #ax_peaks.plot(number_of_loci, peak_number_theory, ls = ':', color = 'black')
    
ax_peaks.set_xlabel('loci', fontsize =15)
ax_peaks.set_ylabel('fitness peaks', fontsize =15)
ax_peaks.set_xticks(number_of_loci)
if normalize_peaks == True:
    ax_peaks.legend()
fig_peaks.savefig(path+'/mean_number_of_peaks.pdf', transparent = True)


ax_AP.scatter(number_of_loci, mean_of_AP_fulfill, s = 10, color = color)
ax_AP.set_xlabel('loci', fontsize =15)
ax_AP.set_ylabel('AP [%]', fontsize =15)
ax_AP.set_xticks(number_of_loci)
fig_AP.savefig(path+'/mean_of_AP_fulfill.pdf', transparent = True)

plt.show()