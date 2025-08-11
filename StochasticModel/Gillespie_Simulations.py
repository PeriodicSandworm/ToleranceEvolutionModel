#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 08:43:12 2025

@author: jguenzl
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from Stochastic_utils import compute_surv_full_period

def stochastic_DRC_step(t, period, t_mic, eta):
    '''This function parametrizes the step DRC for the stochastic model 
    '''
    if t % period < t_mic:
        return(eta)
    elif t % period >= t_mic:
        return(0)

class stochastic_population:
    '''Class to simulate a general population of bacteria with Gillespie algorithm.'''
    def __init__(self, N, K, g, d, eta, q):
        '''Initialize the population with initial size N, carrying capacity K, growth rate g, death rate d, antibiotic effect eta, and mutation probability q. 
        Except for K (carrying capacity, integer) and q (mutation probability, np array length N-1), all parameters are numpy arrays of the same length as N.'''
        self.N = np.copy(N)
        self.K = K
        self.g = np.copy(g)
        self.d = np.copy(d)
        self.eta = np.copy(eta)
        self.q = np.copy(q)

    def gillespie_event(self, grow, die, t, period, t_mic):
        '''Perform a single Gillespie event, returning the time of the respective event.'''
        # Calculate the birth, death, and mutation rates
        birth = np.copy(grow)
        birth[:-1] *= (1 - self.q)
        mutation = np.copy(grow[:-1]) * self.q
        death = die

        # Combine the rates into a single array
        rates = np.concatenate((birth, death, mutation))
        k_tot = np.sum(rates[rates>=1e-8])      # set all very small rates to zero to avoid numerical issues

        # If the total rate is zero, return infinity to indicate no event
        if k_tot <= 0:
            return np.inf
        
        # Draw the time until the next event
        dt = np.random.exponential(1 / k_tot)
        
        # Some technical quantities
        t_mod = t % period
        t_next_mod = t_mod + dt

        # Check if the next event occurs before a change in the environment (i.e. MIC or period end)
        if (t_mod < t_mic and t_next_mod <= t_mic + 1e-6) or (t_mod >= t_mic and t_next_mod <= period + 1e-6):
            # Choose the event based on the rates
            probs = rates / k_tot
            event = np.random.choice(len(rates), p=probs)
            n = len(self.N)
            # Update the population size based on the event type
            if event < n:
                self.N[event] += 1
            elif event < 2 * n:
                self.N[event % n] -= 1
            else:
                self.N[event % n + 1] += 1
            return dt
        # If the next event occurs after a change in the environment, set the time to the MIC or period end
        elif t_mod < t_mic and t_next_mod > t_mic:
            return t_mic - t_mod
        elif t_mod >= t_mic and t_next_mod > period:
            return period - t_mod

    def run_gillespie(self, period, t_mic, treatment_duration, drug_type):
        '''Run the Gillespie simulation for a given period and treatment duration, returning the times and colony sizes.'''
        # Initialize time and colony size lists
        t = 0.0
        times = [0.0]
        colonies = [self.N.copy()]
        # Run the simulation until the end of treatment is reached
        while t < treatment_duration:
            # Check if the population size is zero or far from extinction
            # This is a simple check to avoid running the simulation trouble 
            size = np.sum(self.N)
            if size == 0 or size >= 1e4:
                break
            
            # Calculate the growth and death rates based on the region in the pharmacokinetic profile (above of below MIC), the drug type and the current population size
            alpha = stochastic_DRC_step(t, period, t_mic, self.eta)

            if drug_type == 'biostatic':
                growth_term = (self.g - alpha) * (1 - size / self.K)
                growth_term = np.where(np.abs(growth_term) < 1e-10, 0, growth_term) # avoid numerical issues
                grow = growth_term * self.N
                die = self.d * self.N
            elif drug_type == 'biocidal':
                grow = self.g * (1 - size / self.K) * self.N
                die = (self.d + alpha) * self.N
            else:
                raise ValueError("Unsupported drug type.")
            # Perform a Gillespie event and update the time
            dt = self.gillespie_event(grow, die, t, period, t_mic)

            if t + dt > treatment_duration:
                break

            t += dt
            # save everything in a list
            times.append(t)
            colonies.append(self.N.copy())

        return np.array(times), np.array(colonies)
    
def num_survival(N, g, d, eta, q, K, period, treatment_duration, t_mic, repeat = 1000, drug_type = 'biostatic'):
    '''This function computes the survival probability and the size after treatment of a stochastic population numerically from repeated simulations.'''
    
    survived = 0
    size = np.empty(0, dtype = 'object')

    for j in range(0, repeat):
        # Initialize the population with initial size N, carrying capacity K, growth rate g, death rate d, antibiotic effect eta, and mutation probability q
        pop = stochastic_population(N, K, g, d, eta, q)
        # svae the last entry of the population size
        population = np.copy(pop.run_gillespie(period, t_mic, treatment_duration, drug_type = drug_type)[1])

        # Check if the population survived the treatment and at which size
        # first the special case that nothing happened
        if (population == N).all():
            survived += 1
            size = np.append(size, population)
        # then the case that the population size is zero
        elif np.sum(population[-1]) < 1:
            pass
        # otherwise the population survived and we save the last entry of the population size
        else:
            survived += 1
            size = np.append(size, population[-1])  
    
    # Calculate the survival probability and the average population size and return these
    surv = survived/repeat
    size = np.reshape(size, (np.int64(len(size)/len(N)), len(N)))

    population_size = np.sum(size, axis = 0)/repeat
    return(surv, population_size)


def survival_analysis(path, name, N0, growth, death, eta, q, drug, datapoints, mode="time", time_range = None, N0_range=None, frequency_range=None, q_range=None, show_results =False):
    """
    Compute survival probability as a function of time, initial population size N0, standing variation frequency, or mutation probability q.
    mode: "time", "N0", "standing variation", or "mutation"
    time_range: array-like, values of time to test if mode == "time"
    N0_range: array-like, values of N0 to test if mode == "N0"
    frequency_range: array-like, values of standing variation frequency to test if mode == "standing variation"
    q_range: array-like, values of q to test if mode == "mutation"
    """
    repeat = int(2000)

    # Set the carrying capacity
    K = 1e7
    # Set Pharmacokinetics
    period = 6
    C = 20
    alpha = 0.7
    t_mic = np.round(np.log(C)/alpha, decimals = 4)
    end = 300

    if mode == "time":
        if time_range is None:
            time_range = np.linspace(0, end, datapoints)
        xvals = time_range
        xlabel = r'time t'
        xvar = "end"

    elif mode == "N0":
        if N0_range is None:
            N0_range = np.linspace(1, 99, datapoints, dtype=int)
        xvals = N0_range
        xlabel = r'Initial population size $N_0$'
        xvar = "N0"
    
    elif mode == "standing variation":
        if frequency_range is None:
            frequency_range = np.linspace(0, 0.5, datapoints)
        xvals = frequency_range
        xlabel = r'Standing variation frequency $f$'
        xvar = "frequency"


    elif mode == "mutation":
        if q_range is None:
            q_range = np.array([np.array([10**w]) for w in np.linspace(-4, -1, datapoints)])
        xvals = q_range
        xlabel = r'Mutation probability $q$'
        xvar = "q"
    else:
        raise ValueError("mode must be 'time', 'N0', 'standing variationor 'q'")

    num_surv = np.zeros(len(xvals), dtype='float')
    num_size = np.zeros((len(xvals), len(N0)), dtype='float')
    theo_surv = np.zeros(len(xvals), dtype='float')

    for i, x in enumerate(xvals):
        if mode == "time":
            num_surv[i], num_size[i] = num_survival(N0, growth, death, eta, q, K, period, x, t_mic, repeat=repeat, drug_type=drug)
            if np.all(q == 0) and x <= period:
                theo_surv[i] = compute_surv_full_period(x, t_mic, N0[0], growth[0], death[0], eta[0], drug_type=drug)

        elif mode == "N0":
            # we assume here, that we start with an initially homogeneous wt population and only compute one treatment period
            N0_array = np.zeros_like(growth, dtype=int)
            N0_array[0] =x
            num_surv[i], num_size[i] = num_survival(N0_array, growth, death, eta, q, K, period, period, t_mic, repeat=repeat, drug_type=drug)
            if np.all(q == 0):
                theo_surv[i] = compute_surv_full_period(period, t_mic, x, growth[0], death[0], eta[0], drug_type=drug)

        elif mode == "standing variation": 
            # the default case assumes a total population size of N =100 and a two phenotype population with a given standing variation frequency of the tolerant mutant
            N0 = np.zeros_like(growth, dtype=int)
            N0[0] = int(100 * (1 - x))  # wildtype
            N0[1] = int(100 * x)        # tolerant mutant
            num_surv[i], num_size[i] = num_survival(N0, growth, death, eta, q, K, period, period, t_mic, repeat=repeat, drug_type=drug)

        elif mode == "mutation":
            num_surv[i], num_size[i] = num_survival(N0, growth, death, eta, x, K, period, end, t_mic, repeat=repeat, drug_type=drug)
            

    # Save results
    if not os.path.exists(path + mode + '/' + name):
        os.makedirs(path + mode + '/' + name)

    file = open(os.path.join(path, mode, name, 'metadata.txt'), 'w')
    file.write("Metadata:\n")
    file.write("Drug Type: " + drug + "\n")
    file.write("C = " + str(C) + "\n")
    file.write("alpha = " + str(alpha) + "\n")
    file.write("t_mic = " + str(t_mic) + "\n")
    file.write("period = " + str(period) + "\n")
    file.write("K = " + str(K) + "\n")
    file.write("N0 = " + str(N0) + "\n")
    file.write("g = " + str(growth) + "\n")
    file.write("d = " + str(death) + "\n")
    file.write("eta = " + str(eta) + "\n")
    file.write("q = " + str(q) + "\n")
    file.close()

    # Build header
    num_size_columns = num_size.shape[1]
    header = [xvar]
    header += [f"N(end)_{i+1}" for i in range(num_size_columns)]
    header += ["p0"]
    if theo_surv is not None and np.any(theo_surv):
        header += ["p0_theo"]
        header_str = ",".join(header)
        results = np.column_stack((xvals, num_size, num_surv, theo_surv))
    else:
        header_str = ",".join(header)
        results = np.column_stack((xvals, num_size, num_surv))
    np.savetxt(path + mode + '/' + name + 'data.txt', results, fmt="%.2e", delimiter=',', header=header_str)

    # Visualising
    if show_results == True:
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.scatter(xvals, num_surv, marker='.', s=5, c='#1f77b4')
        if theo_surv is not None and np.any(theo_surv):
            ax.plot(xvals, theo_surv, color='red')
        ax.set_xlabel(xlabel, fontsize=15)
        ax.set_ylabel(r'Survival probability $p_{surv}$', fontsize=15)
        fig.savefig(path + mode + '/' + name + 'surv_prob.pdf', transparent=True)

#exemplary usage
if __name__ == "__main__":
    # Define parameters
    path = "path/to/your/directory/"
    name = 'example/'
    N0 = np.array([100, 0])  # Initial population size
    growth = np.array([0.7, 0.7])  # Growth rate
    death = np.array([0.4, 0.3])  # Death rate
    eta = np.array([0.7, 0.4])  # Antibiotic effect
    q = np.array([0.0])  # Mutation probability
    drug = 'biostatic'  # Drug type
    datapoints = 2  # Number of data points to compute

    # Run survival analysis for different modes
    survival_analysis(path, name, N0, growth, death, eta, q, drug, datapoints, mode="mutation", show_results=True)
