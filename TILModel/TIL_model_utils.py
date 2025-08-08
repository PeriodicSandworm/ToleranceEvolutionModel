# -*- coding: utf-8 -*- 
# author: Jonas Guenzl
# date: 2023-10-30

import numpy as np
import matplotlib.pyplot as plt
from itertools import chain, combinations
from effective_fitness_functions import effective_fitness_hill
from effective_fitness_functions import effective_fitness_step
from effective_fitness_functions import fitness_landscape
from scipy.spatial.distance import cdist
import networkx as nx
import itertools


def get_node_pos(loci, all_genotypes, node):
    sum_genotypes = np.sum(all_genotypes, axis = 1)
    ones = np.sum(node)
    all_such_binaries = all_genotypes[sum_genotypes == ones]
    l = len(all_such_binaries)
    position = np.where((all_such_binaries == node).all(axis=1))[0]
    
    x_pos = position[0] - l/2
    y_pos = ones
    return (x_pos, y_pos)

def powerset(indexset_genotype):
    s = list(indexset_genotype)
    powerset = chain.from_iterable(np.array(list(combinations(s, r))) for r in range(len(s)+1))
    return(powerset)

def convert_index_set_to_genotype(index_set, L):
    genotype = np.zeros(L)
    for index in index_set:
        genotype[index] = 1
    return(genotype)
    
def compute_subset(genotype, L):
    """This function takes an arbitrary genotype and computes the Subset of that genotype in the 
    index representation."""
    index_set = np.where(genotype == 1)[0]
    subset_index = powerset(index_set)
    
    subset = [convert_index_set_to_genotype(np.int64(entry), L) for entry in subset_index]
    return(subset)

def compute_superset(genotype, L):
    """This function takes an arbitrary genotype and computes the superset of that genotype in the 
    index representation."""
    index_set = np.where(genotype == 0)[0]
    superset_index = powerset(index_set)
    superset_index = [np.concatenate((entry, np.where(genotype == 1)[0][:])) for entry in superset_index]
    
    superset = [convert_index_set_to_genotype(np.int64(entry), L) for entry in superset_index]
    return(superset)

class phenotype:    
    def __init__(self, growth, death, C, nu, alphatau, DRC_type = 'hill'):
        self.g = growth
        self.d = death
        self.C = C
        self.nu = nu
        self.a = alphatau
        if DRC_type == 'hill':
            self.fit = effective_fitness_hill(growth, death, C, nu, alphatau)
        elif DRC_type == 'step':
            self.fit = effective_fitness_step(growth, death, C, alphatau)
    
    def set_genotype(self, genotype):
        self.gen = genotype
        
        
class Full_Landscape:
    '''This object is now a ful fitness landscape, where we can run analysis' on. ''' 
    def __init__(self, L):
        """First init a Landscape with L-loci-biallelic genotypes."""
        self.loci = L
        self.genotypes = np.array(list(itertools.product([0,1],repeat = L)))
        
    def set_phenotypes(self, growth_i, death_i, C, nu, alphatau, DRC_type = 'hill'):
        """Set an Array of phenotype objects that live in this landscape."""
        boolean = np.array(self.genotypes, dtype="bool")
        self.phenotypes = np.array([phenotype(np.exp(np.sum(np.log(growth_i[boolean[i]]))), np.exp(np.sum(np.log(death_i[boolean[i]]))), C, nu, alphatau, DRC_type) for i in range(len(self.genotypes))])
    
    def get_index(self, genotype):
        """This function simply returns the index position of a test genotype in the list of all genotypes"""
        index = np.where((self.genotypes == genotype).all(axis=1))[0]
        return(index[0])
    
    def get_genotype(self, index):
        """This function finds the genotype at a certain position"""
        genotype = self.genotypes[index]
        return(genotype)
        
    def compute_adjecency_matrix(self):
        """This Function computes the adjecency matrix of the fitness Landscape. Adjacency matrix will be stored
        as self.adjacency."""
        dist_matrix = cdist(self.genotypes, self.genotypes)
        fitness = np.array([self.phenotypes[i].fit for i in range(len(self.phenotypes))])
        direction_matrix = np.sign(fitness[None, :] - fitness[:, None])
    
        adj_matrix = dist_matrix*direction_matrix
        adj_matrix[adj_matrix != 1] = 0
        
        self.adjacency = adj_matrix.astype(int)
        
        del(fitness, direction_matrix, dist_matrix)
        
    def get_peaks(self):
        '''This function computes and stores the index position of all the fitness peaks in the landscape.'''
        try:
            self.peaks = np.where(np.sum(self.adjacency, axis =1) == 0)[0]
        except AttributeError:
            print("Adjacency matrix needs to be computed first. Please run obj.compute_adjecency_matrix() first.")
        
    def get_fitness_graph(self, draw = False, color_peaks = True, save = False, path = None, name = None):
        """This function creates a fitness graph of the landscape. """
        try:
            self.Graph = nx.from_numpy_array(self.adjacency, create_using=nx.DiGraph)
        except AttributeError:
            print("Adjacency matrix needs to be computed first. Please run obj.compute_adjecency_matrix() first.")
        
        labels = [''.join(row.astype(str)) for row in self.genotypes]
        node_labels = {i: labels[i] for i in range(len(labels))}

        node_positions = {}
        for i in range(len(self.genotypes)):
            node_positions[i] = get_node_pos(self.loci, self.genotypes, self.genotypes[i])
        
        if draw == True:
            if color_peaks == True:
                try:
                    for j in self.peaks:
                        self.Graph.nodes[j]['color'] = 'red'
                except AttributeError:
                    print("Landscape peaks need to be computed first. Please run obj.get_peaks() first.")

            node_colors = [self.Graph.nodes[node].get('color', 'lightgrey') for node in self.Graph.nodes()]
            
            if save == True:
                fig = plt.figure()
                nx.draw(self.Graph, ax=fig.add_subplot(), arrows = True, pos=node_positions, node_color=node_colors, node_size=2000, labels = node_labels, font_size = 15)
                plt.tight_layout()
                fig.savefig(path+name, transparent = True)
            
            else:
                nx.draw(self.Graph, arrows = True, pos=node_positions, node_color=node_colors, node_size=2000, labels = node_labels)
                
    
                
            
    def run_evolution(self, generations, random_start = False):
        """This function runs an evolution of some start genotype for a certain number of generations. """
        
        if random_start == True:
            start = np.random.choice(self.genotypes)
        else:
            start = np.zeros(self.loci)
            
        active_gtype = start
        genetic_path = np.array([active_gtype])
        
        g_index = self.get_index(active_gtype)
        neighbours = self.genotypes[np.array(self.adjacency[g_index], dtype = 'bool')]

        
        for j in range(generations):
            
            
            if len(neighbours) == 0:
                break
                
            else:
                test_gtype = neighbours[np.random.choice(np.arange(len(neighbours)))]
                test_index = self.get_index(test_gtype)
    
                if self.phenotypes[test_index].fit > self.phenotypes[g_index].fit:
            
                    active_gtype = np.copy(test_gtype)                # set the new genotype
                    g_index = self.get_index(active_gtype)   # set the new neighbours
                    
                    neighbours = self.genotypes[np.array(self.adjacency[g_index], dtype = 'bool')]
                    genetic_path = np.vstack((genetic_path, test_gtype))

                else:
                    neighbours = np.delete(neighbours, (neighbours ==  test_gtype).all(axis=1))
                            
        self.genetic_path = genetic_path
        
    def show_genetic_path(self, save = False, path =None, name =None):
        self.get_fitness_graph()
        labels = [''.join(row.astype(str)) for row in self.genotypes]
        node_labels = {i: labels[i] for i in range(len(labels))}

        node_positions = {}
        for i in range(len(self.genotypes)):
            node_positions[i] = get_node_pos(self.loci, self.genotypes, self.genotypes[i])
            
        for j in range(len(self.genetic_path)):
            index_1 = self.get_index(self.genetic_path[j])
            self.Graph.nodes[index_1]['color'] = 'purple'
            
            if j < len(self.genetic_path)-1:
                index_2 = self.get_index(self.genetic_path[j+1])
                self.Graph.edges[index_1, index_2]['color'] = 'purple'
                
            try:
                for j in self.peaks:
                        self.Graph.nodes[j]['color'] = 'red'
            except AttributeError:
                print("Landscape peaks need to be computed first. Please run obj.get_peaks() first.")
                
            
            
        node_colors = [self.Graph.nodes[node].get('color', 'lightgrey') for node in self.Graph.nodes()]
        edge_colors = [self.Graph.edges[edge].get('color', 'black') for edge in self.Graph.edges()]
        
        nx.draw(self.Graph, arrows = True, pos=node_positions, node_color=node_colors, edge_color=edge_colors, node_size=1000, labels = node_labels)
        
        if save == True:
            plt.savefig(path+name, transparent = True)
    
    def show_genetic_path_in_landscape(self, ax, growth_array, death_array, C, nu, alphatau, aspect, color_peaks = False, save = False, path = None, name = None, DRC_type ='hill'):
        #compute the smooth fitness landscape 
        fitness_landscape(ax, growth_array, death_array, C, alphatau, aspect, nu,DRC_type)

            
        get_g = np.vectorize(lambda obj: obj.g)
        get_d = np.vectorize(lambda obj: obj.d)
        g_values = get_g(self.phenotypes)
        ax.scatter(get_g(self.phenotypes), get_d(self.phenotypes), color = 'grey')
        
        
        path_index = np.zeros(len(self.genetic_path), dtype = int)
        for j in range(len(self.genetic_path)):
            path_index[j] = self.get_index(self.genetic_path[j])
        print(path_index)    
        ax.scatter(get_g(self.phenotypes[path_index]), get_d(self.phenotypes[path_index]), color = 'purple')
        ax.plot(get_g(self.phenotypes[path_index]), get_d(self.phenotypes[path_index]), color = 'purple')
        
        if color_peaks == True:
            ax.scatter(get_g(self.phenotypes[self.peaks]), get_d(self.phenotypes[self.peaks]), color = 'red')
            
        plt.tight_layout()
        if save == True:
            plt.savefig(path+name, transparent = True)
            
    def test_AP(self):
        """ This function tests if any fitness peak is acessible from all its sub-/superset."""
        try:
            peak_genotypes = [self.get_genotype(self.peaks[i]) for i in range(len(self.peaks))]
        except AttributeError:
            print("Landscape peaks need to be computed first. Please run obj.get_peaks() first.")
            
        # compute the sub-/superset genotypes first
        # this returns a list of all subset genotypes
        for geno in peak_genotypes:
            subset = compute_subset(geno, self.loci)
            superset = compute_superset(geno, self.loci)
        
        # find the graph index of evry entry in sub-/superset
        subset_ind = np.array([self.get_index(geno) for geno in subset])
        superset_ind = np.array([self.get_index(geno) for geno in superset])        
        
        # start for the subset 
        for i, node in enumerate(subset_ind):
            # compute index of subset neighbours
            # take the upper driangular to only allow for increasing number of mutations (00 -> 01) and not (01 ->00)
            neighbour_matrix = np.triu(cdist(subset, subset))  
            neighbour_matrix[neighbour_matrix != 1] = 0          #set everythin else than 1 to 0 
            neighbours = np.where(neighbour_matrix[i] == 1)[0]   # now find index of the neighbours in the subset
            
            for neighbour_node in subset_ind[neighbours]:
                #print(self.Graph.has_edge(node, neighbour_node))
                if self.Graph.has_edge(node, neighbour_node) == False:
                    print("AP not fulfilled for edge " + str(self.get_genotype(node)) + " to " + str(self.get_genotype(neighbour_node)))
                    return(False)    # return False since the AP is not fulfilled
        
        
        # now do the same for the superset 
        for i, node in enumerate(superset_ind):
            # compute index of subset neighbours
            # take the lower triangular to only allow for decreasing number of mutations (11 -> 01) and not (01 ->11)
            neighbour_matrix = np.tril(cdist(superset, superset))  
            neighbour_matrix[neighbour_matrix != 1] = 0          #set everythin else than 1 to 0 
            neighbours = np.where(neighbour_matrix[i] == 1)[0]   # now find index of the neighbours in the subset
            
            for neighbour_node in superset_ind[neighbours]:
                if self.Graph.has_edge(node, neighbour_node) == False:
                    print("AP not fulfilled for edge " + str(self.get_genotype(node)) + " to " + str(self.get_genotype(neighbour_node)))
                    return(False)    # return False since the AP is not fulfilled
        
        return(True)   # Function returns True only if the Peak is accessible from all direct paths        
                    
        
      
    
    def smooth_landscape(self, ax, growth_array, death_array, C, nu, alphatau, aspect, death_mean, growth_mean, show_genotypes = False, color_peaks = False, save = False, path = None, name = None, show_mean = False, DRC_type ='hill'):       
        #compute the smooth fitness landscape 
        fitness_landscape(ax, growth_array, death_array, C, nu, alphatau, aspect, DRC_type)
            
        if show_genotypes == True:
            get_g = np.vectorize(lambda obj: obj.g)
            get_d = np.vectorize(lambda obj: obj.d)
            g_values = get_g(land.phenotypes)
            ax.scatter(get_g(self.phenotypes), get_d(self.phenotypes), color = 'grey')
            
            if color_peaks == True:
                ax.scatter(get_g(self.phenotypes[self.peaks]), get_d(self.phenotypes[self.peaks]), color = 'red')
            
            if show_mean == True:
                #This plots the curve of the mean of both distributions
                ax.plot(growth_mean ** np.arange(0,self.loci +1), death_mean ** np.arange(0,self.loci +1) , color = 'black', marker ='.', ls = ":", alpha = 0.7)
            #ax.set_yscale('log') 
            #ax.set_xscale('log')  
            
            plt.tight_layout()
            if save == True:
                plt.savefig(path+name, transparent = True)
        
        