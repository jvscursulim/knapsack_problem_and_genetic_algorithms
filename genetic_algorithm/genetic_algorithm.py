import numpy as np
from gene import Gene
from chromosome import Chromosome
from typing import List, Optional

def create_chromosomes(population_size: int, genes_list: List[Gene]) -> List[Chromosome]:
    """Creates a list of chromosomes.

    Args:
        population_size (int): The size of chromosome population.
        genes_list (List[Gene]): The list of genes of the problem.

    Returns:
        List[Chromosome]: A with the new chromosomes created.
    """
    chromosome_list = [Chromosome(gene_sequence = [np.random.randint(0,2) for _ in range(len(genes_list))]) for _ in range(population_size)]
    
    return chromosome_list

def fitness_assignment(knapsack_max_weight: int, genes_list: List[Gene], chromosome_list: List[Chromosome]) -> None:
    """Determines if the chromosomes are valid solutions.

    Args:
        chromosome_list (List[Chromosome]): A list of chromosomes.
    """
    for chromosome in chromosome_list:
        chromosome.fitness_assigment(max_weight = knapsack_max_weight, genes_list = genes_list)
        
def tournament_selection():
    pass

def spinning_roulette_wheel_selection():
    pass

def cross_over_chromosomes():
    pass

def mutation():
    pass