import numpy as np
import time
from gene import Gene
from chromosome import Chromosome
from typing import List, Optional, Tuple


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

        
def tournament_selection(chromosome_list: List[Chromosome]) -> List[Chromosome]:
    """Selects the chromosomes with higher score.

    Args:
        chromosome_list (List[Chromosome]): A list of chromosomes.

    Returns:
        List[Chromosome]: A list with the tournament winners.
    """
    tournament_winners = []
    
    positions = [*range(len(chromosome_list))]
    pairs_list = []
    
    for _ in range(2):
        
        rng_numbers = np.random.choice(positions, size = 2)
        while rng_numbers[0] == rng_numbers[1]:
            
            rng_numbers = np.random.choice(positions, size = 2)
        
        numbers_tuple = tuple(rng_numbers)
        pairs_list.append(numbers_tuple)
        positions.pop(positions.index(numbers_tuple[0]))
        positions.pop(positions.index(numbers_tuple[1]))
        
    for pair in pairs_list:
        
        index1 = pair[0]
        index2 = pair[1]
        
        if chromosome_list[index1].score > chromosome_list[index2].score:
            
            tournament_winners.append(chromosome_list[index1])
        else:
            
            tournament_winners.append(chromosome_list[index2])
                    
    return tournament_winners
        

def spinning_roulette_wheel_selection(chromosome_list: List[Chromosome]) -> List[Chromosome]:
    """Selects the chromosomes.

    Args:
        chromosome_list (List[Chromosome]): A list of chromosomes.

    Returns:
        List[Chromosome]: A list with chromosomes winners.
    """
    chromosome_weights_list = [chromosome.total_weight for chromosome in chromosome_list]
    weights_sum = sum(chromosome_weights_list)
    probs = [weight/weights_sum for weight in chromosome_weights_list]
    
    srw_winners = []
    
    for _ in range(2):
        
        srw_winners.append(np.random.choice(chromosome_list, size = 1, p = probs)[0])
    
    return srw_winners


def crossover_chromosomes(parent1: Chromosome, parent2: Chromosome, chunck_size: int) -> List[Chromosome]:
    """Combines the genes of two chromosomes to create new chromosomes.

    Args:
        parent1 (Chromosome): First chromosome parent.
        parent2 (Chromosome): Second chromosome parent.
        chunck_size (int): The amount of genes we want to use in the crossover.

    Raises:
        ValueError: If chunck size is greater than the gene sequence length.

    Returns:
        List[Chromosome]: A list with chromosome childrens.
    """
    if chunck_size >= len(parent1.gene_sequence):
        
        raise ValueError("The chunck_size cannot be greater than the length of gene_sequence.")
    else:
        
        childrens = []
        
        gene_cut_init_p1 = parent1.gene_sequence[:chunck_size]
        gene_cut_final_p1 = parent1.gene_sequence[chunck_size:]
        gene_cut_init_p2 = parent2.gene_sequence[:chunck_size]
        gene_cut_final_p2 = parent2.gene_sequence[chunck_size:]
        
        gene_cut_init_p1.extend(gene_cut_final_p2)
        gene_cut_init_p2.extend(gene_cut_final_p1)
        
        childrens.append(Chromosome(gene_sequence = gene_cut_init_p1))
        childrens.append(Chromosome(gene_sequence = gene_cut_init_p2))
        
        return childrens
        

def mutation(chromosome: List[Chromosome], mutation_rate: float = 0.05) -> None:
    """Inserts a mutation in the gene sequence of a chromosome.

    Args:
        chromosome (Chromosome): A chromosome.
        mutation_rate (float, optional): The rate of mutation. Defaults to 0.05.
    """ 
    for i in range(len(chromosome.gene_sequence)):
            
        rng_number = np.random.random()
        
        if rng_number < mutation_rate:
                
            chromosome.gene_sequence[i] = int(not chromosome.gene_sequence[i])
                
            
def genetic_algorithm_tournament_version(epochs: int, knapsack_max_weight: int, stopping_criterion: int, genes_list: List[Gene], population_size: int, init_chromosomes: Optional[List[Chromosome]] = None, crossover_rate: float = 0.95, chunck_size: int = 1, mutation_rate: float = 0.05) -> Tuple[int, float, List]:
    """Creates a genetic algorithm.

    Args:
        epochs (int): The maximum number of the iterations.
        knapsack_max_weight (int): The maximum weight capacity of the knapsack.
        stopping_criterion (int): The stopping criterion of the iteration process. 
        genes_list (List[Gene]): The list of the genes that define the knapsack problem.
        population_size (int): The size of the chromosome population.
        init_chromosomes (Optional[List[Chromosome]], optional): The list with the initial chromosomes. Defaults to None.
        crossover_rate (float, optional): The crossover chromosomes rate. Defaults to 0.95.
        chunck_size (int): . Defaults to 1.
        mutation_rate (float, optional): The mutation rate of a gene. Defaults to 0.05.

    Returns:
        Tuple[int, float, List]: A tuple with the number of epochs to reach the solution, the time spent and a list with the best gene sequence of the chromosomes that represents the solution.
    """
    if init_chromosomes is None:
        
        start = time.time()
        chromosomes_list = create_chromosomes(population_size = population_size, genes_list = genes_list)
              
        for gen in range(epochs):
            
            for chromosome in chromosomes_list:
                
                chromosome.fitness_assignment(max_weight = knapsack_max_weight, genes_list = genes_list)
            
            chromosomes_scores = [chromosome.score for chromosome in chromosomes_list]
            max_score = max(chromosomes_scores)
            max_score_index = chromosomes_scores.index(max_score)
            
            if max_score >= stopping_criterion:
                break
            
            new_chromosomes = []
            
            while len(new_chromosomes) < population_size:
            
                tournament_winners = tournament_selection(chromosome_list = chromosomes_list)
                
                rng_number1 = np.random.random()
                
                if rng_number1 < crossover_rate:
                        
                    childrens = crossover_chromosomes(parent1 = tournament_winners[0], parent2 = tournament_winners[1], chunck_size = chunck_size)
                    
                    for child in childrens:
                        
                        mutation(chromosome = child, mutation_rate = mutation_rate)
                        
                        if population_size%2 == 1:
                            
                            if len(new_chromosomes) == population_size - 1:
                                
                                new_chromosomes.append(child)
                                break
                            else:
                                
                                new_chromosomes.append(child)
                        else:
                            
                            new_chromosomes.append(child)
                else:
                    
                    if len(new_chromosomes)%2 == 1:
                        
                        if len(new_chromosomes) == population_size - 1:
                            
                            new_chromosomes.append(tournament_winners[np.random.randint(0,2)])
                        else:
                            
                            for winner in tournament_winners:
                                new_chromosomes.append(winner)
                    else:
                        
                        for winner in tournament_winners:
                            new_chromosomes.append(winner)
                             
            chromosomes_list = [*new_chromosomes]
        
        end = time.time()
    else:
        
        start = time.time()
        population_size = len(init_chromosomes)
        chromosomes_list = [*init_chromosomes]
              
        for gen in range(epochs):
            
            for chromosome in chromosomes_list:
                
                chromosome.fitness_assignment(max_weight = knapsack_max_weight, genes_list = genes_list)
            
            chromosomes_scores = [chromosome.score for chromosome in chromosomes_list]
            max_score = max(chromosomes_scores)
            max_score_index = chromosomes_scores.index(max_score)
            
            if max_score >= stopping_criterion:
                break
            
            new_chromosomes = []
            
            while len(new_chromosomes) < population_size:
            
                tournament_winners = tournament_selection(chromosome_list = chromosomes_list)
                
                rng_number1 = np.random.random()
                
                if rng_number1 < crossover_rate:
                        
                    childrens = crossover_chromosomes(parent1 = tournament_winners[0], parent2 = tournament_winners[1], chunck_size = chunck_size)
                    
                    for child in childrens:
                        
                        mutation(chromosome = child, mutation_rate = mutation_rate)
                        
                        if population_size%2 == 1:
                            
                            if len(new_chromosomes) == population_size - 1:
                                
                                new_chromosomes.append(child)
                                break
                            else:
                                
                                new_chromosomes.append(child)
                        else:
                            
                            new_chromosomes.append(child)
                else:
                    
                    if len(new_chromosomes)%2 == 1:
                        
                        if len(new_chromosomes) == population_size - 1:
                            
                            new_chromosomes.append(tournament_winners[np.random.randint(0,2)])
                        else:
                            
                            for winner in tournament_winners:
                                new_chromosomes.append(winner)
                    else:
                        
                        for winner in tournament_winners:
                            new_chromosomes.append(winner)
                             
            chromosomes_list = [*new_chromosomes]
        
        end = time.time()
        
    time_spent = end - start
    
    return gen, time_spent, chromosomes_list[max_score_index].gene_sequence


def genetic_algorithm_spinning_roulette_wheel_version(epochs: int, knapsack_max_weight: int, stopping_criterion: int, genes_list: List[Gene], population_size: int, init_chromosomes: Optional[List[Chromosome]] = None, crossover_rate: float = 0.95, chunck_size: int = 1, mutation_rate: float = 0.05) -> Tuple[int, float, List]:
    """Creates a genetic algorithm.

    Args:
        epochs (int): The maximum number of the iterations.
        knapsack_max_weight (int): The maximum weight capacity of the knapsack.
        stopping_criterion (int): The stopping criterion of the iteration process. 
        genes_list (List[Gene]): The list of the genes that define the knapsack problem.
        population_size (int): The size of the chromosome population.
        init_chromosomes (Optional[List[Chromosome]], optional): The list with the initial chromosomes. Defaults to None.
        crossover_rate (float, optional): The crossover chromosomes rate. Defaults to 0.95.
        chunck_size (int): . Defaults to 1.
        mutation_rate (float, optional): The mutation rate of a gene. Defaults to 0.05.

    Returns:
        Tuple[int, float, List]: A tuple with the number of epochs to reach the solution, the time spent and a list with the best gene sequence of the chromosomes that represents the solution.
    """
    if init_chromosomes is None:
        
        start = time.time()
        chromosomes_list = create_chromosomes(population_size = population_size, genes_list = genes_list)
              
        for gen in range(epochs):
            
            for chromosome in chromosomes_list:
                
                chromosome.fitness_assignment(max_weight = knapsack_max_weight, genes_list = genes_list)
            
            chromosomes_scores = [chromosome.score for chromosome in chromosomes_list]
            max_score = max(chromosomes_scores)
            max_score_index = chromosomes_scores.index(max_score)
            
            if max_score >= stopping_criterion:
                break
            
            new_chromosomes = []
            
            while len(new_chromosomes) < population_size:
            
                srw_winners = spinning_roulette_wheel_selection(chromosome_list = chromosomes_list)
                
                rng_number1 = np.random.random()
                
                if rng_number1 < crossover_rate:
                        
                    childrens = crossover_chromosomes(parent1 = srw_winners[0], parent2 = srw_winners[1], chunck_size = chunck_size)
                    
                    for child in childrens:
                        
                        mutation(chromosome = child, mutation_rate = mutation_rate)
                        
                        if population_size%2 == 1:
                            
                            if len(new_chromosomes) == population_size - 1:
                                
                                new_chromosomes.append(child)
                                break
                            else:
                                
                                new_chromosomes.append(child)
                        else:
                            
                            new_chromosomes.append(child)
                else:
                    
                    if len(new_chromosomes)%2 == 1:
                        
                        if len(new_chromosomes) == population_size - 1:
                            
                            new_chromosomes.append(srw_winners[np.random.randint(0,2)])
                        else:
                            
                            for winner in srw_winners:
                                new_chromosomes.append(winner)
                    else:
                        
                        for winner in srw_winners:
                            new_chromosomes.append(winner)
                             
            chromosomes_list = [*new_chromosomes]
            
        end = time.time()
        
    else:
        
        start = time.time()
        population_size = len(init_chromosomes)
        chromosomes_list = [*init_chromosomes]
              
        for gen in range(epochs):
            
            for chromosome in chromosomes_list:
                
                chromosome.fitness_assignment(max_weight = knapsack_max_weight, genes_list = genes_list)
            
            chromosomes_scores = [chromosome.score for chromosome in chromosomes_list]
            max_score = max(chromosomes_scores)
            max_score_index = chromosomes_scores.index(max_score)
            
            if max_score >= stopping_criterion:
                break
            
            new_chromosomes = []
            
            while len(new_chromosomes) < population_size:
            
                srw_winners = spinning_roulette_wheel_selection(chromosome_list = chromosomes_list)
                
                rng_number1 = np.random.random()
                
                if rng_number1 < crossover_rate:
                        
                    childrens = crossover_chromosomes(parent1 = srw_winners[0], parent2 = srw_winners[1], chunck_size = chunck_size)
                    
                    for child in childrens:
                        
                        mutation(chromosome = child, mutation_rate = mutation_rate)
                        
                        if population_size%2 == 1:
                            
                            if len(new_chromosomes) == population_size - 1:
                                
                                new_chromosomes.append(child)
                                break
                            else:
                                
                                new_chromosomes.append(child)
                        else:
                            
                            new_chromosomes.append(child)
                else:
                    
                    if len(new_chromosomes)%2 == 1:
                        
                        if len(new_chromosomes) == population_size - 1:
                            
                            new_chromosomes.append(srw_winners[np.random.randint(0,2)])
                        else:
                            
                            for winner in srw_winners:
                                new_chromosomes.append(winner)
                    else:
                        
                        for winner in srw_winners:
                            new_chromosomes.append(winner)
                             
            chromosomes_list = [*new_chromosomes]
            
        end = time.time()
    
    time_spent = end - start
    
    return gen, time_spent, chromosomes_list[max_score_index].gene_sequence
  