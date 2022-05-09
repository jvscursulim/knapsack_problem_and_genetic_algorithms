import numpy as np
from gene import Gene
from typing import List

class Chromosome:
    """Chromosome class"""
    
    def __init__(self, gene_sequence: List[Gene]) -> None:
        """Creates a chromosome object.

        Args:
            gene_sequence (List[Gene]): A list with the genes that define the chromosome.
        """
        self.gene_sequence = gene_sequence
        self.total_weight = None
        self.score = None
        
    def fitness_assigment(self, max_weight: int, genes_list: List[Gene]) -> None:
        """Calculates the chromosome's total_weight and the score.

        Args:
            max_weight (int): Knapsack maximum weight (constraint).
            genes_list (List[Gene]): The genes considered in the problem. 
        """
        weights_np = np.array([gene.weight for gene in genes_list])
        points_np = np.array([gene.points for gene in genes_list])
        gene_sequence_np = np.array(self.gene_sequence)
        
        weights_gs_np = weights_np * gene_sequence_np
        points_gs_np = points_np * gene_sequence_np
        
        if weights_gs_np.sum() > max_weight:
            
            self.total_weight = weights_gs_np.sum()
            self.score = 0
        else:
            
            self.total_weight = weights_gs_np.sum()
            self.score = points_gs_np.sum()