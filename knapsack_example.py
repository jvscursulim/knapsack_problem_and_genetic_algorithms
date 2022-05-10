from gene import Gene
from chromosome import Chromosome
from genetic_algorithm import genetic_algorithm_tournament_version
from genetic_algorithm import genetic_algorithm_spinning_roulette_wheel_version

# Best option for this setting: [1,0,0,1]

genes_list = []
genes_list.append(Gene(weight = 18, points = 20))
genes_list.append(Gene(weight = 20, points = 16))
genes_list.append(Gene(weight = 11, points = 19))
genes_list.append(Gene(weight = 15, points = 22))

print("-----------")
print("Genes info:")
print("-----------")
for gene in genes_list:
    print(f"weight: {gene.weight} | points: {gene.points}")
print("-----------")
print("\n")

chromosomes = []
chromosomes.append(Chromosome(gene_sequence = [0,1,0,0]))
chromosomes.append(Chromosome(gene_sequence = [1,0,0,0]))
chromosomes.append(Chromosome(gene_sequence = [0,0,1,0]))
chromosomes.append(Chromosome(gene_sequence = [0,0,0,1]))

print("-----------------")
print("Chromosomes info:")
print("-----------------")
for c in chromosomes:
    print(f"gene sequence: {c.gene_sequence}")
print("-----------------")
print("\n")

gen, time, genes = genetic_algorithm_tournament_version(epochs = 51, stopping_criterion = 42, genes_list = genes_list, knapsack_max_weight = 33, population_size = 4, init_chromosomes = chromosomes)

print("----------------------------------------------------------------------------")
print("Knapsack problem solution using genetic algorithm with tournament selection:")
print(f"Generation: {gen}")
print(f"Time spent: {time} seconds")
print(f"Gene sequence: {genes}")
print("----------------------------------------------------------------------------")
print("\n")

gen, time, genes = genetic_algorithm_spinning_roulette_wheel_version(epochs = 51, stopping_criterion = 42, genes_list = genes_list, knapsack_max_weight = 33, population_size = 4, init_chromosomes = chromosomes)

print("-----------------------------------------------------------------------------------------")
print("Knapsack problem solution using genetic algorithm with spinning roulette wheel selection:")
print(f"Generation: {gen}")
print(f"Time spent: {time} seconds")
print(f"Gene sequence: {genes}")
print("-----------------------------------------------------------------------------------------")
