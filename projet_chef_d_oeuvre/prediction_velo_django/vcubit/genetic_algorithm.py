import numpy as np
from pygad import GA


class GeneticAlgorithm:
    def __init__(self, gdf_density, subset, parent_selection_type='sus',
                 crossover_type='uniform', crossover_probability=0.9,
                 mutation_type='adaptive', mutation_probability=[0.25, 0.05]):

        self.gdf_density = gdf_density
        self.nb_ep = gdf_density[subset].sum().tolist()
        gdf_special = gdf_density[(gdf_density['difference'] < 0) & (gdf_density[subset].sum(axis=1) != 0)]
        self.threshold = gdf_special['difference'].describe()['50%']
        self.gdf_special = gdf_special[gdf_special['difference'] <= self.threshold]
        self.nb_selected = len(self.gdf_special)
        self.subset = subset
        self.parent_selection_type = parent_selection_type
        self.crossover_type = crossover_type
        self.crossover_probability = crossover_probability
        self.mutation_type = mutation_type
        self.mutation_probability = mutation_probability
        self.best_fitness = []
        self.mean_fitness = []


    def set_parameters(self):
        num_generations = 20 + int(20 * np.log(len(self.subset)))
        sol_per_pop = 100 + 4 * len(self.subset)**2
        num_parents_mating = int(0.25 * sol_per_pop)
        return num_generations, sol_per_pop, num_parents_mating

    def fitness_func(self, ga_instance, solution, solution_idx):
        bonus = np.dot(self.gdf_special[self.subset], solution)
        score = (self.gdf_special['density_s'] + bonus) * self.gdf_special['coeff'] - self.gdf_special['inf_vcub_s']
        fitness = 1 / abs(score).mean()
        return fitness

    def on_generation(self, ga_instance):
        gen = ga_instance.generations_completed
        print(f'Progress: {round(100 * gen / self.num_generations)}%\n{gen}/{self.num_generations} generations completed', end='')
        if gen != self.num_generations: print('\033[F\033[F')
        current_fitness = ga_instance.best_solution()[1]
        self.best_fitness.append(1 / current_fitness)
        mean_fitness = sum(ga_instance.last_generation_fitness) / len(ga_instance.last_generation_fitness)
        self.mean_fitness.append(1 / mean_fitness)

    def run(self):

        self.num_generations, self.sol_per_pop, self.num_parents_mating  = self.set_parameters()

        print("--- Algorithme génétique en cours d'exécution ---")

        ga_instance = GA(
            num_generations=self.num_generations,
            num_parents_mating=self.num_parents_mating,
            fitness_func=self.fitness_func,
            sol_per_pop=self.sol_per_pop,
            num_genes=len(self.subset),
            gene_type=float,
            init_range_low=0,
            init_range_high=1,
            parent_selection_type=self.parent_selection_type,
            crossover_type=self.crossover_type,
            crossover_probability=self.crossover_probability,
            mutation_type=self.mutation_type,
            mutation_probability=self.mutation_probability,
            gene_space={'low' : 0, 'high' : 1},
            on_generation=self.on_generation
        )

        ga_instance.run()

        print("\n--- Exécution terminée ---")

        self.best_solution = list(ga_instance.best_solution()[0])
        self.mean_score_selected = abs(self.gdf_special['density_s'] * self.gdf_special['coeff'] - self.gdf_special['inf_vcub_s']).mean()
        self.mean_score_corrected = abs((self.gdf_special['density_s'] + np.dot(self.gdf_special[self.subset], self.best_solution)) * self.gdf_special['coeff'] - self.gdf_special['inf_vcub_s']).mean()

        return self.best_solution, self.best_fitness, self.mean_fitness, self.nb_selected, self.mean_score_selected, self.mean_score_corrected, self.num_generations, self.sol_per_pop, self.nb_ep
