import subprocess
import os
import pandas as pd
import random
import numpy as np
import csv
import re
import argparse
import copy
from datetime import datetime

class MAGNETO:
    def __init__(self, hw_config='', model_name='', popluation_size=100, max_generations=10, power_constraint=1000):
        super(MAGNETO,self).__init__()
        self.hw_config = hw_config
        self.model_name = model_name
        self.num_PEs = 0
        self.freq_MHz = 350
        with open(f'data/hw/{self.hw_config}.m') as f:
            for line in f:
                if 'num_pes' in line:
                    self.num_PEs = int(line.split(":")[1])
                    print(f"Number of PEs: {self.num_PEs}")

        self.dimensions = {}
        self.popluation_size = popluation_size
        self.max_generations = max_generations
        self.power_constraint = power_constraint
        self.elite_size = popluation_size * 0.1
        
        self.best_fitness = -1
        self.best_perf1 = float('inf') # latency
        self.best_perf2 = 0 # utilization
        self.best_perf3 = float('inf') # memory access
        self.best_perf4 = 0 # reuse factor
        
        self.max_cycles = None
        self.max_weighted_memory_access = None
        self.max_averaged_reuse_factor = None
        self.max_edp = None
        self.energy_nJ = None
        
        self.best_fitness_score_list = [] # [fitness score] * generation size
        self.best_perf1_list = [] # [metric1 value] * generation size
        self.best_perf2_list = []
        self.best_perf3_list = []
        self.best_perf4_list = []
        self.best_mapper_list = []

    def run_maestro_to_get_all_metrics(self, mapping_file_path):
        try:
            if os.path.exists(f'{self.model_name}_mapping.csv'):
                os.remove(f'{self.model_name}_mapping.csv')
            command = [
                './maestro',
                f"--HW_file=data/hw/{self.hw_config}.m",
                f"--Mapping_file={mapping_file_path}",
                "--print_res=true",
                "--print_res_csv_file=true",
                "--print_log_file=true"
            ]
            result = subprocess.run(
                command, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True
            )
            
            if result.stderr:
                print("MAESTRO Errors:", result.stderr)

            if result.returncode != 0:
                raise RuntimeError(f"MAESTRO failed with error: {result.stderr}")
            
            result_csv = f"{self.model_name}_mapping.csv" # might has some delay here, but it's fine
            
            if os.path.exists(result_csv):
                return result_csv
            elif os.path.exists(f"{mapping_file_path.split('.')[0]}.csv"): # temporary file
                return f"{mapping_file_path.split('.')[0]}.csv"
            else:
                print("CSV file not generated.")
                return None

        except subprocess.CalledProcessError as e:
            print(f"Error running MAESTRO: {e.stderr}")
            return None

    def select_parents(self, population, fitness_scores):
        Rw = min(fitness_scores)
        Rb = max(fitness_scores)
        if Rw == Rb:
            normalized_fitness_scores = [1.0 for _ in fitness_scores]
        else:
            normalized_fitness_scores = [((fitness - Rw) / (Rb - Rw)) * 3 for fitness in fitness_scores]
        parents = random.choices(population, weights=normalized_fitness_scores, k=2)
        return parents # [parent mapper1, parent mapper2]
    
    def evaluate_chromosome(self, chromosome):
        freq_MHz = self.freq_MHz
        # Generate temporary mapping and run MAESTRO
        temp_model_with_mapping_m = self.integrate_dataflow_in_model(self.model_name, chromosome, is_temp=True)
        temp_result_csv = self.run_maestro_to_get_all_metrics(temp_model_with_mapping_m)
        
        # Read CSV using pandas and convert to NumPy
        df = pd.read_csv(temp_result_csv)
        data = df.to_numpy()

        runtime_col = df.columns.get_loc(' Runtime (Cycles)')
        energy_col = df.columns.get_loc(' Activity count-based Energy (nJ)')
        l1_read_col = df.columns.get_loc(' input l1 read')
        l1_write_col = df.columns.get_loc(' input l1 write')
        filter_read_col = df.columns.get_loc('filter l1 read')
        filter_write_col = df.columns.get_loc(' filter l1 write')
        output_read_col = df.columns.get_loc('output l1 read')
        output_write_col = df.columns.get_loc(' output l1 write')
        l2_read_col = df.columns.get_loc(' input l2 read')
        l2_write_col = df.columns.get_loc(' input l2 write')
        offchip_bw_col = df.columns.get_loc(' Offchip BW Req (Elements/cycle)')
        reuse_input_col = df.columns.get_loc(' input reuse factor')
        reuse_filter_col = df.columns.get_loc(' filter reuse factor')
        reuse_output_col = df.columns.get_loc(' output reuse factor')

        # Vectorized calculations
        cycles = np.sum(data[:, runtime_col])
        onchip_energy_nJ = np.sum(data[:, energy_col])
        l1_access = np.sum(data[:, l1_read_col] + data[:, l1_write_col] +
                        data[:, filter_read_col] + data[:, filter_write_col] +
                        data[:, output_read_col] + data[:, output_write_col])
        l2_access = np.sum(data[:, l2_read_col] + data[:, l2_write_col])
        dram_access = np.sum(data[:, offchip_bw_col] * data[:, runtime_col])
        averaged_reuse_factor = np.mean(
            (data[:, reuse_input_col] + data[:, reuse_filter_col] + data[:, reuse_output_col]) / 3
        )
        dram_access_energy_nJ = np.sum(data[:, offchip_bw_col] * data[:, runtime_col] * 2 * 8 * 30)

        # EDP calculation
        edp = cycles * (onchip_energy_nJ + dram_access_energy_nJ) / (freq_MHz * 1e6)  # Convert to Hz
        energy_nJ = onchip_energy_nJ + dram_access_energy_nJ

        # Weighted memory access calculation
        weighted_memory_access = l1_access + 5 * l2_access + 10 * dram_access

        # Power proxy calculation
        power_proxy_value = (onchip_energy_nJ + dram_access_energy_nJ) / cycles * freq_MHz * 1e-9 * 1e6

        # Normalize the values
        self.max_cycles = max(self.max_cycles or 0, cycles)
        self.max_weighted_memory_access = max(self.max_weighted_memory_access or 0, weighted_memory_access)
        self.max_averaged_reuse_factor = max(self.max_averaged_reuse_factor or 0, averaged_reuse_factor)
        self.max_edp = max(self.max_edp or 0, edp)
        self.energy_nJ = max(self.energy_nJ or 0, energy_nJ)

        norm_cycles = np.log(cycles + 1) / np.log(self.max_cycles + 1)
        norm_weighted_memory_access = (weighted_memory_access - 1) / (self.max_weighted_memory_access + 1)
        norm_averaged_reuse_factor = averaged_reuse_factor / self.max_averaged_reuse_factor
        norm_edp = edp / self.max_edp
        norm_energy = energy_nJ / self.energy_nJ

        # Fitness score calculation
        offset = 5
        # fitness_score = 0.3 * norm_averaged_reuse_factor - 0.5 * norm_edp - 0.2 * norm_weighted_memory_access + offset
        fitness_score = - 0.7 * norm_cycles + 0.3 * norm_averaged_reuse_factor + offset # reuse_factor increases as memory access decreases
        # if power_proxy_value > self.power_constraint:
        #     fitness_score = -100
        if power_proxy_value > self.power_constraint:
            fitness_score -= ((power_proxy_value - self.power_constraint) / self.power_constraint) * 10


        # Cleanup temporary files
        if os.path.exists(temp_model_with_mapping_m):
            os.remove(temp_model_with_mapping_m)
        if os.path.exists(temp_result_csv):
            os.remove(temp_result_csv)

        return fitness_score, (cycles, edp, weighted_memory_access, averaged_reuse_factor)

    def evaluate_population(self, population):
        evaluated_pop = []
        for chromosome in population:
            fitness_score, performance = self.evaluate_chromosome(chromosome)
            evaluated_pop.append((chromosome, fitness_score, performance))
        
        best_idx = np.argmax([x[1] for x in evaluated_pop])
        self.best_mapper_list.append(evaluated_pop[best_idx][0])
        self.best_fitness_score_list.append(evaluated_pop[best_idx][1])
        self.best_perf1_list.append(evaluated_pop[best_idx][2][0]) # cycles
        self.best_perf2_list.append(evaluated_pop[best_idx][2][1])
        self.best_perf3_list.append(evaluated_pop[best_idx][2][2])
        self.best_perf4_list.append(evaluated_pop[best_idx][2][3])
            
        return evaluated_pop # [(chromosome, fitness, (per1, perf2)), ...]

    @staticmethod
    def extract_dimensions(file_path):
        """
            Return a list of dictionaries containing the dimensions of the DNN layers
            CONV - [{'K': 256, 'C': 128, 'R': 3, 'S': 3, 'Y': 14, 'X': 14}]
            GEMM - [{'K': 512, 'C': 512, 'R': 1, 'S': 1, 'Y': 256, 'X': 1 }]
            FC - [{'K': 256, 'C': 1024, 'R': 1, 'S': 1, 'Y': 1, 'X': 1}]
            multi_layers - [{'K': 256, 'C': 128, 'R': 3, 'S': 3, 'Y': 14, 'X': 14}, {'K': 512, 'C': 512, 'R': 1, 'S': 1, 'Y': 256, 'X': 1 }, {'K': 256, 'C': 1024, 'R': 1, 'S': 1, 'Y': 1, 'X': 1}]
        """
        dimensions_list = []
        pattern = re.compile(r"Dimensions\s*{([^}]+)}")

        try:
            with open(file_path, 'r') as f:
                content = f.read()

            # Extract all matches for multi-layer or single-layer
            matches = pattern.findall(content)

            if matches:
                for match in matches:
                    dim_pairs = re.findall(r"(\w+):\s*(\d+)", match.strip())
                    dimensions = {key: int(value) for key, value in dim_pairs}
                    dimensions_list.append(dimensions)

            return dimensions_list if dimensions_list else [{}]

        except FileNotFoundError:
            print(f"File {file_path} not found.")
            return [{}]

    @staticmethod
    def get_factors(n):
        """ Returns a list of factors of n """
        # factors = [i for i in range(1, n + 1) if n % i == 0]
        factors = [i for i in range(1, n + 1)]
        return factors

    def get_available_tile_sizes(self):
        dimensions = self.dimensions
        dim_list = ['K', 'C', 'Y', 'X', 'R', 'S']
        l1_mapper_dim_tiles_dict = {}
        l2_mapper_dim_tiles_dict = {}
        
        largest_key = max(dimensions, key=dimensions.get)
        largest_value = dimensions[largest_key]

        for dim in dim_list:
            factors = self.get_factors(largest_value)

            # R, S is only allowed to have tile size equal to the dimension
            if dim in {'R', 'S'}:
                l1_mapper_dim_tiles_dict[dim] = [dimensions[dim]]
                l2_mapper_dim_tiles_dict[dim] = [dimensions[dim]]
                
            # X, Y is only allowed to have tile size equal to 1 in second index of L2
            elif dim in {'X', 'Y'}:
                l1_mapper_dim_tiles_dict[dim] = [f for f in factors if f <= dimensions[dim]]
                l2_mapper_dim_tiles_dict[dim] = [1]
            else:
                l1_mapper_dim_tiles_dict[dim] = [f for f in factors if f <= dimensions[dim]]
                l2_mapper_dim_tiles_dict[dim] = [f for f in factors if f <= dimensions[dim]]
        
        return {'l1': l1_mapper_dim_tiles_dict, 'l2': l2_mapper_dim_tiles_dict}
    
    def get_available_pod_sizes(self):
        return self.get_factors(self.num_PEs)
    
    def check_constraints_cluster(self, pod): # [['P', cluster_size]]
        """
            Check if the pod size is valid
        """
        return pod <= self.num_PEs
    
    def check_mapspace_validity(self, mapper, level): # chromosome slice: single l1 or l2 mapper [[dim, tile_size], ...]
        """
            Single level of a mapper
        """
        valid = True

        # Check if there are duplicate dimensions
        dimensions = [gene[0] for gene in mapper]
        if len(dimensions) != len(set(dimensions)):
            print("\tDuplicate dimensions found.")
            valid = False

        # Check if there is R or S with tile size not equal to the dimension
        for _, (dim, tile_size) in enumerate(mapper):
            if dim in {'R', 'S'} and tile_size != self.dimensions[dim]:
                print(f"\tTile size of {dim} must be equal to the dimension")
                valid = False
                break
            # Check if X and Y in l2 is equal to 1
            if level == 'l2' and dim in {'X', 'Y'} and tile_size != 1:
                print("\tTile size of X and Y in L2 must be equal to 1")
                valid = False
                break
        
        return valid

    def encode_mapper(self, dimensions, level='l1'):
        """
        Encodes a mapper level by choosing spatial and temporal dimensions with valid factors.
        """
        dim_list = ['K', 'C', 'Y', 'X', 'R', 'S']
        mapper = []
        
        for dim_name in dim_list:
            factors = self.get_factors(dimensions[dim_name])
            if dim_name in {'R', 'S'}:
                tile_size = dimensions[dim_name]
            elif dim_name in {'X', 'Y'}:
                tile_size = 1 if level == 'l2' else random.choice([f for f in factors if f <= dimensions[dim_name]])
            else:
                tile_size = random.choice([f for f in factors if f <= dimensions[dim_name]])
            mapper.append([dim_name, tile_size])

        random.shuffle(mapper)
        if level == 'l2':
            while mapper[0][0] in {'X', 'Y'}:
                random.shuffle(mapper)

        return mapper # [['K', 3], ...]

    def find_best_score(self, fitness_scores):
        best_score = max(fitness_scores)
        best_idx = fitness_scores.index(best_score)
        return best_score, best_idx

    def create_chromosome(self):
        dim_list = ['K', 'C', 'Y', 'X', 'R', 'S']
        sp = random.choice(dim_list)
        sp_factors = self.get_factors(self.dimensions[sp])
        valid_factors = [f for f in sp_factors if 2 <= f <= min(self.dimensions[sp], self.num_PEs)]
        cluster_size = random.choice(valid_factors) if valid_factors else 1
        
        l2_mapper = self.encode_mapper(self.dimensions, level='l2')
        l1_mapper = self.encode_mapper(self.dimensions, level='l1')
        return l2_mapper + [['P', cluster_size]] + l1_mapper

    def create_dataflow_template_string(self, mapper):
        dataflow_template = ''
        dataflow_template += "\t\tDataflow {\n"        
        
        def create_mapper_string(mapper):
            mapper_string = ''
            for i, (dim, tile_size) in enumerate(mapper):
                if i == 0:
                    if dim == 'R':
                        mapper_string += f"\t\t\tSpatialMap(Sz(R),Sz(R)) R;\n"
                    elif dim == 'S':
                        mapper_string += f"\t\t\tSpatialMap(Sz(S),Sz(S)) S;\n"
                    else:
                        mapper_string += f"\t\t\tSpatialMap({tile_size},{tile_size}) {dim};\n" 
                else:
                    if dim == 'R':
                        mapper_string += f"\t\t\tTemporalMap(Sz(R),Sz(R)) R;\n"
                    elif dim == 'S':
                        mapper_string += f"\t\t\tTemporalMap(Sz(S),Sz(S)) S;\n"
                    else:
                        mapper_string += f"\t\t\tTemporalMap({tile_size}, {tile_size}) {dim};\n"
            return mapper_string
        
        l2_mapper, cluster, l1_mapper = mapper[:6], mapper[6], mapper[7:] # [['K', 3], ...]
        
        dataflow_template += create_mapper_string(l2_mapper)
        dataflow_template += f"\t\t\tCluster({cluster[1]}, P);\n"
        dataflow_template += create_mapper_string(l1_mapper)
        dataflow_template += "\t\t}\n"

        return dataflow_template
    
    def integrate_dataflow_in_model(self, model_name, mapper, is_temp=False):
        if is_temp:
            output_model_path = f"temp_magneto_{model_name}.m"
        else:
            output_model_path = f"data/mapping/magneto_{model_name}.m"

        dataflow_template = self.create_dataflow_template_string(mapper)

        with open(f'data/model/{model_name}.m', "r") as infile:
            model_lines = infile.readlines()

        output_lines = []
        for line in model_lines:
            output_lines.append(line)
            if "Dimensions {" in line:
                output_lines.append(dataflow_template)

        with open(output_model_path, "w") as outfile:
            outfile.writelines(output_lines)

        return output_model_path
    
    def save_to_csv(self, lists_dict, layer_name):
        """
            Sample lists_dict: {'Best Fitness Score': [1, 2, 3], 'Best Utilization': [0.1, 0.2, 0.3], 'Best Mapping': [[], [], []]}
        """
        # Get current date and time
        current_time = datetime.now().strftime("%Y_%m_%d_%H:%M_%S")
        filename = f'fitness_{layer_name}_{current_time}.csv'

        # Find the maximum length of lists
        max_length = max(len(lst) for lst in lists_dict.values())

        # Pad shorter lists with None
        padded_lists = {key: lst + [None] * (max_length - len(lst)) for key, lst in lists_dict.items()}

        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(padded_lists.keys())
            for row in zip(*padded_lists.values()):
                writer.writerow(row)
        
        print(f"Data saved to {filename}")
    
    def initialize_population(self):
        population = []
        for _ in range(self.popluation_size):
            mapper = self.create_chromosome()
            population.append(mapper)
        return population
    
    @staticmethod
    def get_matching_dim_index_pairs(slice1, slice2, avaliable_dims=['K', 'C', 'R', 'S', 'Y', 'X']):
        """
        Return index pairs of matching dimensions between two slices
        Return: [(0, 3), (3, 2)...]
        """
        matching_pairs = []
        for i, (dim1, _) in enumerate(slice1):
            for j, (dim2, _) in enumerate(slice2):
                if dim1 == dim2 and dim1 in avaliable_dims:
                    matching_pairs.append((i, j))
        return matching_pairs

    def crossover(self, chromosomes, crossover_prob=0.7, max_retries=5):
        if random.random() >= crossover_prob: return chromosomes

        parent1, parent2 = chromosomes
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)

        def cross_dims(slice1, slice2, dim_pairs):
            if not dim_pairs: return slice1, slice2

            crossed_slice1 = copy.deepcopy(slice1)
            crossed_slice2 = copy.deepcopy(slice2)
            
            idx1, idx2 = random.choice(dim_pairs)
            crossed_slice1[idx1], crossed_slice2[idx2] = slice2[idx2], slice1[idx1]
            return crossed_slice1, crossed_slice2
        
        rand_value = random.random()
        
        if rand_value < 0.4: # intra-level crossover
            dim_pairs1 = self.get_matching_dim_index_pairs(parent1[:6], parent2[:6])
            dim_pairs2 = self.get_matching_dim_index_pairs(parent1[7:], parent2[7:])
            child1_slice1, child2_slice1 = cross_dims(parent1[:6], parent2[:6], dim_pairs1) # l1
            child1_slice2, child2_slice2 = cross_dims(parent1[7:], parent2[7:], dim_pairs2) # l2
            
            child1 = child1_slice1 + [['P', parent1[6][1]]] + child1_slice2
            child2 = child2_slice1 + [['P', parent2[6][1]]] + child2_slice2
        
        elif rand_value < 0.7: # inter-level crossover
            dim_pairs1 = self.get_matching_dim_index_pairs(parent1[:6], parent2[7:], avaliable_dims=['K', 'C', 'R', 'S'])
            dim_pairs2 = self.get_matching_dim_index_pairs(parent1[7:], parent2[:6], avaliable_dims=['K', 'C', 'R', 'S'])
            child1_slice1, child2_slice2 = cross_dims(parent1[:6], parent2[7:], dim_pairs1)
            child1_slice2, child2_slice1 = cross_dims(parent1[7:], parent2[:6], dim_pairs2)
            
            child1 = child1_slice1 + [['P', parent1[6][1]]] + child1_slice2
            child2 = child2_slice1 + [['P', parent2[6][1]]] + child2_slice2
        
        else: # cluster crossover
            child1 = parent1[:6] + [['P', parent2[6][1]]] + parent1[7:]
            child2 = parent2[:6] + [['P', parent1[6][1]]] + parent2[7:]
            
        # Check constraints
        valid_slice1 = (
            self.check_mapspace_validity(child1[:6], level='l2') and
            self.check_mapspace_validity(child1[7:], level='l1')
        )
        valid_slice2 = (
            self.check_mapspace_validity(child2[:6], level='l2') and
            self.check_mapspace_validity(child2[7:], level='l1')
        )
        valid_cluster = (
            self.check_constraints_cluster(child1[6][1]) and 
            self.check_constraints_cluster(child2[6][1])
        )

        if valid_slice1 and valid_slice2 and valid_cluster:
            return child1, child2
        else:
            return parent1, parent2
    
    def mutate_tiles(self, chromosome, mutation_prob=0.2):
        if random.random() >= mutation_prob: return chromosome
        
        dim_list = ['K', 'C', 'Y', 'X', 'R', 'S']        
        parent = chromosome
        child = copy.deepcopy(parent)
        available_tile_sizes = self.get_available_tile_sizes()
        
        # L2 Mapper mutation
        idx_l2 = random.randint(0, 5)
        valid_factors_l2 = available_tile_sizes['l2'][parent[idx_l2][0]]
        if parent[idx_l2][0] in dim_list:
            child[idx_l2][1] = random.choice(valid_factors_l2)
        
        # L1 Mapper mutation
        idx_l1 = random.randint(7, 12)
        valid_factors_l1 = available_tile_sizes['l1'][parent[idx_l1][0]]
        if parent[idx_l1][0] in dim_list:
            child[idx_l1][1] = random.choice(valid_factors_l1)

        # Check constraints
        valid_child = (
            self.check_mapspace_validity(child[:6], level='l2') and
            self.check_mapspace_validity(child[7:], level='l1')
        )
            
        if valid_child:
            return child
        else:
            return parent
    
    def mutate_shuffle(self, chromosome, mutation_prob=0.2):
        if random.random() >= mutation_prob: return chromosome
        
        if random.random() < 0.5:
            l2_mapper = chromosome[:6]
            random.shuffle(l2_mapper)
        else:
            l1_mapper = chromosome[7:]
            random.shuffle(l1_mapper)
        return chromosome

    def mutate_cluster(self, chromosome, mutation_prob=0.2):
        if random.random() >= mutation_prob: return chromosome
        
        available_pod_sizes = self.get_available_pod_sizes()
        child = copy.deepcopy(chromosome)
        
        child[6][1] = random.choice(available_pod_sizes)

        # Constraints validation
        if self.check_constraints_cluster(child[6][1]):
            return child
        else:
            return chromosome

    def replace_population(self, population, fitness_scores, elite_ratio=0.1, random_ratio=0.2):
        """
        - top elite_ratio(%) maintained as is
        - random_ratio(%) replaced with random individuals
        """
        pop_size = len(population)
        sorted_indices = sorted(range(pop_size), key=lambda i: fitness_scores[i], reverse=True)

        # 엘리트 개체: 상위 elite_ratio%
        elite_count = int(pop_size * elite_ratio)
        elite_indices = set(sorted_indices[:elite_count])

        # 나머지 중에서 random_ratio%를 무작위 교체
        non_elite_indices = [i for i in range(pop_size) if i not in elite_indices]
        num_to_restart = int(len(non_elite_indices) * random_ratio)
        indices_to_restart = random.sample(non_elite_indices, num_to_restart)

        for idx in indices_to_restart:
            population[idx] = self.create_chromosome()

        print(f"Restarted {num_to_restart} individuals; kept top {elite_count} elites.")
        return population

    def run_ga(self, dimensions): 
        self.dimensions = dimensions[0]
        population = self.initialize_population()
        best_fitness, best_idx = -1, -1

        result_csv = f"{self.model_name}_mapping.csv"
        if os.path.exists(result_csv):
            os.remove(result_csv)

        no_improvement_generations = 0
        last_best_fitness = -float('inf')

        # Evaluate the initial population
        evaluated_population = self.evaluate_population(population)

        fitness_scores = [fitness for _, fitness, _ in evaluated_population]
        best_fitness, best_idx = self.find_best_score(fitness_scores)
        elite_individual = copy.deepcopy(population[best_idx])
        
        for i in range(self.max_generations):
            print(f"Generation {i + 1}......")
    
            # Step 1: Selection
            selected_parents = []
            for _ in range(len(population) // 2):
                parents = self.select_parents(population, fitness_scores)
                selected_parents.append(parents)
            
            # Step2: Crossover
            offspring = []
            for parent1, parent2 in selected_parents: # len(population) // 2 times
                # Randomly choose between inter-level, intra-level and cluster crossover
                child1, child2 = self.crossover([parent1, parent2], crossover_prob=0.6)
                offspring.extend([child1, child2]) # len(offspring) == len(population)
            
            # Step3: Mutation
            for individual in offspring:
                rand_value = random.random()
                # Randomly choose between tile mutation, shuffle mutation and cluster mutation
                if rand_value < 0.3:
                    new_child = self.mutate_shuffle(individual, mutation_prob=0.5)
                elif rand_value < 0.6:
                    new_child = self.mutate_tiles(individual, mutation_prob=0.8)
                else:
                    new_child = self.mutate_cluster(individual, mutation_prob=0.6)
                individual[:] = new_child

            population = offspring
            
            # Step 4: Replacement
            worst_index = fitness_scores.index(min(fitness_scores))
            population[worst_index] = elite_individual

            # Step 5: Evaluate the new populations
            evaluated_population = self.evaluate_population(population)
            fitness_scores = [fitness for _, fitness, _ in evaluated_population]
            performance_scores = [perf for _, _, perf in evaluated_population]

            # Step 7: Identify new best individual
            best_fitness, best_idx = self.find_best_score(fitness_scores)
            elite_individual = copy.deepcopy(population[best_idx])  # Update elite individual

            best_perf1, best_perf2, best_perf3, best_perf4 = performance_scores[best_idx]
            self.best_fitness = best_fitness
            self.best_perf1 = best_perf1
            self.best_perf2 = best_perf2
            self.best_perf3 = best_perf3
            self.best_perf4 = best_perf4
            print(f'\tBest fitness score: {best_fitness:.4f}, Best Cycle: {best_perf1:.4f}, Best EDP: {best_perf2:.4f}, Best Memory Access: {best_perf3:.4f}, Best Reuse Factor: {best_perf4:.4f}, \n \tBest Mapper: {population[best_idx]}')

            # Step 6: Track stagnation and apply replacement if needed
            if best_fitness <= last_best_fitness:
                no_improvement_generations += 1
            else:
                no_improvement_generations = 0
                last_best_fitness = best_fitness
            
            if no_improvement_generations >= 3:  # Stagnation threshold
                population = self.replace_population(population, fitness_scores, elite_ratio=0.1, random_ratio=0.3)
                no_improvement_generations = 0


        self.save_to_csv({
            'Best Fitness Score': self.best_fitness_score_list,
            f'Best Cycle': self.best_perf1_list,
            f'Best EDP': self.best_perf2_list,
            f'Best Memory Access': self.best_perf3_list,
            f'Best Reuse Factor': self.best_perf4_list,
            f'Best Mapper': self.best_mapper_list,
        }, self.model_name)

        print("\tBest mapper: ", population[best_idx])
        return population[best_idx]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--generation', type=int, default=10, help='Number of generations')
    parser.add_argument('--population', type=int, default=10, help='Number of populations')
    parser.add_argument('--model_name', type=str, default='single_conv', help='Model defined in data/model in MAESTO format')
    parser.add_argument('--hwconfig', type=str, default='mobile', choices=('mobile', 'cloud'), help='Hardware config defined in data/hw in MAESTRO format')
    parser.add_argument('--power_budget_mw', type=int, default='1000', help='Power budget in mW')
    args = parser.parse_args()
    
    # Run the Genetic Algorithm
    model = MAGNETO(
        hw_config=args.hwconfig,
        model_name=args.model_name,
        popluation_size=args.population,
        max_generations=args.generation,
        power_constraint=args.power_budget_mw
    )
    dnn_layer_dict = model.extract_dimensions(f'data/model/{args.model_name}.m')
    best_mapper = model.run_ga(dimensions=dnn_layer_dict)
    
    # Integrate the best mapper in the entire model
    final_model_with_mapping_m = model.integrate_dataflow_in_model(args.model_name, best_mapper)
    final_result_csv = model.run_maestro_to_get_all_metrics(final_model_with_mapping_m)

    print("Optimal mapper: ", best_mapper)
    print(f'Final mapping file saved in {final_model_with_mapping_m}') # data/mapping/final_result_single_conv.m
