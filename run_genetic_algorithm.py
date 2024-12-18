import subprocess
import os
import shutil
import pandas as pd
import math
import random
import numpy as np
import csv
import re
from functools import reduce
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import StandardScaler
from datetime import datetime

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
        
        print(f"Extracted Dimensions: {dimensions_list}")
        return dimensions_list if dimensions_list else [{}]

    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return [{}]
    
class GAMAPPER:
    def __init__(self, hw_config_file='', model_name='', fitness=['utilization', 'edp'], popluation_size=100, max_generations=10, alpha=0.7, beta=0.3):
        super(GAMAPPER,self).__init__()
        self.hw_config_file = hw_config_file
        self.hw_mapping_file = f'{model_name}_mapping'
        self.model_name = model_name
        self.fitness1 = fitness[0]
        self.fitness2 = fitness[1]
        self.alpha = alpha
        self.beta = beta

        self.num_PEs = 0
        with open(f'data/hw/{self.hw_config_file}.m') as f:
            for line in f:
                if 'num_pes' in line:
                    self.num_PEs = int(line.split(":")[1])
                    print(f"Number of PEs: {self.num_PEs}")

        self.dimensions = {}
        self.popluation_size = popluation_size
        self.max_generations = max_generations
        
        self.best_fitness = -1
        self.best_perf_fitness1 = 0
        self.best_perf_fitness2 = float('inf')
        
        self.best_fitness_score_list = [] # [fitness score] * generation size
        self.best_perf_fitness1_list = [] # [metric1 value] * generation size
        self.best_perf_fitness2_list = [] # [metric2 value] * generation size

    def run_maestro_to_get_all_metrics(self, mapping_file_path):
        try:
            if os.path.exists(f'{self.model_name}_mapping.csv'):
                os.remove(f'{self.model_name}_mapping.csv')
            command = [
                './maestro',
                f"--HW_file=data/hw/{self.hw_config_file}.m",
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

            # print("MAESTRO Output:", result.stdout)
            if result.stderr:
                print("MAESTRO Errors:", result.stderr)

            if result.returncode != 0:
                raise RuntimeError(f"MAESTRO failed with error: {result.stderr}")
            
            result_csv = f"{self.model_name}_mapping.csv" # might has some delay here, but it's fine
            
            if os.path.exists(result_csv):
                return result_csv
            else:
                print("CSV file not generated.")
                return None

        except subprocess.CalledProcessError as e:
            print(f"Error running MAESTRO: {e.stderr}")
            return None

    def select_parents(self, population, pop_fitness_score_list,tournament_size=3):
        def normalize_pop_fitness_score_list(pop_fitness_score_list):
            Rw = min(pop_fitness_score_list)
            Rb = max(pop_fitness_score_list)
            if Rw == Rb:  # Prevent division by zero
                return [1.0 for _ in pop_fitness_score_list]
            normalized_pop_fitness_score_list = [(Ri - Rw) + (Rb - Rw)/3 for Ri in pop_fitness_score_list]
            return normalized_pop_fitness_score_list
        
        normalized_pop_fitness_score_list = normalize_pop_fitness_score_list(pop_fitness_score_list)
        parents = random.choices(population, weights=normalized_pop_fitness_score_list, k=2)
        return parents

    def fetch_metric_by_name(self, df_row, fitness_string='utilization'):
        """
        Fetches the metric value from the DataFrame row based on the metric name.
        Normalized metric_value range is [0, 1]
        """
        metric_value = None
        normalized_metric_value = None
        
        if fitness_string == 'utilization':
            metric_value = df_row['Avg number of utilized PEs'] / self.num_PEs
            normalized_metric_value = metric_value
        elif fitness_string == 'latency':
            metric_value = df_row[' Runtime (Cycles)']
        elif fitness_string == 'energy':
            metric_value = df_row[' Activity count-based Energy (nJ)']
        elif fitness_string == 'edp':
            energy = df_row[' Activity count-based Energy (nJ)']
            runtime = df_row[' Runtime (Cycles)']
            edp = energy * runtime
            max_edp = 2e11 # for edge hw config
            metric_value = edp
            normalized_metric_value = np.log(edp + 1) / np.log(max_edp + 1)
        elif fitness_string == 'power':
            metric_value = df_row[' Power']
        elif fitness_string == 'area':
            metric_value = df_row[' Area']

        if not normalized_metric_value:
            normalized_metric_value = metric_value
        return metric_value, normalized_metric_value

    def evaluate_fitness_from_metrics(self, csv_file_path):
        try:
            df = pd.read_csv(csv_file_path)
            
            pop_fitness_score_list = [] # [calculated fitness score * population_size]
            pop_performance_list = [] # [(fitness1 value, fitness2 value) * population_size]
            
            pop_fitness1_list = [] # [fitness1 value * population_size]
            pop_fitness2_list = [] # [fitness2 value * population_size]
            pop_normalized_fitness1_list = [] # [fitness1 norm value * population_size]
            pop_normalized_fitness2_list = [] # [fitness2 norm value * population_size]
            
            for _, row in df.iterrows():
                fitness1_value, normalized_fitness1_value = self.fetch_metric_by_name(row, fitness_string=self.fitness1)
                fitness2_value, normalized_fitness2_value = self.fetch_metric_by_name(row, fitness_string=self.fitness2)

                # Append raw performance values
                pop_fitness1_list.append(fitness1_value)
                pop_fitness2_list.append(fitness2_value)
                pop_performance_list.append((fitness1_value, fitness2_value))
                
                # Append normalized performance values for calculating fitness score below
                pop_normalized_fitness1_list.append(normalized_fitness1_value)
                pop_normalized_fitness2_list.append(normalized_fitness2_value)

            # Calculate fitness scores
            for u, e in zip(pop_normalized_fitness1_list, pop_normalized_fitness2_list):
                offset = 10
                fitness_value = self.alpha * u - self.beta * e + offset
                pop_fitness_score_list.append(fitness_value)

            best_idx = np.argmax(pop_fitness_score_list)
            self.best_fitness_score_list.append(pop_fitness_score_list[best_idx])
            self.best_perf_fitness1_list.append(pop_fitness1_list[best_idx])
            self.best_perf_fitness2_list.append(pop_fitness2_list[best_idx])

            return pop_fitness_score_list, pop_performance_list

        except FileNotFoundError:
            print(f"CSV 파일 '{csv_file_path}'을 찾을 수 없습니다.")
            return [], []
        except Exception as e:
            print(f"에러 발생: {e}")
            return [], []

    def evaluate_population(self, population):
        """
            Evaluate the population by running MAESTRO and getting the fitness scores
            Return: List of fitness scores for each individual in the population
        """
        mapping_file_path = self.create_mapping_file(
            population,
            dimensions=self.dimensions,
            model_name=self.model_name
        )
        metric_csv = self.run_maestro_to_get_all_metrics(mapping_file_path)
        pop_fitness_score_list, pop_performance_list = self.evaluate_fitness_from_metrics(metric_csv)
        return pop_fitness_score_list, pop_performance_list

    @staticmethod
    def get_factors(n):
        """ Returns a list of factors of n """
        factors = [i for i in range(1, n + 1) if n % i == 0]
        return factors
    
    def get_available_tile_sizes(self):
        dimensions = self.dimensions
        dim_list = ['K', 'C', 'Y', 'X', 'R', 'S']
        l1_mapper_dim_tiles_dict = {}
        l2_mapper_dim_tiles_dict = {}
        
        for dim in dim_list:
            factors = self.get_factors(dimensions[dim])
            
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
    
    def check_constraints(self, mapper, level):
        """
            Single level of a mapper
        """
        valid = True
        dim_list = ['K', 'C', 'Y', 'X', 'R', 'S']
        
        # Check if 'P' is fixed as the first dimension
        if mapper[0][0] != 'P':
            # print("First gene must be 'P'")
            valid = False
            
        # Check if there are duplicate dimensions
        dimensions = [gene[0] for gene in mapper]
        if len(dimensions) != len(set(dimensions)):
            # print(f"Duplicate dimensions detected: {dimensions}")
            valid = False
        
        # Check if there is X or Y in L2 mapper index 1
        if level == 'l2':
            if mapper[1][0] in {'X', 'Y'}:
                # print("X or Y cannot be in L2 mapper")
                valid = False
        
        # Check if there is R or S with tile size not equal to the dimension
        for i, (dim, tile_size) in enumerate(mapper):
            if dim in {'R', 'S'} and tile_size != self.dimensions[dim]:
                # print(f"Tile size of {dim} must be equal to the dimension")
                valid = False
                break
            # Check if X and Y in l2 is equal to 1
            if level == 'l2' and dim in {'X', 'Y'} and tile_size != 1:
                # print("Tile size of X and Y in L2 must be equal to 1")
                valid = False
                break
            
        # Check if the cluster dimension is less than or equal to the number of PEs
        if level == 'l1':
            cluster_size = mapper[0][1]  # Cluster 크기 (m 값)
            spatial_size = mapper[1][1]  # SpatialMap 크기 (n 값)
            
            # 1. PE 수 초과 확인
            if cluster_size > self.num_PEs or cluster_size <= 0:
                # print(f"Invalid Cluster size: {cluster_size}. Must be within PE limit.")
                valid = False

            # 2. Cluster 크기 >= SpatialMap 크기 확인
            if cluster_size < spatial_size:
                # print(f"SpatialMap size {spatial_size} cannot exceed Cluster size {cluster_size}.")
                valid = False
            
            # 3. 유효한 매핑 차원 확인
            if mapper[0][0] != 'P':
                # print(f"Invalid mapping dimensions: {mapper[0][0]}. Expected 'P'.")
                valid = False

            # 4. PE가 1개일 경우 비허용
            if cluster_size == 1:
                # print("Cluster size cannot be 1.")
                valid = False
        
        return valid

    def encode_mapper(self, dimensions, level='l1'):
        """
        Encodes a mapper level by choosing spatial and temporal dimensions with valid factors.
        """
        dim_list = ['K', 'C', 'Y', 'X', 'R', 'S']
        sp = random.choice(dim_list)
        sp_factors = self.get_factors(dimensions[sp])
        valid_factors = [f for f in sp_factors if 2 <= f <= min(dimensions[sp], self.num_PEs)]
        sp_sz = random.choice(valid_factors) if valid_factors else 1

        df = []
        for dim_name in dim_list:
            factors = self.get_factors(dimensions[dim_name])
            if dim_name in {'R', 'S'}:
                tile_size = dimensions[dim_name]
            elif dim_name in {'X', 'Y'}:
                tile_size = 1 if level == 'l2' else random.choice([f for f in factors if f <= dimensions[dim_name]])
            else:
                tile_size = random.choice([f for f in factors if f <= dimensions[dim_name]])
            df.append([dim_name, tile_size])

        random.shuffle(df)
        if level == 'l2':
            while df[0][0] in {'X', 'Y'}:
                random.shuffle(df)
        mapper = [['P', sp_sz]] + df

        return mapper
    
    def find_best_score(self, pop_fitness_score_list):
        best_score = max(pop_fitness_score_list)
        best_idx = pop_fitness_score_list.index(best_score)
        return best_score, best_idx

    def create_chromosome(self):
        l2_mapper = self.encode_mapper(self.dimensions, level='l2')
        l1_mapper = self.encode_mapper(self.dimensions, level='l1')
        return l2_mapper + l1_mapper

    def create_mapping_file(self, population, dimensions, model_name):
        """
            Create mapping files for only an evaluation
        """
        mapping_file_path = f"data/mapping/{model_name}_mapping.m"
        
        mapping_template = f"Network {model_name} {{\n"
        dataflow_template = ''
        
        for idx, mapper in enumerate(population):
            dataflow_template += f"\tLayer CONV{idx} {{\n"
            dataflow_template += f"\t\tType: CONV\n"
            dimensions_str = ', '.join([f"{k}:{v}" for k, v in dimensions.items()])
            dataflow_template += f"\t\tDimensions {{ {dimensions_str} }}\n"
            dataflow_template += "\t\tDataflow {\n"
            
            l2_mapper, l1_mapper = mapper[:7], mapper[7:]
            
            for i, mapping in enumerate([l2_mapper, l1_mapper]):
                if i == 0: # L2
                    for j, (dim, tile_size) in enumerate(mapping[1:]):
                        if j == 0:
                            dataflow_template += f"\t\t\tSpatialMap({tile_size},{tile_size}) {dim};\n"
                        else:
                            dataflow_template += f"\t\t\tTemporalMap({tile_size}, {tile_size}) {dim};\n"
                else: # L1 
                    for j, (dim, tile_size) in enumerate(mapping):
                        if j == 0:
                            # cluster = min(dimensions[dim], 16384)
                            dataflow_template += f"\t\t\tCluster({tile_size}, {dim});\n"
                        elif j == 1:
                            dataflow_template += f"\t\t\tSpatialMap({tile_size}, {tile_size}) {dim};\n"
                        else:
                            dataflow_template += f"\t\t\tTemporalMap({tile_size}, {tile_size}) {dim};\n"
                    
            dataflow_template += "\t\t}\n"
            dataflow_template += "\t}\n"
        
        mapping_template += f"{dataflow_template}}}\n"

        with open(mapping_file_path, 'w') as f:
            f.write(mapping_template)
        
        return mapping_file_path
    
    def save_to_csv(self, lists_dict, layer_name):
        """
            Sample input: {'Best Fitness Score': [1, 2, 3], 'Best Utilization': [0.1, 0.2, 0.3]}
        """
        # Get current date and time
        current_time = datetime.now().strftime("%Y.%m.%d_%H:%M")
        filename = f'best_scores_{layer_name}_{current_time}.csv'
        
        # Find the maximum length of lists
        max_length = max(len(lst) for lst in lists_dict.values())

        # Pad shorter lists with None
        padded_lists = {key: lst + [None] * (max_length - len(lst)) for key, lst in lists_dict.items()}

        # Save to CSV
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write header
            writer.writerow(padded_lists.keys())
            # Write rows
            for row in zip(*padded_lists.values()):
                writer.writerow(row)
        
        print(f"Data saved to {filename}")
    
    def initialize_population(self):
        population = []
        for _ in range(self.popluation_size):
            mapper = self.create_chromosome()
            population.append(mapper)
        return population
    
    def get_matching_dim_pairs(self, child1, child2, start, end):
        """
        child1, child2 내의 동일 차원 이름을 갖는 인덱스 쌍을 반환한다.
        start, end 범위 내에서 탐색.
        반환 예: [(idx1_child1, idx2_child2), ...]
        """
        dim_map_child2 = {}
        # child2의 차원별 인덱스 모음
        for i in range(start, end):
            dim_name = child2[i][0]
            if dim_name not in dim_map_child2:
                dim_map_child2[dim_name] = []
            dim_map_child2[dim_name].append(i)

        matching_pairs = []
        # child1의 각 인덱스에 대해 동일 차원 이름을 child2에서 찾는다
        for i in range(start, end):
            dim_name = child1[i][0]
            if dim_name in dim_map_child2:
                for j in dim_map_child2[dim_name]:
                    matching_pairs.append((i, j))

        return matching_pairs
    
    def crossover_tile(self, parents, crossover_prob=0.7):
        parent1, parent2 = parents
        offspring = []
        crossover_success = False
        
        if random.random() < crossover_prob:
            idx1, idx2 = sorted(random.sample(range(1, 7), 2))
            idx3, idx4 = sorted(random.sample(range(8, 14), 2))
            
            child1 = [gene[:] for gene in parent1]
            child2 = [gene[:] for gene in parent2]

            # Perform L2 crossover and check constraints
            child1[idx1][1], child2[idx2][1] = parent2[idx1][1], parent1[idx2][1]
            if not (self.check_constraints(child1[:7], level='l2') and self.check_constraints(child2[:7], level='l2')):
                child1[idx1][1], child2[idx2][1] = parent1[idx1][1], parent2[idx2][1]
            
            # Perform L1 crossover and check constraints
            child1[idx3][1], child2[idx4][1] = parent2[idx3][1], parent1[idx4][1]
            if not (self.check_constraints(child1[7:], level='l1') and self.check_constraints(child2[7:], level='l1')):
                child1[idx3][1], child2[idx4][1] = parent1[idx3][1], parent2[idx4][1]
            
            offspring.extend([child1, child2])
        else:
            offspring.extend([parent1, parent2])

        return offspring
    
    def crossover_dim(self, parents, crossover_prob=0.7):
        parent1, parent2 = parents
        offspring = []
        
        if random.random() < crossover_prob:
            child1 = [gene[:] for gene in parent1]
            child2 = [gene[:] for gene in parent2]

            # 가능한 차원 쌍 후보를 수집
            # L2 범위: 1~6, L1 범위: 8~13
            l2_dim_pairs = self.get_matching_dim_pairs(child1, child2, start=1, end=7)
            l1_dim_pairs = self.get_matching_dim_pairs(child1, child2, start=8, end=14)

            def perform_crossover(mapper1, mapper2, dim_pairs): # 원본데이터에 바로 반영됨
                if dim_pairs: # [(idx1_child1, idx2_child2), ...]
                    idx1, idx2 = random.choice(dim_pairs)
                    mapper1[idx1], mapper2[idx2] = mapper2[idx1], mapper1[idx2]

            perform_crossover(child1, child2, l2_dim_pairs)
            perform_crossover(child1, child2, l1_dim_pairs)
            
            # Check child1 validity
            if self.check_constraints(child1[:7], level='l2') and self.check_constraints(child1[7:], level='l1'):
                offspring.append(child1)
            else:
                offspring.append(parent1)
                
            # Check child2 validity
            if self.check_constraints(child2[:7], level='l2') and self.check_constraints(child2[7:], level='l1'):
                offspring.append(child2)
            else:
                offspring.append(parent2)

        else:
            offspring.extend([parent1, parent2])
        return offspring
    
    def mutate_tiles(self, population, mutation_prob=0.2):
        available_tile_sizes = self.get_available_tile_sizes()
        dim_list = ['K', 'C', 'Y', 'X', 'R', 'S']
        max_attempts = 10
        
        for mapper in population:
            original_mapper = [gene[:] for gene in mapper]  # Save the original state
            
            dynamic_mutation_prob = min(0.2 + (1 - self.best_perf_fitness1) * 0.5, 0.8)
            if random.random() < dynamic_mutation_prob:
                mutation_success = False
                attempts = 0

                while not mutation_success and attempts < max_attempts:
                    idx_l2 = random.randint(1, 6)
                    idx_l1 = random.randint(8, 13)
                    
                    # L2 Mapper mutation
                    valid_factors_l2 = available_tile_sizes['l2'][mapper[idx_l2][0]]
                    if mapper[idx_l2][0] in dim_list:
                        mapper[idx_l2][1] = random.choice(valid_factors_l2)
                    
                    # L1 Mapper mutation]
                    valid_factors_l1 = available_tile_sizes['l1'][mapper[idx_l1][0]]
                    if mapper[idx_l1][0] in dim_list:
                        mapper[idx_l1][1] = random.choice(valid_factors_l1)

                    # Check constraints
                    if self.check_constraints(mapper[:7], level='l2') and \
                    self.check_constraints(mapper[7:], level='l1'):
                        mutation_success = True
                    else:
                        mapper[idx_l2][1] = original_mapper[idx_l2][1]
                        mapper[idx_l1][1] = original_mapper[idx_l1][1]
                        attempts += 1

                if not mutation_success:
                    # 최종적으로 실패 시 원복
                    mapper[:] = original_mapper

        return population

    def mutate_pods(self, population, mutation_prob=0.2):
        available_pod_sizes = self.get_available_pod_sizes()
        max_attempts = 10
        idx1 = 7  # Pod index
        
        for mapper in population:
            original_mapper = [gene[:] for gene in mapper]  # Save the original state
            dynamic_mutation_prob = min(0.2 + (1 - self.best_perf_fitness1) * 0.5, 0.8)
            
            if random.random() < dynamic_mutation_prob:
                mutation_success = False
                attempts = 0
                
                # Loop until a valid mutation is found
                while not mutation_success and attempts < max_attempts:
                    mapper[idx1][1] = random.choice(available_pod_sizes)

                    # Constraints validation
                    if self.check_constraints(mapper[:7], level='l2') and \
                    self.check_constraints(mapper[7:], level='l1'):
                        mutation_success = True
                    attempts += 1

                    if not mutation_success:
                        mapper[idx1][1] = original_mapper[idx1][1]

        return population
    
    def replace_least_fit_individuals(self, population, offspring, pop_fitness_score_list):
        """
            Replace the least fit individuals in the population with the offspring
        """
        sorted_indices = sorted(range(len(pop_fitness_score_list)), key=lambda x: pop_fitness_score_list[x])
        for i in range(len(offspring)):
            worst_idx = sorted_indices[i]
            population[worst_idx] = offspring[i]
        return population
    
    def restart_population(self, population, restart_prob=0.2):
        """
        Reinitialize a portion of the population if restart_prob is met.
        """
        if random.random() < restart_prob:
            num_to_restart = int(len(population) * 0.3)  # Restart 30% of the population
            new_individuals = [self.create_chromosome() for _ in range(num_to_restart)]
            population[-num_to_restart:] = new_individuals
            print(f"Restarted {num_to_restart} individuals due to stagnation.")
        return population
    
    def run_ga(self, dimensions):
        self.dimensions = dimensions
        population = self.initialize_population()
        best_fitness, best_idx = -1, -1
        
        result_csv = f"{self.model_name}_mapping.csv"
        if os.path.exists(result_csv):
            os.remove(result_csv)

        no_improvement_generations = 0
        last_best_fitness = -float('inf')
        
        pop_fitness_score_list, pop_performance_list = self.evaluate_population(population)
        best_fitness, best_idx = self.find_best_score(pop_fitness_score_list)
        best_perf_fitness1, best_perf_fitness2 = pop_performance_list[best_idx]
        elite_individual = population[best_idx]
        
        for i in range(self.max_generations): # Number of generations
            print(f"Generation {i + 1}......")
            
            # Step0: Track stagnation and apply restart if needed
            if best_fitness <= last_best_fitness:
                no_improvement_generations += 1
                print(f"No improvement for {no_improvement_generations} generations.")
            else:
                no_improvement_generations = 0
                last_best_fitness = best_fitness
            if no_improvement_generations >= 3:  # Stagnation threshold
                population = self.restart_population(population, restart_prob=0.5)
                no_improvement_generations = 0
            
            # Step1: Select parents
            selected_parents = []
            for _ in range(len(population) // 2):
                parents = self.select_parents(population, pop_fitness_score_list)
                selected_parents.append(parents)
            
            # Step2: Crossover
            offspring = []
            for parent1, parent2 in selected_parents:
                # if random.random() < 0.5:
                children = self.crossover_tile([parent1, parent2], crossover_prob=0.8)
                # else:
                children = self.crossover_dim([parent1, parent2], crossover_prob=0.7)
                offspring.extend(children)
            
            # Step3: Mutation
            self.mutate_tiles(offspring, mutation_prob=0.5)
            self.mutate_pods(offspring, mutation_prob=0.5)
            
            # Step4: Replace least fit individual with new offspring
            self.replace_least_fit_individuals(population, offspring, pop_fitness_score_list)
            print("\telite_individual", elite_individual)
            population[0] = elite_individual

            pop_fitness_score_list, pop_performance_list = self.evaluate_population(population)
            best_fitness, best_idx = self.find_best_score(pop_fitness_score_list)
            best_perf_fitness1, best_perf_fitness2 = pop_performance_list[best_idx]
            elite_individual = population[best_idx]

            self.best_fitness = best_fitness
            self.best_perf_fitness1 = best_perf_fitness1
            self.best_perf_fitness2 = best_perf_fitness2
            print('\t', len(population), 'individuals in the population')
            print('\t', len(offspring), 'individuals in the offspring')
            print(f'\tBest fitness score: {best_fitness}, Best Fitness1: {best_perf_fitness1}', f'Best Fitness2: {best_perf_fitness2}')

        self.save_to_csv({
            'Best Fitness Score': self.best_fitness_score_list,
            f'Best {self.fitness1}': self.best_perf_fitness1_list,
            f'Best {self.fitness2}': self.best_perf_fitness2_list
        }, self.model_name)

        return population[best_idx]

if __name__ == "__main__":
    model_name = 'single_layer_conv' # {'single_layer_conv', 'single_layer_gemm', 'single_layer_fc', 'multi_layers'}
    hardware_config = 'accelerator_edge' # {'accelerator_edge'}
    dnn_layer_dict = extract_dimensions(f'data/model/{model_name}.m')
    
    model = GAMAPPER(
        hw_config_file=hardware_config,
        model_name=model_name,
        fitness=['utilization', 'edp'],
        popluation_size=10,
        max_generations=5,
        alpha=0.7, beta=0.3
    )
    best_mapper = model.run_ga(dimensions=dnn_layer_dict[0])

    # Run the best individual
    # best_mapper = population[best_idx]
    final_mapping_file_path = model.create_mapping_file([best_mapper], dimensions=dnn_layer_dict[0], model_name=f'final_result_{model_name}')
    final_result_csv = model.run_maestro_to_get_all_metrics(final_mapping_file_path)
    
    print("Best mapper: ", best_mapper)
    print(f'Final mapping file saved in {final_mapping_file_path}') # data/mapping/final_result_single_layer_conv.m