import subprocess
import os
import shutil
import pandas as pd
import math
import random
import numpy as np
import csv
import re
import argparse
from functools import reduce
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import StandardScaler
from datetime import datetime
    
class MAGNETO:
    def __init__(self, hw_config='', model_name='', popluation_size=100, max_generations=10, power_constraint=1000):
        super(MAGNETO,self).__init__()
        self.hw_config = hw_config
        self.model_name = model_name
        self.num_PEs = 0
        with open(f'data/hw/{self.hw_config}.m') as f:
            for line in f:
                if 'num_pes' in line:
                    self.num_PEs = int(line.split(":")[1])
                    print(f"Number of PEs: {self.num_PEs}")

        self.dimensions = {}
        self.popluation_size = popluation_size
        self.max_generations = max_generations
        self.power_constraint = power_constraint
        
        self.best_fitness = -1
        self.best_perf_fitness1 = float('inf') # latency
        self.best_perf_fitness2 = 0 # utilization
        self.best_perf_fitness3 = float('inf') # memory access
        self.best_perf_fitness4 = 0 # reuse factor
        
        self.max_latency = None
        self.max_weighted_memory_access = None
        self.max_averaged_reuse_factor = None
        self.max_edp = None
        
        self.best_fitness_score_list = [] # [fitness score] * generation size
        self.best_perf_fitness1_list = [] # [metric1 value] * generation size
        self.best_perf_fitness2_list = []
        self.best_perf_fitness3_list = []
        self.best_perf_fitness4_list = []
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

    def evaluate_fitness_from_metrics(self, csv_file_path, population):
        try:
            df = pd.read_csv(csv_file_path)
            
            pop_fitness_score_list = [] # [calculated fitness score * population_size]
            pop_performance_list = [] # [(fitness1 value, fitness2 value) * population_size]

            pop_fitness1_list = [] # [fitness1 value * population_size]
            pop_fitness2_list = [] # [fitness2 value * population_size]
            pop_fitness3_list = [] # [fitness3 value * population_size]
            pop_fitness4_list = [] # [fitness4 value * population_size]
            pop_normalized_fitness1_list = [] # [fitness1 norm value * population_size]
            pop_normalized_fitness2_list = [] # [fitness2 norm value * population_size]
            pop_normalized_fitness3_list = [] # [fitness3 norm value * population_size]
            pop_normalized_fitness4_list = [] # [fitness4 norm value * population_size]
            
            power_over_constraint_count = 0
            for i, row in df.iterrows():
                freq_MHz = 350 # FIXME: Hardcoded
                latency = row[' Runtime (Cycles)']
                energy_nJ = row[' Activity count-based Energy (nJ)']
                edp = latency * energy_nJ / (freq_MHz * 1e6)
                utilization = row['Avg number of utilized PEs'] / self.num_PEs
                l1_access = row[' input l1 read'] + row[' input l1 write'] + row['filter l1 read'] + row[' filter l1 write'] + row['output l1 read'] + row[' output l1 write']
                l2_access = row[' input l2 read'] + row[' input l2 write'] + row[' filter l2 read'] + row[' filter l2 write'] + row[' output l2 read'] + row[' output l2 write']
                dram_access = row[' Offchip BW Req (Elements/cycle)'] * row[' Runtime (Cycles)']
                
                weighted_memory_access = l1_access + 2*l2_access + 6*dram_access
                averaged_reuse_factor = (row[' input reuse factor'] + row[' filter reuse factor'] + row[' output reuse factor']) / 3 # 0~1000
                
                power_proxy_value = self.get_power_proxy_value(population[i])
                
                # Normalize the values
                if self.max_latency is None:
                    self.max_latency = latency
                else: # Update the max latency
                    self.max_latency = max(self.max_latency, latency)
                if self.max_weighted_memory_access is None:
                    self.max_weighted_memory_access = weighted_memory_access
                else:
                    self.max_weighted_memory_access = max(self.max_weighted_memory_access, weighted_memory_access)
                if self.max_averaged_reuse_factor is None:
                    self.max_averaged_reuse_factor = averaged_reuse_factor
                else:
                    self.max_averaged_reuse_factor = max(self.max_averaged_reuse_factor, averaged_reuse_factor)
                if self.max_edp is None:
                    self.max_edp = edp
                else:
                    self.max_edp = max(self.max_edp, edp)

                norm_latency = np.log(latency + 1) / np.log(self.max_latency + 1)
                # norm_latency = latency / self.max_latency
                norm_utilization = utilization
                # norm_weighted_memory_access = np.log(weighted_memory_access + 1) / np.log(self.max_weighted_memory_access + 1)
                
                norm_weighted_memory_access = (weighted_memory_access - 1) / (self.max_weighted_memory_access + 1)
                norm_averaged_reuse_factor = averaged_reuse_factor / self.max_averaged_reuse_factor
                norm_edp = edp / self.max_edp
                
                # Append raw performance values
                pop_fitness1_list.append(latency)
                pop_fitness2_list.append(edp)
                pop_fitness3_list.append(weighted_memory_access)
                pop_fitness4_list.append(averaged_reuse_factor)
                pop_performance_list.append((latency, edp, weighted_memory_access, averaged_reuse_factor))
                
                # Append normalized performance values for calculating fitness score below
                pop_normalized_fitness1_list.append(norm_latency)
                pop_normalized_fitness2_list.append(norm_edp)
                pop_normalized_fitness3_list.append(norm_weighted_memory_access)
                pop_normalized_fitness4_list.append(norm_averaged_reuse_factor)

                offset = 5
                # lambda_val = 0.7
                fitness_score = 0.3 * norm_averaged_reuse_factor - 0.5 * norm_edp - 0.2 * norm_weighted_memory_access
                # fitness_score = lambda_val * (1 - norm_latency) + (1 - lambda_val) * energy_efficiency_metric + offset
                if power_proxy_value > self.power_constraint:
                    # print(f"Power constraint violated: {power_proxy_value}")
                    power_over_constraint_count += 1
                    fitness_score = -100
                pop_fitness_score_list.append(fitness_score)

            best_idx = np.argmax(pop_fitness_score_list)
            self.best_fitness_score_list.append(pop_fitness_score_list[best_idx])
            self.best_perf_fitness1_list.append(pop_fitness1_list[best_idx])
            self.best_perf_fitness2_list.append(pop_fitness2_list[best_idx])
            self.best_perf_fitness3_list.append(pop_fitness3_list[best_idx])
            self.best_perf_fitness4_list.append(pop_fitness4_list[best_idx])
            self.best_mapper_list.append(population[best_idx])
            
            print(f"power_over_constraint_count: {power_over_constraint_count} / {self.popluation_size}")
            # print("\tpop_normalized_fitness1_list:", pop_normalized_fitness1_list)
            # print("\tpop_normalized_fitness2_list:", pop_normalized_fitness2_list)
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
        pop_fitness_score_list, pop_performance_list = self.evaluate_fitness_from_metrics(metric_csv, population)
        return pop_fitness_score_list, pop_performance_list

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

            print(f"Extracted Dimensions: {dimensions_list}")
            return dimensions_list if dimensions_list else [{}]

        except FileNotFoundError:
            print(f"File {file_path} not found.")
            return [{}]

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
        # best_score = min(pop_fitness_score_list)
        best_idx = pop_fitness_score_list.index(best_score)
        return best_score, best_idx

    def create_chromosome(self):
        l2_mapper = self.encode_mapper(self.dimensions, level='l2')
        l1_mapper = self.encode_mapper(self.dimensions, level='l1')
        return l2_mapper + l1_mapper

    def create_dataflow_template_string(self, dimensions, mapper):
        dataflow_template = ''
        dataflow_template += "\t\tDataflow {\n"

        l2_mapper, l1_mapper = mapper[:7], mapper[7:]

        for i, mapping in enumerate([l2_mapper, l1_mapper]):
            if i == 0: # L2
                for j, (dim, tile_size) in enumerate(mapping[1:]):
                    if j == 0:
                        if dim == 'R':
                            dataflow_template += f"\t\t\tSpatialMap(Sz(R),Sz(R)) R;\n"
                        elif dim == 'S':
                            dataflow_template += f"\t\t\tSpatialMap(Sz(S),Sz(S)) S;\n"
                        else:
                            dataflow_template += f"\t\t\tSpatialMap({tile_size},{tile_size}) {dim};\n"
                    else:
                        if dim == 'R':
                            dataflow_template += f"\t\t\tTemporalMap(Sz(R),Sz(R)) R;\n"
                        elif dim == 'S':
                            dataflow_template += f"\t\t\tTemporalMap(Sz(S),Sz(S)) S;\n"
                        else:
                            dataflow_template += f"\t\t\tTemporalMap({tile_size}, {tile_size}) {dim};\n"
            else: # L1
                for j, (dim, tile_size) in enumerate(mapping):
                    if j == 0:
                        dataflow_template += f"\t\t\tCluster({tile_size}, {dim});\n"
                    elif j == 1:
                        if dim == 'R':
                            dataflow_template += f"\t\t\tSpatialMap(Sz(R),Sz(R)) R;\n"
                        elif dim == 'S':
                            dataflow_template += f"\t\t\tSpatialMap(Sz(S),Sz(S)) S;\n"
                        else:
                            dataflow_template += f"\t\t\tSpatialMap({tile_size}, {tile_size}) {dim};\n"
                    else:
                        if dim == 'R':
                            dataflow_template += f"\t\t\tTemporalMap(Sz(R),Sz(R)) R;\n"
                        elif dim == 'S':
                            dataflow_template += f"\t\t\tTemporalMap(Sz(S),Sz(S)) S;\n"
                        else:
                            dataflow_template += f"\t\t\tTemporalMap({tile_size}, {tile_size}) {dim};\n"
        dataflow_template += "\t\t}\n"
        return dataflow_template

    def create_mapping_file(self, population, dimensions, model_name):
        """
            Create mapping files for only an evaluation
            Parameters:
                population: List of mappers
                dimensions: Dictionary of DNN layer dimensions, [{'K': 256, 'C': 128, 'R': 3, 'S': 3, 'Y': 14, 'X': 14}, {...}, ...]
                model_name: Name of the model that is
        """
        mapping_file_path = f"data/mapping/{model_name}_mapping.m"
        
        mapping_template = f"Network {model_name} {{\n"
        dataflow_template = ''

        for idx, mapper in enumerate(population):
            dataflow_template += f"\tLayer Layer{idx} {{\n"
            dataflow_template += f"\t\tType: CONV\n"
            dimensions_str = ', '.join([f"{k}:{v}" for k, v in dimensions.items()])
            dataflow_template += f"\t\tDimensions {{ {dimensions_str} }}\n"
            dataflow_template += self.create_dataflow_template_string(dimensions, mapper)
            dataflow_template += "\t}\n"
        
        mapping_template += f"{dataflow_template}}}\n"

        with open(mapping_file_path, 'w') as f:
            f.write(mapping_template)
        
        return mapping_file_path
    
    def integrate_dataflow_in_model(self, dimensions, model_name, mapper, is_temp=False):
        if is_temp:
            output_model_path = f"temp_magneto_{model_name}.m"
        else:
            output_model_path = f"data/mapping/magneto_{model_name}.m"

        dataflow_template = self.create_dataflow_template_string(dimensions, mapper)

        # Read and process the model file
        with open(f'data/model/{model_name}.m', "r") as infile:
            model_lines = infile.readlines()

        output_lines = []
        for line in model_lines:
            output_lines.append(line)
            if "Dimensions {" in line:
                # Add Dataflow string after the Dimensions block
                output_lines.append(dataflow_template)

        # Write the model with mapper to the output file
        with open(output_model_path, "w") as outfile:
            outfile.writelines(output_lines)

        return output_model_path

    def get_power_proxy_value(self, mapper): # mapping: [['K',14], ..]
        temp_model_with_mapping_m = self.integrate_dataflow_in_model(self.dimensions, self.model_name, mapper, is_temp=True)
        interim_result_csv = model.run_maestro_to_get_all_metrics(temp_model_with_mapping_m)

        freq_MHz = 300
        energy_nJ = 0
        cycles = 0
        dram_access_energy_nJ = 0
        
        df = pd.read_csv(interim_result_csv)
        # column_sums = df.sum()
        for i, row in df.iterrows():
            energy_nJ += row[' Activity count-based Energy (nJ)']
            cycles += row[' Runtime (Cycles)']
            # 16 bits, 30 pJ per bit
            dram_access_energy_nJ += (row[' Offchip BW Req (Elements/cycle)'] * row[' Runtime (Cycles)']) * 2*8*30 

        power_proxy_value = (energy_nJ + dram_access_energy_nJ) / cycles * freq_MHz * 1e-9 * 1e6
        
        if os.path.exists(temp_model_with_mapping_m):
            os.remove(temp_model_with_mapping_m)
        if os.path.exists(interim_result_csv):
            os.remove(interim_result_csv)
        
        return power_proxy_value
    
    def save_to_csv(self, lists_dict, layer_name):
        """
            Sample lists_dict: {'Best Fitness Score': [1, 2, 3], 'Best Utilization': [0.1, 0.2, 0.3], 'Best Mapping': [[], [], []]}
        """
        # Get current date and time
        current_time = datetime.now().strftime("%Y_%m_%d_%H:%M")
        filename = f'fitness_{layer_name}_{current_time}.csv'

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
            
            dynamic_mutation_prob = min(0.2 + (1 - self.best_perf_fitness1), 0.8)
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
            dynamic_mutation_prob = min(0.2 + (1 - self.best_perf_fitness1), 0.8)
            
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
            num_to_restart = int(len(population) * 0.5)  # Restart 50% of the population
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

        # Evaluate the initial population
        pop_fitness_score_list, pop_performance_list = self.evaluate_population(population)
        best_fitness, best_idx = self.find_best_score(pop_fitness_score_list)
        best_perf_fitness1, best_perf_fitness2, best_perf_fitness3, best_perf_fitness4 = pop_performance_list[best_idx]

        for i in range(self.max_generations):
            print(f"Generation {i + 1}......")

            elite_individual = population[best_idx]
            population[0] = elite_individual
            # print(f"Elite individual: {elite_individual}")

            # Step0: Track stagnation and apply restart if needed
            if best_fitness <= last_best_fitness:
                no_improvement_generations += 1
                # print(f"No improvement for {no_improvement_generations} generations.")
            else:
                no_improvement_generations = 0
                last_best_fitness = best_fitness
            if no_improvement_generations >= 3:  # Stagnation threshold
                population = self.restart_population(population, restart_prob=0.8)
                no_improvement_generations = 0
                # population[0] = elite_individual
            
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
            population[0] = elite_individual

            # Evaluate the new populations
            pop_fitness_score_list, pop_performance_list = self.evaluate_population(population)
            best_fitness, best_idx = self.find_best_score(pop_fitness_score_list)
            best_perf_fitness1, best_perf_fitness2, best_perf_fitness3, best_perf_fitness4 = pop_performance_list[best_idx]

            self.best_fitness = best_fitness
            self.best_perf_fitness1 = best_perf_fitness1
            self.best_perf_fitness2 = best_perf_fitness2
            self.best_perf_fitness3 = best_perf_fitness3
            self.best_perf_fitness4 = best_perf_fitness4
            # print('\t', len(population), 'individuals in the population')
            # print('\t', len(offspring), 'individuals in the offspring')
            print(f'\tBest fitness score: {best_fitness:.4f}, Best Latency: {best_perf_fitness1:.4f}, Best Utilization: {best_perf_fitness2:.4f}, Best Memory Access: {best_perf_fitness3:.4f}, Best Reuse Factor: {best_perf_fitness4:.4f}, \n \tBest Mapper: {population[best_idx]}')

        self.save_to_csv({
            'Best Fitness Score': self.best_fitness_score_list,
            f'Best Latency': self.best_perf_fitness1_list,
            f'Best Utilization': self.best_perf_fitness2_list,
            f'Best Memory Access': self.best_perf_fitness3_list,
            f'Best Reuse Factor': self.best_perf_fitness4_list,
            f'Best Mapper': self.best_mapper_list,
        }, self.model_name)

        print("\tBest mapper: ", population[best_idx])
        return population[best_idx]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--generation', type=int, default=10, help='Number of generations')
    parser.add_argument('--population', type=int, default=10, help='Number of populations')
    parser.add_argument('--model', type=str, default='single_conv', help='Model defined in data/model in MAESTO format')
    parser.add_argument('--hwconfig', type=str, default='mobile', choices=('mobile', 'cloud'), help='Hardware config defined in data/hw in MAESTRO format')
    parser.add_argument('--power_budget_mw', type=int, default='1000', help='Power budget in mW')
    args = parser.parse_args()
    current_time = datetime.now().strftime("%Y_%m_%d")
    
    # Run the Genetic Algorithm
    model = MAGNETO(
        hw_config=args.hwconfig,
        model_name=args.model,
        popluation_size=args.population,
        max_generations=args.generation,
        power_constraint=args.power_budget_mw
    )
    dnn_layer_dict = model.extract_dimensions(f'data/model/{args.model}.m')
    best_mapper = model.run_ga(dimensions=dnn_layer_dict[0])
    
    # Integrate the best mapper in the entire model
    final_mapping_file_path = model.integrate_dataflow_in_model(dnn_layer_dict[0], args.model, best_mapper)
    final_result_csv = model.run_maestro_to_get_all_metrics(final_mapping_file_path)

    print("Optimal mapper: ", best_mapper)
    print(f'Final mapping file saved in {final_mapping_file_path}') # data/mapping/final_result_single_conv.m