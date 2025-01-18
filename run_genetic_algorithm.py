import subprocess
import os
import pandas as pd
import random
import numpy as np
import csv
import re
import argparse
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
        
        self.best_fitness = -1
        self.best_perf_fitness1 = float('inf') # latency
        self.best_perf_fitness2 = 0 # utilization
        self.best_perf_fitness3 = float('inf') # memory access
        self.best_perf_fitness4 = 0 # reuse factor
        
        self.max_cycles = None
        self.max_weighted_memory_access = None
        self.max_averaged_reuse_factor = None
        self.max_edp = None
        self.energy_nJ = None
        
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
        data = df.to_numpy()  # Convert entire DataFrame to NumPy array for efficient processing

        # Column indices based on CSV structure (replace with actual indices as necessary)
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
        fitness_score = - 0.6 * norm_edp - 0.4 * norm_weighted_memory_access + offset # reuse_factor increases as memory access decreases
        if power_proxy_value > self.power_constraint:
            fitness_score = -100

        # Cleanup temporary files
        if os.path.exists(temp_model_with_mapping_m):
            os.remove(temp_model_with_mapping_m)
        if os.path.exists(temp_result_csv):
            os.remove(temp_result_csv)

        return fitness_score, (cycles, edp, weighted_memory_access, averaged_reuse_factor)

    def evaluate_population(self, population):
        pop_fitness_score_list = [] # [calculated fitness score * population_size]
        pop_performance_list = [] # [(fitness1 value, fitness2 value) * population_size]

        for chromosome in population:
            fitness_score, performance = self.evaluate_chromosome(chromosome)
            pop_fitness_score_list.append(fitness_score)
            pop_performance_list.append(performance)
        
        best_idx = np.argmax(pop_fitness_score_list)
        self.best_fitness_score_list.append(pop_fitness_score_list[best_idx])
        self.best_perf_fitness1_list.append(pop_performance_list[best_idx][0]) # cycles
        self.best_perf_fitness2_list.append(pop_performance_list[best_idx][1]) # edp
        self.best_perf_fitness3_list.append(pop_performance_list[best_idx][2]) # weighted memory access
        self.best_perf_fitness4_list.append(pop_performance_list[best_idx][3]) # averaged reuse factor
        self.best_mapper_list.append(population[best_idx])
            
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
        valid = True
        if pod[0][0] == 1:
            # print("Cluster size cannot be 1.")
            valid = False
        if pod > self.num_PEs:
            valid = False
        return valid
    
    def check_constraints_mapper(self, mapper, level): # single l1 or l2 mapper [[dim, tile_size], ...]
        """
            Single level of a mapper
        """
        valid = True
        dim_list = ['K', 'C', 'Y', 'X', 'R', 'S']
        
        # Check if 'P' is fixed as the first dimension
        # if mapper[0][0] != 'P':
        #     # print("First gene must be 'P'")
        #     valid = False
            
        # Check if there are duplicate dimensions
        dimensions = [gene[0] for gene in mapper]
        if len(dimensions) != len(set(dimensions)):
            # print(f"Duplicate dimensions detected: {dimensions}")
            valid = False
        
        # Check if there is X or Y in L2 mapper index 0
        if level == 'l2':
            if mapper[0][0] in {'X', 'Y'}:
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
            # cluster_size = mapper[0][1]  # Cluster 크기 (m 값)
            spatial_size = mapper[0][1]  # SpatialMap 크기 (n 값)
            
            # 1. PE 수 초과 확인
            # if cluster_size > self.num_PEs or cluster_size <= 0:
            #     # print(f"Invalid Cluster size: {cluster_size}. Must be within PE limit.")
            #     valid = False

            # 2. Cluster 크기 >= SpatialMap 크기 확인
            # if cluster_size < spatial_size:
            #     # print(f"SpatialMap size {spatial_size} cannot exceed Cluster size {cluster_size}.")
            #     valid = False
            
            # 3. 유효한 매핑 차원 확인
            # if mapper[0][0] != 'P':
            #     # print(f"Invalid mapping dimensions: {mapper[0][0]}. Expected 'P'.")
            #     valid = False

            # 4. PE가 1개일 경우 비허용
            # if cluster_size == 1:
            #     # print("Cluster size cannot be 1.")
            #     valid = False
        
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

    def find_best_score(self, pop_fitness_score_list):
        best_score = max(pop_fitness_score_list)
        best_idx = pop_fitness_score_list.index(best_score)
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

    def crossover_tile(self, parents, crossover_prob=0.7, max_retries=5):
        parent1, parent2 = parents
        child1, child2 = None, None

        if random.random() < crossover_prob:
            retries = 0
            valid_child1, valid_child2 = False, False

            while retries < max_retries:
                # Create temporary children for retry
                temp_child1 = [gene[:] for gene in parent1]
                temp_child2 = [gene[:] for gene in parent2]

                idx1, idx2 = sorted(random.sample(range(0, 6), 2))
                idx3, idx4 = sorted(random.sample(range(7, 13), 2))

                # Perform L2 crossover
                temp_child1[idx1][1], temp_child2[idx2][1] = parent2[idx1][1], parent1[idx2][1]
                valid_l2 = self.check_constraints_mapper(temp_child1[:6], level='l2') and self.check_constraints_mapper(temp_child2[:6], level='l2')
                if not valid_l2:
                    temp_child1[idx1][1], temp_child2[idx2][1] = parent1[idx1][1], parent2[idx2][1]  # Revert

                # Perform L1 crossover
                temp_child1[idx3][1], temp_child2[idx4][1] = parent2[idx3][1], parent1[idx4][1]
                valid_l1 = self.check_constraints_mapper(temp_child1[7:], level='l1') and self.check_constraints_mapper(temp_child2[7:], level='l1')
                if not valid_l1:
                    temp_child1[idx3][1], temp_child2[idx4][1] = parent1[idx3][1], parent2[idx4][1]  # Revert

                # Validate both children
                valid_child1 = valid_l2 and valid_l1
                valid_child2 = valid_l2 and valid_l1

                if valid_child1 or valid_child2:
                    child1 = temp_child1 if valid_child1 else parent1
                    child2 = temp_child2 if valid_child2 else parent2
                    break  # Exit retry loop if valid children are created

                retries += 1

            child1 = parent1 if not valid_child1 else child1
            child2 = parent2 if not valid_child2 else child2
            return child1, child2
        else:
            return parent1, parent2

    def crossover_dim(self, parents, crossover_prob=0.7, max_retries=5):
        parent1, parent2 = parents
        child1, child2 = None, None
        
        if random.random() < crossover_prob:
            child1 = [gene[:] for gene in parent1]
            child2 = [gene[:] for gene in parent2]

            # 가능한 차원 쌍 후보를 수집
            # L2 범위: 1~6, L1 범위: 8~13
            l2_dim_pairs = self.get_matching_dim_pairs(child1, child2, start=0, end=6)
            l1_dim_pairs = self.get_matching_dim_pairs(child1, child2, start=7, end=13)

            def perform_crossover(mapper1, mapper2, dim_pairs): # 원본데이터에 바로 반영됨
                if dim_pairs: # [(idx1_child1, idx2_child2), ...]
                    idx1, idx2 = random.choice(dim_pairs)
                    mapper1[idx1], mapper2[idx2] = mapper2[idx1], mapper1[idx2]

            retries = 0
            valid_child1, valid_child2 = False, False

            while retries < max_retries:
                temp_child1 = [gene[:] for gene in parent1]
                temp_child2 = [gene[:] for gene in parent2]

                perform_crossover(temp_child1, temp_child2, l2_dim_pairs)
                perform_crossover(temp_child1, temp_child2, l1_dim_pairs)

                # Validate temp_child1
                valid_child1 = (
                    self.check_constraints_mapper(temp_child1[:6], level='l2') and
                    self.check_constraints_mapper(temp_child1[7:], level='l1')
                )
                # Validate temp_child2
                valid_child2 = (
                    self.check_constraints_mapper(temp_child2[:6], level='l2') and
                    self.check_constraints_mapper(temp_child2[7:], level='l1')
                )

                if valid_child1 or valid_child2:
                    child1 = temp_child1 if valid_child1 else child1
                    child2 = temp_child2 if valid_child2 else child2
                    break
                retries += 1
            
            child1 = parent1 if not valid_child1 else child1
            child2 = parent2 if not valid_child2 else child2
            return child1, child2
        else:
            return parent1, parent2
    
    def mutate_tiles(self, chromosome, mutation_prob=0.2):
        available_tile_sizes = self.get_available_tile_sizes()
        dim_list = ['K', 'C', 'Y', 'X', 'R', 'S']
        max_attempts = 5
        
        original_mapper = [gene[:] for gene in chromosome]  # Save the original state
        
        if random.random() < mutation_prob:
            mutation_success = False
            attempts = 0

            while not mutation_success and attempts < max_attempts:
                idx_l2 = random.randint(0, 5)
                idx_l1 = random.randint(7, 12)
                
                # L2 Mapper mutation
                valid_factors_l2 = available_tile_sizes['l2'][chromosome[idx_l2][0]]
                if chromosome[idx_l2][0] in dim_list:
                    chromosome[idx_l2][1] = random.choice(valid_factors_l2)
                
                # L1 Mapper mutation
                valid_factors_l1 = available_tile_sizes['l1'][chromosome[idx_l1][0]]
                if chromosome[idx_l1][0] in dim_list:
                    chromosome[idx_l1][1] = random.choice(valid_factors_l1)

                # Check constraints
                if self.check_constraints_mapper(chromosome[:6], level='l2') and \
                self.check_constraints_mapper(chromosome[7:], level='l1'):
                    mutation_success = True
                else:
                    chromosome[idx_l2][1] = original_mapper[idx_l2][1]
                    chromosome[idx_l1][1] = original_mapper[idx_l1][1]
                    attempts += 1

            if not mutation_success:
                chromosome[:] = original_mapper

        return chromosome
    
    def mutate_shuffle(self, chromosome, mutation_prob=0.2):
        if random.random() < mutation_prob:
            l2_mapper = chromosome[:6]
            random.shuffle(l2_mapper)
        else:
            l1_mapper = chromosome[7:]
            random.shuffle(l1_mapper)
        return chromosome

    def mutate_pods(self, chromosome, mutation_prob=0.2):
        available_pod_sizes = self.get_available_pod_sizes()
        max_attempts = 5
        idx1 = 6  # Pod index
        
        original_mapper = [gene[:] for gene in chromosome]  # Save the original state
        
        if random.random() < mutation_prob:
            mutation_success = False
            attempts = 0
            
            # Loop until a valid mutation is found
            while not mutation_success and attempts < max_attempts:
                chromosome[idx1][1] = random.choice(available_pod_sizes)

                # Constraints validation
                if self.check_constraints_mapper(chromosome[:6], level='l2') and \
                self.check_constraints_mapper(chromosome[7:], level='l1'):
                    mutation_success = True
                attempts += 1

                if not mutation_success:
                    chromosome[idx1][1] = original_mapper[idx1][1]

        return chromosome
    
    # def replace_least_fit_individuals(self, population, offspring, pop_fitness_score_list):
    #     """
    #         Replace the least fit individuals in the population with the offspring
    #     """
    #     sorted_indices = sorted(range(len(pop_fitness_score_list)), key=lambda x: pop_fitness_score_list[x])
    #     for i in range(len(offspring)):
    #         worst_idx = sorted_indices[i]
    #         population[worst_idx] = offspring[i]
    #     return population
    
    def replace_population(self, population, fitness_scores, elite_ratio=0.1, random_ratio=0.2):
        """
        - 상위 elite_ratio(%) 개체는 항상 유지 (엘리트)
        - 나머지 개체 중에서 일부(random_ratio%)를 랜덤 선택해 새로 초기화
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
        pop_fitness_score_list, pop_performance_list = self.evaluate_population(population)
        best_fitness, best_idx = self.find_best_score(pop_fitness_score_list)
        best_perf_fitness1, best_perf_fitness2, best_perf_fitness3, best_perf_fitness4 = pop_performance_list[best_idx]

        for i in range(self.max_generations):
            print(f"Generation {i + 1}......")

            elite_individual = population[best_idx]
            population[0] = elite_individual

            # Step0: Track stagnation and apply replacement
            if best_fitness <= last_best_fitness:
                no_improvement_generations += 1
            else:
                no_improvement_generations = 0
                last_best_fitness = best_fitness
            if no_improvement_generations >= 3:  # Stagnation threshold
                population = self.replace_population(population, pop_fitness_score_list, elite_ratio=0.1, random_ratio=0.5)
                no_improvement_generations = 0
            
            # Step1: Selection
            selected_parents = []
            for _ in range(len(population) // 2):
                parents = self.select_parents(population, pop_fitness_score_list)
                selected_parents.append(parents) # len(selected_parents) == len(population) // 2
            
            # Step2: Crossover
            offspring = []
            for parent1, parent2 in selected_parents: # len(population) // 2 times
                if random.random() < 0.5:
                    child1, child2 = self.crossover_tile([parent1, parent2], crossover_prob=0.6)
                else:
                    child1, child2 = self.crossover_dim([parent1, parent2], crossover_prob=0.8)
                offspring.extend([child1, child2]) # len(offspring) == len(population)
            
            # Step3: Mutation
            for individual in offspring:
                # Randomly choose between tile mutation, shuffle mutation and pod mutation
                if random.random() < 0.3:
                    new_child = self.mutate_shuffle(individual, mutation_prob=0.8)
                elif random.random() < 0.6:
                    new_child = self.mutate_tiles(individual, mutation_prob=0.8)
                else:
                    new_child = self.mutate_pods(individual, mutation_prob=0.6)
                individual[:] = new_child
            
            population = offspring
            
            # Step4: Replacement
            # self.replace_least_fit_individuals(population, offspring, pop_fitness_score_list)
            worst_index = pop_fitness_score_list.index(min(pop_fitness_score_list))
            population[worst_index] = elite_individual

            # Evaluate the new populations
            pop_fitness_score_list, pop_performance_list = self.evaluate_population(population)
            best_fitness, best_idx = self.find_best_score(pop_fitness_score_list)
            best_perf_fitness1, best_perf_fitness2, best_perf_fitness3, best_perf_fitness4 = pop_performance_list[best_idx]

            self.best_fitness = best_fitness
            self.best_perf_fitness1 = best_perf_fitness1
            self.best_perf_fitness2 = best_perf_fitness2
            self.best_perf_fitness3 = best_perf_fitness3
            self.best_perf_fitness4 = best_perf_fitness4
            print(f'\tBest fitness score: {best_fitness:.4f}, Best Cycle: {best_perf_fitness1:.4f}, Best EDP: {best_perf_fitness2:.4f}, Best Memory Access: {best_perf_fitness3:.4f}, Best Reuse Factor: {best_perf_fitness4:.4f}, \n \tBest Mapper: {population[best_idx]}')

        self.save_to_csv({
            'Best Fitness Score': self.best_fitness_score_list,
            f'Best Cycle': self.best_perf_fitness1_list,
            f'Best EDP': self.best_perf_fitness2_list,
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