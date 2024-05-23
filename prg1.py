import numpy as np
from sklearn.datasets import load_iris
from collections import defaultdict

# Load the Iris dataset
iris = load_iris()
data = iris['data']
target = iris['target']
target_names = iris['target_names']

# Convert the dataset into a list of lists
iris_data = np.column_stack((data, target)).tolist()

# Function to compute mean and standard deviation
def compute_stats(data, indices):
    stats = []
    for idx in indices:
        column_data = [row[idx] for row in data]
        mean = np.mean(column_data)
        std_dev = np.std(column_data)
        stats.append((mean, std_dev))
    return stats

# Overall mean and standard deviation for each measurement column
overall_stats = compute_stats(iris_data, [0, 1, 2, 3])

# Print overall statistics
print("Overall statistics:")
for i, (mean, std_dev) in enumerate(overall_stats):
    print(f"Column {i+1}: Mean = {mean:.2f}, Std Dev = {std_dev:.2f}")
print()

# Separate the data by species
species_data = defaultdict(list)
for row in iris_data:
    species_data[target_names[int(row[4])]].append(row)

# Compute mean and standard deviation for each species and measurement column
species_stats = {}
for species, data in species_data.items():
    species_stats[species] = compute_stats(data, [0, 1, 2, 3])

# Print statistics for each species
for species, stats in species_stats.items():
    print(f"Statistics for {species}:")
    for i, (mean, std_dev) in enumerate(stats):
        print(f"  Column {i+1}: Mean = {mean:.2f}, Std Dev = {std_dev:.2f}")
    print()

# Determine the "best" measurement for guessing species
# We'll consider the "best" measurement to be the one with the smallest variance within species
variances = defaultdict(list)
for species, stats in species_stats.items():
    for i, (mean, std_dev) in enumerate(stats):
        variances[i].append(std_dev ** 2)

average_variances = {i: np.mean(var_list) for i, var_list in variances.items()}
best_measurement = min(average_variances, key=average_variances.get)

print(f"The best measurement for guessing the Iris species is Column {best_measurement+1}.")
