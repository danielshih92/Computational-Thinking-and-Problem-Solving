import pandas as pd
import numpy as np
import random

class Example:
    def __init__(self, features):
        self.features = np.array(features)

    def distance(self, other):
        return np.linalg.norm(self.features - other.features)

class Cluster:
    def __init__(self, examples):
        self.examples = examples
        self.centroid = self.computeCentroid()

    def computeCentroid(self):
        return Example(np.mean([e.features for e in self.examples], axis=0))

    def getCentroid(self):
        return self.centroid

    def update(self, examples):
        old_centroid = self.centroid
        self.examples = examples
        self.centroid = self.computeCentroid()
        return old_centroid.distance(self.centroid)

def kmeans(examples, k, verbose=False):
    # Get k randomly chosen initial centroids, create cluster for each
    initial_centroids = random.sample(examples, k)
    clusters = [Cluster([e]) for e in initial_centroids]

    # Iterate until centroids do not change
    converged = False
    num_iterations = 0
    while not converged:
        num_iterations += 1
        new_clusters = [[] for _ in range(k)]

        # Associate each example with closest centroid
        for e in examples:
            smallest_distance = e.distance(clusters[0].getCentroid())
            index = 0
            for i in range(1, k):
                distance = e.distance(clusters[i].getCentroid())
                if distance < smallest_distance:
                    smallest_distance = distance
                    index = i
            new_clusters[index].append(e)

        # Update each cluster; check if a centroid has changed
        converged = True
        for i in range(k):
            if clusters[i].update(new_clusters[i]) > 0.0:
                converged = False
        if verbose:
            print('Iteration #' + str(num_iterations))
            for c in clusters:
                print(c)
            print('')  # add blank line
    return clusters

def z_scale_features(df):
    # Copy the dataframe
    df_std = df.copy()
    
    # Apply the z-score method
    for column in df_std.columns:
        df_std[column] = (df_std[column] - df_std[column].mean()) / df_std[column].std()
        
    return df_std


# Load the data
df = pd.read_csv('cardiacPatientData.txt', header=None, skiprows=1)

# Assign column names
df.columns = ['Heart rate', 'heart attacks', 'Age', 'ST elevation', '1=died']

# Extract features
X = df.drop('1=died', axis=1)

# Scale the features
X_scaled = z_scale_features(X)

# Convert scaled data to numpy array
X = X_scaled.values

# Extract labels
y = df['1=died'].values

# Convert data to examples
examples = [Example(x) for x in X]

# Cluster the data
clusters = kmeans(examples, k=2)

# Count 'died' and 'alive' in each cluster
for i, cluster in enumerate(clusters):
    labels = [y[j] for j in range(len(y)) if examples[j] in cluster.examples]
    num_died = labels.count(1)  # 1 is 'died'
    print(f"Cluster of size {len(cluster.examples)} with fraction of death positives = {num_died / len(cluster.examples):.4f} and {num_died} death.")
