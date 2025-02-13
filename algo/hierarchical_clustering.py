import random
import math

def euclidean_distance(p1, p2):
    return math.sqrt(sum((p1[i] - p2[i]) ** 2 for i in range(len(p1))))

def average_linkage(cluster1, cluster2):
    return sum(euclidean_distance(p1, p2) for p1 in cluster1 for p2 in cluster2) / (len(cluster1) * len(cluster2))

def agglomerative_clustering(X):
    clusters = [[point] for point in X]
    
    while len(clusters) > 1:
        min_distance = float('inf')
        merge_idx = (-1, -1)
        
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                dist = average_linkage(clusters[i], clusters[j])
                if dist < min_distance:
                    min_distance = dist
                    merge_idx = (i, j)
        
        c1, c2 = merge_idx
        clusters[c1].extend(clusters[c2])
        del clusters[c2]
    
    return clusters

# Generate sample dataset
random.seed(42)
X = [[random.uniform(-10, 10), random.uniform(-10, 10)] for _ in range(10)]

# Apply Agglomerative Hierarchical Clustering
clusters = agglomerative_clustering(X)

# Print results
for i, cluster in enumerate(clusters):
    print(f"Cluster {i}: {cluster}")
