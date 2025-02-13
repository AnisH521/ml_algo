import random
import math

def initialize_centroids(X, k):
    random.seed(42)
    return random.sample(X, k)

def euclidean_distance(p1, p2):
    return math.sqrt(sum((p1[i] - p2[i]) ** 2 for i in range(len(p1))))

def assign_clusters(X, centroids):
    labels = []
    for point in X:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        labels.append(distances.index(min(distances)))
    return labels

def compute_centroids(X, labels, k):
    new_centroids = [[0] * len(X[0]) for _ in range(k)]
    counts = [0] * k
    
    for i, label in enumerate(labels):
        for j in range(len(X[0])):
            new_centroids[label][j] += X[i][j]
        counts[label] += 1
    
    for i in range(k):
        if counts[i] > 0:
            new_centroids[i] = [value / counts[i] for value in new_centroids[i]]
    
    return new_centroids

def kmeans(X, k, max_iters=100, tol=1e-4):
    centroids = initialize_centroids(X, k)
    for _ in range(max_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = compute_centroids(X, labels, k)
        if all(euclidean_distance(centroids[i], new_centroids[i]) < tol for i in range(k)):
            break
        centroids = new_centroids
    return labels, centroids

# Generate sample dataset
random.seed(42)
X = [[random.uniform(-10, 10), random.uniform(-10, 10)] for _ in range(300)]
print(X[:5])

# Apply K-Means clustering
labels, centroids = kmeans(X, k=3)

# Print results
for i, centroid in enumerate(centroids):
    print(f"Centroid {i}: {centroid}")
