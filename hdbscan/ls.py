import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hdbscan
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
import time

# Record the start time
start_time = time.time()

# Load the dataset from a CSV file (replace 'FC1Dataset11.csv' with your file path)
df = pd.read_csv('ls.csv')

# Extract the features (assuming your dataset has columns '1' and '2')
X = df[['1', '2']].values

# Load the true labels (replace '3' with the actual column name containing the true labels)
true_labels = df['3'].values

# If your dataset is not pre-scaled, you might want to standardize it.
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Define the HDBSCAN model
clusterer = hdbscan.HDBSCAN(min_cluster_size=5)  # You can adjust parameters as needed

# Fit the HDBSCAN model to the dataset
clusterer.fit(X)

# Get the cluster labels
labels = clusterer.labels_

# Number of clusters found (excluding noise points)
n_clusters_ = len(set(labels)) 

# Calculate the silhouette score
silhouette_avg = silhouette_score(X, labels)

# Calculate the adjusted mutual information (AMI) score without true labels
ami_score = adjusted_mutual_info_score(true_labels, labels)

# Calculate the adjusted Rand index (ARI) without true labels
ari_score = adjusted_rand_score(true_labels, labels)

# Print the number of clusters found

print(f'Adjusted Mutual Information (AMI) Score: {ami_score}')
print(f'Adjusted Rand Index (ARI) Score: {ari_score}')
print(f'Silhouette Score: {silhouette_avg}')
print(f'Number of detected clusters: {n_clusters_}')

# Plot the results
plt.figure()
plt.title('HDBSCAN Clustering on "ls" Dataset')

# Plot the clusters
for cluster in set(labels):
    if cluster == -1:
        col = [0, 0, 0, 1]  # Black for noise points
    else:
        col = plt.cm.Spectral(cluster / len(set(labels)))

    class_member_mask = (labels == cluster)
    xy = X[class_member_mask]
    plt.scatter(xy[:, 0], xy[:, 1], s=30, c=[col], label=f'Cluster {cluster}' if cluster != -1 else 'Noise')

plt.xlabel('1')
plt.ylabel('2')
plt.legend()
# Record the end time
end_time = time.time()

# Calculate the elapsed time in seconds
elapsed_time = end_time - start_time

print(f"Time taken: {elapsed_time:.4f} seconds")
plt.show()


