import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Create K-means clustering model
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Predict cluster labels
labels = kmeans.labels_

# Evaluate the performance
silhouette_avg = silhouette_score(X, labels)
print("Silhouette Score:", silhouette_avg)

# Compare cluster assignments with true labels
df = pd.DataFrame({'Labels': labels, 'True Labels': y})
print(df)

# Visualize the clusters (considering the first two features)
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('K-means Clustering on Iris Dataset')
plt.show()
