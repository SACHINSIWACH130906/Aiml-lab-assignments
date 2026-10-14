import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = np.array([
    [1, 2], [2, 3], [3, 3],
    [8, 8], [9, 9], [10, 8]
])

model = KMeans(n_clusters=2, random_state=0)
model.fit(data)

labels = model.labels_
print("Cluster Labels:")
print(labels)

plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-Means Clustering")
plt.show()