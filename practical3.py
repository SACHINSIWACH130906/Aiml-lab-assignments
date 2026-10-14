import numpy as np
import matplotlib.pyplot as plt

data = np.array([
    [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]
])

mean = np.mean(data, axis=0)
centered_data = data - mean
cov_matrix = np.cov(centered_data.T)

eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
principal_component = eigenvectors[:, np.argmax(eigenvalues)]

reduced_data = centered_data.dot(principal_component)

print("Original Data:")
print(data)
print("\nReduced Data:")
print(reduced_data)

plt.scatter(reduced_data, np.zeros(len(reduced_data)))
plt.xlabel("Principal Component 1")
plt.title("PCA Dimensionality Reduction")
plt.show()