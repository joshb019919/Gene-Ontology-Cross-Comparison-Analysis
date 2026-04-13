from scipy.cluster.hierarchy import ward, dendrogram
import matplotlib.pyplot as plt
import numpy as np

dataset = np.array([[1,2], [2,3], [2,5], [8,7], [8,6], [9,5]])

linkage_matrix = ward(dataset)
plt.figure(figsize=(10,7))
dendrogram(linkage_matrix)
plt.title("Ward Method Dendrogram")
plt.show()
