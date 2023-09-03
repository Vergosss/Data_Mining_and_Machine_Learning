import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage


data = [[0,0.4,0.2,0.8,0.3],[0.4,0,0.64,0.5,0.9],[0.2,0.64,0,0.41,0.2],[0.8,0.5,0.41,0,0.7],[0.3,0.9,0.8,0.7,0]]
print(data)
linkage_data = linkage(data, method='complete', metric='euclidean')
dendrogram(linkage_data)

plt.show()
