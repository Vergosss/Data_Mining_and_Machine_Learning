import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
x = [4, 5, 10, 4, 3, 11, 14 , 8, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
data = list(zip(x,y)) #to zip pernei orismata iterables kai kanei 1-1 antistixisi se tuple. meta ta kanei mia lista apo tuples
print(data)
linkage_data = linkage(data, method= 'ward', metric='euclidean')
dendrogram(linkage_data) #dendrograma gia to clustering
plt.show()
#hierarchical_cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
#labels = hierarchical_cluster.fit_predict(data)
#plt.scatter(x,y, c=labels)
#plt.show()
###K-NN
classes = [0, 0, 1, 0, 0, 1, 1, 0, 1, 1]
plt.scatter(x, y, c=classes)#vale sygkekrimenh klash-xroma se kathe simio
plt.show()
knn = KNeighborsClassifier(n_neighbors=1)#n geitones apo ton algoprithmo knn
knn.fit(data,classes)#fitaro ston knn algorithmo ta dedomena
new_x = 8#prospatho na katigoriopoihso ena neo x
new_y = 21#prospatho na katigoriso ena neo y
new_point = [(new_x, new_y)]
prediction = knn.predict(new_point)#provlepse pou tha katigoriopoihthei to neo simio
plt.scatter(x + [new_x],y + [new_y],c=classes + [prediction[0]])#prostheto sto cluster to neo simio
plt.show()


