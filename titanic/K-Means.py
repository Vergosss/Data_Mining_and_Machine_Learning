import numpy as mp
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
plt.scatter(x,y)
plt.show()
data = list(zip(x,y))
#an exo n simia ta megista clusters(systades) tha einai n diladi ena cluster gia kathe simio
inertias = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

plt.plot(range(1,11), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()
#me ton parapano kodika vrisko th veltisti timh tou K- otan k=2 h kabyli ginetai pio grammiki
#trexo ton kmeans sta dedomena tora
kmeans = KMeans(n_clusters=2)#veltisti timi tou K
kmeans.fit(data)#xose ston algorithmo ta dedomena
plt.scatter(x,y,c=kmeans.labels_)#diespeire ta dedomena meso ton labels/clusters pou o algorithmos proteine
plt.show()#dio xromata 2 clusters