import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from IPython.display import display
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler #if we have big variance between elements standardscaling wouldnt work on small numbers
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# read dataset
#df = pd.read_csv('data.csv')
df = pd.read_csv('data.csv')

Countries = []
Countries = df['Entity']
Countries = list(set(Countries))

# fill missing values
df['Deaths'] = df.groupby('Entity')['Deaths'].transform(lambda x: x.fillna(x.ffill(axis=0)))
df['Deaths'] = df.groupby('Entity')['Deaths'].transform(lambda x: x.fillna(x.bfill(axis=0)))
# df['Deaths'].fillna('2.1',inplace=True)
df['Cases'] = df.groupby('Entity')['Cases'].transform(lambda x: x.fillna(x.ffill(axis=0)))
df['Cases'] = df.groupby('Entity')['Cases'].transform(lambda x: x.fillna(x.bfill(axis=0)))
df['Daily tests'] = df.groupby('Entity')['Daily tests'].transform(lambda x: x.fillna(x.ffill(axis=0)))
df['Daily tests'] = df.groupby('Entity')['Daily tests'].transform(lambda x: x.fillna(x.bfill(axis=0)))
#

columns_to_drop = ['Latitude','Longitude','Average temperature per year','Hospital beds per 1000 people','Medical doctors per 1000 people','GDP/Capita','Median age','Population aged 65 and over (%)','Date','Daily tests','Deaths']
columns_to_drop2 = ['Latitude','Longitude','Average temperature per year','Hospital beds per 1000 people','Medical doctors per 1000 people','GDP/Capita','Median age','Population aged 65 and over (%)','Date','Daily tests','Cases']
columns_to_drop3 = ['Latitude','Longitude','Average temperature per year','Hospital beds per 1000 people','Medical doctors per 1000 people','GDP/Capita','Median age','Population aged 65 and over (%)','Date','Deaths']
death_rate_per_country = df.groupby('Entity').apply(
    lambda x: 100 * (x['Deaths'].max() / x['Population'].min().astype(float)))


#####
print('----------------------------\n')


df_last = df.groupby('Entity').tail(1).drop(['Entity', 'Date'], axis=1)  # rikse tis mi arithmitikes stiles
continents = df_last[
    'Continent'].unique()  # afou to df_last einai 1 pleiada ana xora tote apla pernoume tin ipiro kathe xoras(exei duplicates)
print(continents)


cluster_ungrouped = df.drop(columns_to_drop,axis=1)#pernoume mono ta features pou mas endiaferoun
cluster_ungrouped2 = df.drop(columns_to_drop2,axis=1)#cases,deaths,population ktlp..
cluster_ungrouped3 = df.drop(columns_to_drop3,axis=1)#daily tests,cases
#
cluster_grouped = cluster_ungrouped.groupby(['Entity','Continent'])[['Population','Cases']].agg('max')
cluster_grouped2 = cluster_ungrouped2.groupby(['Entity','Continent'])[['Population','Deaths']].agg('max')
cluster_grouped3 = cluster_ungrouped3.groupby(['Entity','Continent']).agg({'Daily tests': 'sum' , 'Cases': 'max'})
#
print('---------------------------\nCluster ungrouped data:\n',cluster_ungrouped)
print('---------------------------\nCluster ungrouped 2:\n',cluster_ungrouped2)
print('---------------------------\nCluster ungrouped 3:\n',cluster_ungrouped3)

#
print('---------------------------\nCluster grouped Cases/Population data(entity,continent):\n',cluster_grouped)
print('---------------------------\nCluster grouped Deaths/Population data(entity,continent):\n',cluster_grouped2)
print('---------------------------\nCluster grouped Cases/Total tests data(entity,continent):\n',cluster_grouped3)
#
cluster_grouped2['Death rate'] = 100*(cluster_grouped2['Deaths']/cluster_grouped2['Population'])
#
cluster_grouped['Case rate'] = 100*(cluster_grouped['Cases']/cluster_grouped['Population'])
#
cluster_grouped3['Positivity rate'] = 100*(cluster_grouped3['Cases']/cluster_grouped3['Daily tests'])
#
print('---------------------------\nCluster grouped Cases/Population/Case rate data(entity,continent):\n',cluster_grouped)
#
print('---------------------------\nCluster grouped Deaths/Population/Death rate data(entity,continent):\n',cluster_grouped2)
#
print('---------------------------\nCluster grouped Cases/Total tests/Positivity rate data(entity,continent):\n',cluster_grouped3)

#
scaled_cluster_grouped2 = StandardScaler().fit_transform(cluster_grouped2)#scale data for improved clustering
#me convert to dataframe because the above is numpy nd-array------SCALING
scaled_cluster_grouped2 = pd.DataFrame(scaled_cluster_grouped2,columns=cluster_grouped2.columns)
#
scaled_cluster_grouped = StandardScaler().fit_transform(cluster_grouped)
scaled_cluster_grouped = pd.DataFrame(scaled_cluster_grouped,columns=cluster_grouped.columns)
#
scaled_cluster_grouped3 = StandardScaler().fit_transform(cluster_grouped3)
scaled_cluster_grouped3 = pd.DataFrame(scaled_cluster_grouped3,columns=cluster_grouped3.columns)
#
print('---------------------------\nScaled cluster grouped2:\n',scaled_cluster_grouped2)
#
print('---------------------------\nScaled cluster grouped:\n',scaled_cluster_grouped)
#
print('---------------------------\nScaled cluster grouped3:\n',scaled_cluster_grouped3)

#
wcss = []#wcss=athrisma ton tetragonon ton apostaseon metaxy kentroeidon kai  kathe simiou

dataset_length = len(cluster_grouped.index)# row count of dataset
print('---------------------------\nDataset length is ',dataset_length)
#elbow method

for k in range(1,dataset_length+1):#efoson exoume n simia/eggrafes o max arithmos ton clusters einai n- ta eksetazoume ola
    kmeans = KMeans(n_clusters=k)#apply kmeans with i number of clusters
    kmeans.fit(scaled_cluster_grouped2.values)#fit in the algorithm our data whatever it is
    wcss.append(kmeans.inertia_)#append the i clusters kmeans wcss to the list

plt.plot(range(1,dataset_length+1),wcss,marker='o')#aksonas x ta k aksonas y to wcss/inertias
plt.title('Optimal number of clusters')#titlos grafimatos
plt.xlabel('Number of k')#onomasia aksona x pou einai to k diladi o arithmos ton cluster
plt.ylabel('WCSS')#onomasia aksona y to wcss
plt.show()#emfanisi grafikis
optimal = 6
##
#perform clustering
kmeans = KMeans(n_clusters=optimal)
kmeans.fit(scaled_cluster_grouped2.values)#vazo ta dedomena sto modelo
cluster_labels = kmeans.labels_#labels of each point-i.e the clusters
cluster_centers = kmeans.cluster_centers_ #centroids of my clusters
#dimensionality reduction
pca = PCA(n_components=2)#2? px plithismos-death rate-genika dataset(shape(1))-> pca_num_components = stiles -1
#from 3 columns i want to visualize x and y axis so i reduce them to 2 using pca
reduced_scaled_cluster_grouped2 = pd.DataFrame(data=pca.fit_transform(scaled_cluster_grouped2.values),columns=['PCA1','PCA2'])
reduced_cluster_centers = pca.transform(cluster_centers)
print('------------------------------\nReduced scaled cluster grouped2: \n',reduced_scaled_cluster_grouped2)
#visualization

#plt.scatter(scaled_cluster_grouped2['Population'],scaled_cluster_grouped2['Deaths'],c=kmeans.labels_)#plotare ta dedomena/scattare ta me xromata / katigories tis systades pou vrike
#cluster_labels=kmeans.fit_predict(death_rates) #Compute cluster centers and predict cluster index for each sample.
plt.scatter(reduced_scaled_cluster_grouped2['PCA1'],reduced_scaled_cluster_grouped2['PCA2'],c=kmeans.labels_)
plt.scatter(reduced_cluster_centers[:,0],reduced_cluster_centers[:,1],marker='x',s=100,c='red')#visualize centroids
plt.xlabel('Population')
plt.ylabel('Deaths')
print(cluster_labels)
plt.show()

####OR
##3D Scatter plot since i have 3 attributes
fig = plt.figure(figsize = (10, 7)) #figure to plot on
ax = plt.axes(projection ="3d") #axes with 3d projection
#
ax.scatter3D(scaled_cluster_grouped2['Population'], scaled_cluster_grouped2['Deaths'], scaled_cluster_grouped2['Death rate'], c=kmeans.labels_)
ax.scatter3D(cluster_centers[:,0],cluster_centers[:,1],cluster_centers[:,2],marker='x',s=100,c='red')

#scatter in 3d space the 3 columns of the new dataframe
plt.title("Population/Deaths/Death rate per country")#plot title
plt.show()#emfanise th grafiki
###--An exo >3 stiles tote meiono th diastatikotita(me PCA) se 2 h 3 kai xrisimopoio  scatter() kai scatter3d antistixa
#an exo 2 h 3 stiles tote amesa me scatter h scatter3d kano apeikonisi ton dedomenon

######--------------------------CASE RATE----------------------#######

wcss = []#wcss=athrisma ton tetragonon ton apostaseon metaxy kentroeidon kai  kathe simiou

#elbow method

for k in range(1,dataset_length+1):#efoson exoume n simia/eggrafes o max arithmos ton clusters einai n- ta eksetazoume ola
    kmeans = KMeans(n_clusters=k)#apply kmeans with i number of clusters
    kmeans.fit(scaled_cluster_grouped.values)#fit in the algorithm our data whatever it is
    wcss.append(kmeans.inertia_)#append the i clusters kmeans wcss to the list

plt.plot(range(1,dataset_length+1),wcss,marker='o')#aksonas x ta k aksonas y to wcss/inertias
plt.title('Optimal number of clusters')#titlos grafimatos
plt.xlabel('Number of k')#onomasia aksona x pou einai to k diladi o arithmos ton cluster
plt.ylabel('WCSS')#onomasia aksona y to wcss
plt.show()#emfanisi grafikis
optimal = 6

##

#perform clustering
kmeans = KMeans(n_clusters=optimal)
kmeans.fit(scaled_cluster_grouped.values)#vazo ta dedomena sto modelo
cluster_labels = kmeans.labels_#labels of each point-i.e the clusters
cluster_centers = kmeans.cluster_centers_ #centroids of my clusters

##

#dimensionality reduction
pca = PCA(n_components=2)#2? px plithismos-death rate-genika dataset(shape(1))-> pca_num_components = stiles -1
#from 3 columns i want to visualize x and y axis so i reduce them to 2 using pca
reduced_scaled_cluster_grouped = pd.DataFrame(data=pca.fit_transform(scaled_cluster_grouped.values),columns=['PCA1','PCA2'])
reduced_cluster_centers = pca.transform(cluster_centers)
print('------------------------------\nReduced scaled cluster grouped: \n',reduced_scaled_cluster_grouped)
#visualization

#plt.scatter(scaled_cluster_grouped2['Population'],scaled_cluster_grouped2['Deaths'],c=kmeans.labels_)#plotare ta dedomena/scattare ta me xromata / katigories tis systades pou vrike
#cluster_labels=kmeans.fit_predict(death_rates) #Compute cluster centers and predict cluster index for each sample.
plt.scatter(reduced_scaled_cluster_grouped['PCA1'],reduced_scaled_cluster_grouped['PCA2'],c=kmeans.labels_)
plt.scatter(reduced_cluster_centers[:,0],reduced_cluster_centers[:,1],marker='x',s=100,c='red')#visualize centroids
plt.xlabel('Population')
plt.ylabel('Cases')
print(cluster_labels)
plt.show()

####OR
##3D Scatter plot since i have 3 attributes
fig = plt.figure(figsize = (10, 7)) #figure to plot on
ax = plt.axes(projection ="3d") #axes with 3d projection
#
ax.scatter3D(scaled_cluster_grouped['Population'], scaled_cluster_grouped['Cases'], scaled_cluster_grouped['Case rate'], c=kmeans.labels_)
ax.scatter3D(cluster_centers[:,0],cluster_centers[:,1],cluster_centers[:,2],marker='x',s=100,c='red')
#scatter in 3d space the 3 columns of the new dataframe
plt.title("Population/Cases/Case rate per country")#plot title
plt.show()#emfanise th grafiki
###--An exo >3 stiles tote meiono th diastatikotita(me PCA) se 2 h 3 kai xrisimopoio  scatter() kai scatter3d antistixa
#an exo 2 h 3 stiles tote amesa me scatter h scatter3d kano apeikonisi ton dedomenon




######------Positivity rate------######


wcss = []#wcss=athrisma ton tetragonon ton apostaseon metaxy kentroeidon kai  kathe simiou

#elbow method

for k in range(1,dataset_length+1):#efoson exoume n simia/eggrafes o max arithmos ton clusters einai n- ta eksetazoume ola
    kmeans = KMeans(n_clusters=k)#apply kmeans with i number of clusters
    kmeans.fit(scaled_cluster_grouped3.values)#fit in the algorithm our data whatever it is
    wcss.append(kmeans.inertia_)#append the i clusters kmeans wcss to the list

plt.plot(range(1,dataset_length+1),wcss,marker='o')#aksonas x ta k aksonas y to wcss/inertias
plt.title('Optimal number of clusters')#titlos grafimatos
plt.xlabel('Number of k')#onomasia aksona x pou einai to k diladi o arithmos ton cluster
plt.ylabel('WCSS')#onomasia aksona y to wcss
plt.show()#emfanisi grafikis
optimal = 6

##

#perform clustering
kmeans = KMeans(n_clusters=optimal)
kmeans.fit(scaled_cluster_grouped3.values)#vazo ta dedomena sto modelo
cluster_labels = kmeans.labels_#labels of each point-i.e the clusters
cluster_centers = kmeans.cluster_centers_ #centroids of my clusters

##

#dimensionality reduction
pca = PCA(n_components=2)#2? px plithismos-death rate-genika dataset(shape(1))-> pca_num_components = stiles -1
#from 3 columns i want to visualize x and y axis so i reduce them to 2 using pca
reduced_scaled_cluster_grouped3 = pd.DataFrame(data=pca.fit_transform(scaled_cluster_grouped3.values),columns=['PCA1','PCA2'])
reduced_cluster_centers = pca.transform(cluster_centers)
print('------------------------------\nReduced scaled cluster grouped: \n',reduced_scaled_cluster_grouped3)
#visualization

#plt.scatter(scaled_cluster_grouped2['Population'],scaled_cluster_grouped2['Deaths'],c=kmeans.labels_)#plotare ta dedomena/scattare ta me xromata / katigories tis systades pou vrike
#cluster_labels=kmeans.fit_predict(death_rates) #Compute cluster centers and predict cluster index for each sample.
plt.scatter(reduced_scaled_cluster_grouped3['PCA1'],reduced_scaled_cluster_grouped3['PCA2'],c=kmeans.labels_)
plt.scatter(reduced_cluster_centers[:,0],reduced_cluster_centers[:,1],marker='x',s=100,c='red')#visualize centroids
plt.xlabel('Cases')
plt.ylabel('Total tests')
print(cluster_labels)
plt.show()

####OR
##3D Scatter plot since i have 3 attributes
fig = plt.figure(figsize = (10, 7)) #figure to plot on
ax = plt.axes(projection ="3d") #axes with 3d projection
#
ax.scatter3D(scaled_cluster_grouped3['Cases'], scaled_cluster_grouped3['Daily tests'], scaled_cluster_grouped3['Positivity rate'], c=kmeans.labels_)
ax.scatter3D(cluster_centers[:,0],cluster_centers[:,1],cluster_centers[:,2],marker='x',s=100,c='red')
#scatter in 3d space the 3 columns of the new dataframe
plt.title("Cases/Total tests/Positivity rate per country")#plot title
plt.show()#emfanise th grafiki
###--An exo >3 stiles tote meiono th diastatikotita(me PCA) se 2 h 3 kai xrisimopoio  scatter() kai scatter3d antistixa
#an exo 2 h 3 stiles tote amesa me scatter h scatter3d kano apeikonisi ton dedomenon