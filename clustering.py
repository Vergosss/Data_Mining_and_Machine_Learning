import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from IPython.display import display
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
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

death_rate_per_country = df.groupby('Entity').apply(
    lambda x: 100 * (x['Deaths'].max() / x['Population'].min().astype(float)))


#####
print('----------------------------\n')


df_temp = df.loc[(df['Date'] > '2020-02-25') & (df['Date'] <= '2020-10-13')]
countries = ['Greece', 'Cyprus', 'Slovenia', 'Croatia']  # countries to test the plot

df_last = df.groupby('Entity').tail(1).drop(['Entity', 'Date'], axis=1)  # rikse tis mi arithmitikes stiles
continents = df_last[
    'Continent'].unique()  # afou to df_last einai 1 pleiada ana xora tote apla pernoume tin ipiro kathe xoras(exei duplicates)
print(continents)


cluster_ungrouped = df.drop(columns_to_drop,axis=1)
cluster_ungrouped2 = df.drop(columns_to_drop2,axis=1)

cluster_grouped = cluster_ungrouped.groupby(['Entity','Continent'])[['Population','Cases']].agg('max')
cluster_grouped2 = cluster_ungrouped2.groupby(['Entity','Continent'])[['Population','Deaths']].agg('max')
print('---------------------------\nCluster ungrouped data:\n',cluster_ungrouped)
print('---------------------------\nCluster ungrouped data:\n',cluster_ungrouped2)
#
print('---------------------------\nCluster grouped Cases/Population data(entity,continent):\n',cluster_grouped)
print('---------------------------\nCluster grouped Deaths/Population data(entity,continent):\n',cluster_grouped2)

#
scaled_cluster_grouped2 = StandardScaler().fit_transform(cluster_grouped2)#scale data for improved clustering
print(scaled_cluster_grouped2)
inertias = []#wcss=athrisma ton tetragonon ton apostaseon metaxy kentroeidon kai  kathe simiou

dataset_length = len(cluster_grouped.index)# row count of dataset
print('---------------------------\nDataset length is ',dataset_length)
#elbow method

for k in range(1,dataset_length+1):#efoson exoume n simia/eggrafes o max arithmos ton clusters einai n- ta eksetazoume ola
    kmeans = KMeans(n_clusters=k)#apply kmeans with i number of clusters
    kmeans.fit(scaled_cluster_grouped2)#fit in the algorithm our data whatever it is
    inertias.append(kmeans.inertia_)#append the i clusters kmeans wcss to the list

plt.plot(range(1,dataset_length+1),inertias,marker='o')#aksonas x ta k aksonas y to wcss/inertias
plt.title('Optimal number of clusters')#titlos grafimatos
plt.xlabel('Number of k')#onomasia aksona x pou einai to k diladi o arithmos ton cluster
plt.ylabel('WCSS')#onomasia aksona y to wcss
plt.show()#emfanisi grafikis
optimal = 4
##
#perform clustering
kmeans = KMeans(n_clusters=optimal)
kmeans.fit(scaled_cluster_grouped2)#vazo ta dedomena sto modelo
cluster_labels = kmeans.labels_#labels of each point

#visualization

#plt.scatter(scaled_cluster_grouped2['Population'],scaled_cluster_grouped2['Deaths'],c=kmeans.labels_)#plotare ta dedomena/scattare ta me xromata / katigories tis systades pou vrike
#cluster_labels=kmeans.fit_predict(death_rates) #Compute cluster centers and predict cluster index for each sample.
plt.scatter(cluster_grouped2['Population'],cluster_grouped2['Deaths'],c=kmeans.labels_)
plt.xlabel('Population')
plt.ylabel('Deaths')
print(cluster_labels)
plt.show()
start_date = pd.to_datetime("2021-01-01")
print(start_date)
