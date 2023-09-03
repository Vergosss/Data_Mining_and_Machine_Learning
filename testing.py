from sklearn.svm import SVR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from IPython.display import display
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('test.csv')
#df = pd.read_csv('data.csv')
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
