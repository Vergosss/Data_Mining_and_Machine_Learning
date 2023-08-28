#libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
#read dataset
df = pd.read_csv('test.csv')
#general information about 
print('In the whole dataset null values of each attribute are:\n',df.isnull().sum())
print('------------------\nInformation of every attribute in the dataset:\n',df.info())
print('------------------\nDescribe every attribute in the dataset:\n',df.describe())
Countries = []
Countries = df['Entity']
Countries = list(set(Countries))
#df = df.groupby('Entity')#group data by country
print('--------------------\nDescribe deaths attribute for each group:\n',df.groupby('Entity').describe()['Deaths'])#omadopoihsh vash xoras kai ypologizei statistika gia kathe attribute mono gia thn xora ayth
#Countries = list(df['Entity'])
stats = df.groupby('Entity')[['Daily tests','Cases','Deaths']].mean()#gia kathe xora athrisma/mesos kok ton deaths
#
print('---------------------\nFor each attribute per group print its mean:\n',stats)
print('---------------------\nCountries are:\n',Countries)
###
missing_deaths_per_country = df.groupby('Entity')['Deaths'].apply(lambda x: x.isnull().sum())
missing_cases_per_country = df.groupby('Entity')['Cases'].apply(lambda x: x.isnull().sum())#poses times leipoun apo kathe xora
missing_daily_tests_per_country = df.groupby('Entity')['Daily tests'].apply(lambda x: x.isnull().sum())
###

print('-------------------\nMissing deaths per country-Missing cases per country-Missing Daily Tests per country:\n',missing_deaths_per_country,'\n',missing_cases_per_country,'\n',missing_daily_tests_per_country)
#fill missing values
df['Deaths'] = df.groupby('Entity').transform(lambda x: x.fillna(x.mean()))
#df['Deaths'].fillna('2.1',inplace=True)
missing_deaths_per_country = df.groupby('Entity')['Deaths'].apply(lambda x: x.isnull().sum())
print('-------------------\nMissing deaths now per country:\n',missing_deaths_per_country)
print(df)
