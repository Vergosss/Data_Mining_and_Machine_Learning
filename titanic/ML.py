import seaborn as sb
from matplotlib import pyplot as plt
import pandas as pd


dataset='data.csv'
df=pd.read_csv(dataset)
df.info()
print(df.describe)
nulls=df.isnull().sum().sort_values(ascending=False)#gia kathe sthlh athrizei tis null times se fthinousa seira
print(nulls)
print(df['Cases'].mean())
df['Cases']=df['Cases'].fillna(df['Cases'].mean())#sta cases epeidh einai arithmitko dedomeno vazo stis adeies times th mesh
df.to_csv('data2.csv')
dataset2='data2.csv'
df2=pd.read_csv(dataset2)
print(df2.isnull().sum().sort_values(ascending=False))#gia kathe sthlh athrizei tis null times se fthinousa seira
print(df.corr(method='pearson'))