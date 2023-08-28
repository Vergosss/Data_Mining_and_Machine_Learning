import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
df=pd.read_csv('data.csv')
print(df)
print(df.info())
print(df.describe(include='all'))
NaNs_df=df.isnull().sum().sort_values(ascending=False)
print(NaNs_df)
countries = list(df.Entity.unique())
print(countries)
tests = df.groupby('Entity')['Daily tests'].transform('mean')
cases = df.groupby('Entity')['Cases'].transform('mean')
deaths = df.groupby('Entity')['Deaths'].transform('mean')
print(len(countries))
print('HERE---------------------->')
print(tests)

print(tests)
new_df = pd.DataFrame()
for country,t_means,c_means,d_means in zip(countries,tests,cases,deaths):
    country_df = df[df['Entity'] == country]
    country_df['Daily tests'] = country_df['Daily tests'].fillna(t_means)
    country_df['Cases'] = country_df['Cases'].fillna(c_means)
    country_df['Deaths'] = country_df['Deaths'].fillna(d_means)
    new_df = pd.concat([new_df, country_df], ignore_index=True)

nulls = new_df.isnull().sum().sort_values(ascending=False)
print(nulls)
print(new_df.describe())
new_df.to_csv('newdata.csv')
death_data = new_df.groupby(['Entity'])['Deaths'].sum().reset_index()
death_data = new_df.sort_values(by='Deaths',ascending=False)
plt.figure('Deaths by country')
plt.plot(death_data['Entity'],death_data['Deaths'],color='blue')
plt.title('Covid deaths per country')
plt.show()
print(death_data.describe())

case_data = new_df.groupby(['Entity'])['Cases'].sum().reset_index()
case_data = new_df.sort_values(by='Cases',ascending=False)
plt.figure('Total cases by country between the given dates')
plt.plot(case_data['Entity'],case_data['Cases'],color='red')
plt.title('Total cases by country between the given dates')
plt.show()
print(case_data.describe())

test_data = new_df.groupby(['Entity'])['Daily tests'].sum().reset_index()
test_data = new_df.sort_values(by='Daily tests',ascending=False)
plt.figure('Total tests by country between the given dates')
plt.plot(test_data['Entity'],test_data['Daily tests'],color='yellow')
plt.title('Total tests by country between the given dates')
plt.show()
print(test_data['Daily tests'].describe())
#antistixa anti gia posa eginan synolika boroume na valoume mesa px test gia olo to diastima pou katagrafame ktlp
