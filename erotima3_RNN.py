from sklearn.svm import SVR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from IPython.display import display
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import r2_score
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
#reading file
df = pd.read_csv('data.csv')


Countries = []
Countries = df['Entity']#discrete countries(104) from the dataset
Countries = list(set(Countries))

# fill missing values-first preprocessing phase(before scaling)
df['Deaths'] = df.groupby('Entity')['Deaths'].transform(lambda x: x.fillna(x.ffill(axis=0)))
df['Deaths'] = df.groupby('Entity')['Deaths'].transform(lambda x: x.fillna(x.bfill(axis=0)))
# df['Deaths'].fillna('2.1',inplace=True)
df['Cases'] = df.groupby('Entity')['Cases'].transform(lambda x: x.fillna(x.ffill(axis=0)))
df['Cases'] = df.groupby('Entity')['Cases'].transform(lambda x: x.fillna(x.bfill(axis=0)))
df['Daily tests'] = df.groupby('Entity')['Daily tests'].transform(lambda x: x.fillna(x.ffill(axis=0)))
df['Daily tests'] = df.groupby('Entity')['Daily tests'].transform(lambda x: x.fillna(x.bfill(axis=0)))
#
df_greece = df[df['Entity']=='Greece']#efoson kanoume analysh gia thn ellada apo to dataset perno mono tin ellada

columns_to_drop = ['Entity','Continent','Latitude','Longitude','Average temperature per year','Hospital beds per 1000 people','Medical doctors per 1000 people','GDP/Capita','Median age','Population aged 65 and over (%)','Deaths']

df_greece=df_greece.drop(columns_to_drop,axis=1)#dropping everything except date,population, daily tests ,cases
print('----------------------------------\nDataframe:\n',df_greece)
#
##
df_greece=df_greece[df_greece['Date']>= '2021-01-01']#dedomena elladas meta tis 1/1/2021
print(df_greece)
X = df_greece[['Population','Daily tests']]
Y = df_greece[['Cases']]

#

for lag in range(1, 4):  # shift daily tests, population 3 days back to use for the forecasting and Cases 3 days after to get the needed forecasted value
    df_greece[f'Daily_tests_lag_{lag}'] = df_greece['Daily tests'].shift(lag)
    df_greece[f'Population_lag_{lag}'] = df_greece['Population'].shift(lag)#xrisimopoio ola ta 'endiaferonta' attributes gia th provlepsi
    df_greece[f'Cases_lag_{lag}'] = df_greece['Cases'].shift(lag)#kano provlepsi gia ta cases vasismenos kai stis palioteres times tou
df_greece['Cases_3_days_ahead'] = df_greece['Cases'].shift(-3)  # Target variable-provlepsi 3 meres meta
print(df_greece)
#ousiastika edo afou pia de xreiazomaste tis imerominies tis dioxnoume opos kai ta alla attributes afou
#me ta shifts pira ta proigoumena kai ta epomena...
columnsX = ['Daily_tests_lag_1', 'Daily_tests_lag_2', 'Daily_tests_lag_3','Population_lag_1', 'Population_lag_2', 'Population_lag_3','Cases_lag_1','Cases_lag_2','Cases_lag_3']
columnsY = ['Cases_3_days_ahead']
X = df_greece[['Daily_tests_lag_1', 'Daily_tests_lag_2', 'Daily_tests_lag_3','Population_lag_1', 'Population_lag_2', 'Population_lag_3','Cases_lag_1','Cases_lag_2','Cases_lag_3']]
#metavlites tis opoies tis times tis xrisimopoio gia na ekpaideyso to modelo
Y = df_greece[['Cases_3_days_ahead']]#metavliti stoxos pou theloume na provlepsoume

X=X.dropna()
Y=Y.dropna()#diagrafi null timon

##
X=X.values#oi algorithmoi trexoun me arithmitika mono dedomena opote pernoume tis times mono
Y=Y.values

#
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)#80% train-20%test diaxorismos
#X->metavlites eisodou ypoloipa attributes tou dataset, Y-> metavliti/ klash eksodou/stoxos ayto pou theloume na provlepsoume
#scaling-metasxitismos ton dedomenon gia kalyterh palindromisi
X_train_scaled = StandardScaler().fit_transform(X_train)
#
X_test_scaled = StandardScaler().fit_transform(X_test)
#
Y_train_scaled = Y_train.reshape(-1,1)#metatropi se 2d array
Y_train_scaled = StandardScaler().fit_transform(Y_train_scaled)
#
Y_test_scaled = Y_test.reshape(-1,1)##metatropi se 2d array
Y_test_scaled = StandardScaler().fit_transform(Y_test_scaled)
#create neural network model
model = Sequential()#akolouthiako modelo me 200 neyrones
model.add(LSTM(200, activation='tanh', input_shape = (X_train_scaled.shape[1],1 )))#synartisi energopoihshs neyroniko relu/tanh, 200 neyrones,
model.add(Dropout(0.2))
model.add(Dense(1))#stroma eksodou
model.compile(optimizer='adam', loss='mse')#synartisi mesou tetragonikou sfalmatos,veltistopoihsh adam,ayto tha to xrisimopoihso os metriki apodosis
#training
model.fit(X_train_scaled, Y_train_scaled,epochs=100,batch_size=32)
predicted_cases = model.predict(X_test_scaled)
predicted_cases = predicted_cases.reshape(-1,1)#metatropi se 2d array gia na borei na metasximatistei

predicted_cases = StandardScaler().fit_transform(predicted_cases)
##

Y_test_scaled = pd.DataFrame(Y_test_scaled,columns=columnsY)
predicted_cases = pd.DataFrame(predicted_cases,columns=columnsY)#metatropi ksana se dataframe
print('-----------------------\nR2 score is : \n',r2_score(Y_test_scaled,predicted_cases))

#visualization
plt.plot(Y_test_scaled.index,Y_test_scaled.values,label='Actual Cases', linestyle='-', marker='o')
plt.plot(Y_test_scaled.index,predicted_cases.values,label='Predicted Cases', linestyle='--', marker='x')#plotting actual vs predicted
#meso inverse transform boroume na doume tis actual(mh metasxhmatismenes times)
plt.title('Prediction result of RNN')#plot title
plt.xlabel('Actual values')
plt.ylabel('Predicted cases')
plt.show()#show plot