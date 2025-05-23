from sklearn.svm import SVR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from IPython.display import display
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import r2_score
#from tensorflow import keras
#from tensorflow.keras import layers
from datetime import timedelta,date,datetime
import warnings
warnings.filterwarnings("ignore")

#df = pd.read_csv('test.csv')
df = pd.read_csv('data.csv')


Countries = []
Countries = df['Entity']
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
df_greece['Date'] = pd.to_datetime(df_greece['Date'])
end_date = pd.to_datetime(df_greece[df_greece['Date'] == '2021-01-01']['Date']) + timedelta(days=3)

print('----------------------------------\nEnd date is :',end_date)

###
columns_to_drop = ['Entity','Continent','Latitude','Longitude','Average temperature per year','Hospital beds per 1000 people','Medical doctors per 1000 people','GDP/Capita','Median age','Population aged 65 and over (%)','Deaths']

df_greece.drop(columns_to_drop,axis=1,inplace=True)#dropping everything except date,population, daily tests ,cases
print('----------------------------------\nDataframe:\n',df_greece)
#
endDate = end_date.iloc[0] #--->date we want to forecast
#print(df_greece.info(),'\n\n',df_greece.describe())
print(endDate)
#
print(df_greece[df_greece['Date'] == endDate])
##
df_greece=df_greece[df_greece['Date']>= '2021-01-01']#dedomena elladas meta tis 1/1/2021
print(df_greece)
#
#X=df_greece['ypoloipa']
#Y=df_greece['Cases'] mallon
#Y->metavliti/klash eksodou. Afou thelei pososto thetikotitas tote logika tha einai to cases
#creating lags
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
#############
print('----------------------------\nX is:\n',X)
print('----------------------------\nY is:\n',Y)
#drop null values
X=X.dropna()
Y=Y.dropna()#diagrafi null timon
print('----------------------------\nX is:\n',X)
#
print('----------------------------\nY is:\n',Y)
X=X.values#oi algorithmoi trexoun me arithmitika mono dedomena opote pernoume tis times mono
Y=Y.values
#scaling
#train-test split to build the model
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=42,shuffle=True)#80% train-20%test diaxorismos
#X->metavlites eisodou ypoloipa attributes tou dataset, Y-> metavliti/ klash eksodou/stoxos ayto pou theloume na provlepsoume
#scaling-metasxitismos ton dedomenon gia kalyterh palindromisi
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
#
X_test = scaler.transform(X_test)
#
Y_train = Y_train.reshape(-1,1)#metatropi se 2d array
Y_train =scaler.fit_transform(Y_train)
#
Y_test = Y_test.reshape(-1,1)##metatropi se 2d array
Y_test = scaler.transform(Y_test)
#build regressor
svr_regressor = SVR(kernel='rbf')#grammikos h mh grammikos pyrhnas tou svm linear,rbf ktlp
#dokimazoume grammiko kernel,polyonimiko,rbf k.o.k
svr_regressor.fit(X_train,Y_train)#vazo x_train,y_train-ekpaideyo to modelo -update: vazo tis scaled ekdoseis tous
#provlepsi-prediction-me ta test dedomena


predicted_cases = svr_regressor.predict(X_test)#provlepsi krousmaton
print(predicted_cases)
#predicted_cases = predicted_cases.reshape(-1,1)#metatropi se 2d array gia na borei na metasximatistei

#visualization
#an kanei kalh provlepsh tote h ta simia tis grafikis tha syglinoun se mia eytheia/palindromisi
#an den kanei kalh provlepsi tote tha diaspeirontai pio poly ta simia sth grafiki
###
Y_test = pd.DataFrame(Y_test,columns=columnsY)
predicted_cases = pd.DataFrame(predicted_cases,columns=columnsY)#metatropi ksana se dataframe
print('-----------------------\nR2 score is : \n',r2_score(Y_test,predicted_cases))
#print('r-square_SVR_Test: ', round(svr_regressor.score(predicted_cases, Y_test_scaled), 2))
#plot ta y_test(pragmatika) me ayta poy provlepsa
plt.plot(Y_test.index,Y_test.values,label='Actual Cases', linestyle='-', marker='o')
plt.plot(Y_test.index,predicted_cases.values,label='Predicted Cases', linestyle='--', marker='x')#plotting actual vs predicted
#meso inverse transform boroume na doume tis actual(mh metasxhmatismenes times)
plt.title('Prediction result of SVM')#plot title
plt.xlabel('Actual values')
plt.ylabel('Predicted cases')
plt.show()#show plot