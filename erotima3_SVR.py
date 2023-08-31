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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datetime import timedelta
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

print(end_date)

#X=df_greece['ypoloipa']
#Y=df_greece['Cases'] mallon
#Y->metavliti/klash eksodou. Afou thelei pososto thetikotitas tote logika tha einai to cases-cases/population kati tetio


#train-test split to build the model
#X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=42)#70% train-30%test diaxorismos
#X->metavlites eisodou ypoloipa attributes tou dataset, Y-> metavliti/ klash eksodou

#scaling
#---->X_train_scaled = StandardScaler().fit_transform(X_train)
#---->X_test_scaled = StandardScaler().fit_transform(X_test)
#build regressor
svr_regressor = SVR(kernel='rbf')#grammikos h mh grammikos pyrhnas tou svm linear,rbf ktlp
svr_regressor.fit()#vazo x_train,y_train-ekpaideyo to modelo -update: vazo tis scaled ekdoseis tous
#provlepsi-prediction-me ta test dedomena
predicted_output = svr_regressor.predict(input_test)
#isos metriki r2score(output_test,predicted_output)
#gia kalytero performance extra scaling

#visualization
#scatter(actual_output,predicted_output)
#an kanei kalh provlepsh tote h ta simia tis grafikis tha syglinoun se mia eytheia/palindromisi
#an den kanei kalh provlepsi tote tha diaspeirontai pio poly ta simia sth grafiki

