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
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('test.csv')
#df = pd.read_csv('data.csv')

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
start_date = pd.to_datetime("2021-01-01")#ksekiname apo tis 1/1/2021
#Y->metavliti/klash eksodou. Afou thelei pososto thetikotitas tote
#scaling


#train-test split to build the model
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=42)#70% train-30%test diaxorismos
#X->metavlites eisodou ypoloipa attributes tou dataset, Y-> metavliti/ klash eksodou

#build regressor
svr_regressor = SVR(kernel='rbf')#grammikos h mh grammikos pyrhnas tou svm linear,rbf ktlp
svr_regressor.fit()#vazo x_train,y_train-ekpaideyo to modelo
#provlepsi-prediction-me ta test dedomena
predicted_output = svr_regressor.predict(input_test)
#isos metriki r2score(output_test,predicted_output)
#gia kalytero performance extra scaling

#visualization
#scatter(actual_output,predicted_output)
#an kanei kalh provlepsh tote h ta simia tis grafikis tha syglinoun se mia eytheia/palindromisi
#an den kanei kalh provlepsi tote tha diaspeirontai pio poly ta simia sth grafiki

