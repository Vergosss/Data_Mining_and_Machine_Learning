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
date_df = df.copy()
date_df['Date'] = pd.to_datetime(date_df['Date'])
date_df['Date'] = pd.to_datetime(date_df['Date']) + timedelta(days=3)
print(df['Date'])
print(date_df['Date'])
