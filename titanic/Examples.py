import seaborn as sb
from matplotlib import pyplot as plt
import pandas as pd
dataset='train.csv'
df=pd.read_csv(dataset)
df.info()#gia kathe sthlh deixnei ton arithmo,onoma,mh nulls,typo dedomenon
print(df.describe())#gia kathe arithmitiki non null kathe sthlhs vres count,meso,std,elaxisto,megisto,diastimata
print(df.isnull().sum().sort_values(ascending=False))#vres posa(athrisma 1+1..) stixia kathe stilis einai null kai typose te se fthinousa
df['Age']=df['Age'].fillna(df['Age'].mean())#osa Age tou dataset exoun null timi siblirose ta me thn mesh timh ton mh null stixion
df.to_csv('train2.csv')
print(df.isnull().sum().sort_values(ascending=False))#vres posa(athrisma 1+1..) stixia kathe stilis einai null kai typose te se fthinousa
df['Cabin']=df['Cabin'].fillna('U')#osa cabin einai null ta siblirono me to U
df.to_csv('train2.csv')
print(df.isnull().sum().sort_values(ascending=False))#vres posa(athrisma 1+1..) stixia kathe stilis einai null kai typose te se fthinousa
#print(df['Embarked'].mode()[0])
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])#osa embarked einai null siblirose ta me to pio sixno(megaliteri pithanotita emfanisis)
df.to_csv('train2.csv')
print(df.isnull().sum().sort_values(ascending=False))#vres posa(athrisma 1+1..) stixia kathe stilis einai null kai typose te se fthinousa
df['Sex']=df['Sex'].map({'female':0, 'male':1})#antistixisi tou female se 0 kai tou male se 1
print(df['Sex'].head(10))#ektyposi ton proton 10 stixion tou sex attribute
print(df['Embarked'].describe())#perigrafh tou attribute embarked
df=pd.get_dummies(df,columns=['Embarked'])#gia kathe grammi tou embarked deikse an einai true h false gia kathe mia apo tis dynates times tou embarked
print(df.head())#typose ta prota 5