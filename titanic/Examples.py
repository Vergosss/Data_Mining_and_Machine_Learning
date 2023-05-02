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
print(df['Cabin'])#typose ta stixia tou attribute cabin
deck=df['Cabin'].str[0]#vale sta stixia tou cabin MONO to proto tous gramma
df['Cabin']=deck#vale sta stixia tou cabin MONO to proto tous gramma
print(df.describe())#periegrapse kathe attribute tou dataset
#print(df)#ektypose to dataset
print(df['Cabin'])
lista=list(df['Cabin'].unique())#ftiakse mia lista me ta diakrita(to alfavito) tou attribute cabin
print(lista)#typose thn
map2={item: lista.index(item) for item in lista} #ftiaxnei ena dictionary opou stixio:thesi tou sti lista gia kathe stixio ths listas
df['Cabin']=df['Cabin'].map(map2)#antistixise ta
print(df['Cabin'].head(3))#typose ta prota 3 nea cabins
plt.figure('Male Passengers over age')
#sb.distplot(df[df['Survived']==1].Age.dropna(),label='Survived',bins=15,kde=False)#apo aytous pou zisane dikse ilikia kai rikse ta nulls se 15 kadous
#sb.distplot(df[df['Survived']==0].Age.dropna(),label='Not Survived',bins=15,kde=False)#apo aytous pou de zisane dikse ilikia kai rikse ta nulls se 15 kadous
#sb.distplot(df[df['Survived']==1 & df['Sex']==1].Age.dropna(),label='Survived',bins=15,kde=False)#apo aytous pou zisane dikse ilikia kai rikse ta nulls se 15 kadous
#anti afto kalytera:
men=df[df['Sex']==1]#pare tous antres mono apo to dataset
women=df[df['Sex']==0]#pare tis gynaikes mono apo to dataset
sb.distplot(men[men['Survived']==1].Age.dropna(),label='Survived',bins=15,kde=False)#apo tous antres pou zisane dikse ilikia kai rikse ta nulls se 15 kadous
sb.distplot(men[men['Survived']==0].Age.dropna(),label='Not Survived',bins=15,kde=False)#apo tous antres pou de zisane dikse ilikia kai rikse ta nulls se 15 kadous
plt.legend()
plt.title(label='Male Passengers over age')
plt.show()
##Correlations sex,age and survivabillity
plt.figure('Female Passengers over age')
sb.distplot(women[women['Survived']==1].Age.dropna(),label='Survived',bins=15,kde=False)#apo tis gynaikes pou zisane dikse ilikia kai rikse ta nulls se 15 kadous
sb.distplot(women[women['Survived']==0].Age.dropna(),label='Not Survived',bins=15,kde=False)#apo tis gynaikes pou de zisane dikse ilikia kai rikse ta nulls se 15 kadous
plt.legend()
plt.title(label='Female Passengers over age')
plt.show()
