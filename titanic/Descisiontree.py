import pandas as pd
import sys
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib
import numpy
from matplotlib import pyplot as plt
#matplotlib.use('Agg')
df = pd.read_csv('dtree.csv')
print(df)
#arxika prepei ola ta dedomena na ginoun arithmitika
map1={"UK": 0, "USA": 1, "N": 2}
df['Nationality']=df['Nationality'].map(map1)
map2={"YES": 1, "NO": 0}
df['Go']=df['Go'].map(map2)
print(df)
#X Ta features Y o stoxos target
features=['Age', 'Experience', 'Rank', 'Nationality']
X=df[features]#ta features ola ektos tou stoxou
Y=df['Go']#o stoxos tou dataset
print(X)#typose attributes
print('\n')#nea grammh
print(Y)#typose stoxo
dtree=DecisionTreeClassifier()#o ml algorithmos einai to dentro apofashs
dtree.fit(X, Y)#fitaro/vazo ston algorithmo ta data
#tree.plot_tree(dtree, feature_names=features)#ta plotaro
#plt.savefig(sys.stdout.buffer)
#sys.stdout.flush()
print('[1]->yes and [0]->no. The answer is :')
print(dtree.predict([[40, 10, 7, 1]]))#provlepsi gia ta akoloutha dedomena
x=numpy.random.normal(5.0,1.0,1000)#1000 times pou akolouthoun kanoniki katanomi me mesh timh 5 kai typiki apoklisi 1
y=numpy.random.normal(10.0,2.0,1000)#1000 times pou akolouthoun kanoniki katanomi me mesh timh 10 kai typiki apoklisi 2
plt.scatter(x, y)# diespeire ta dataset ston x kai y antistixa
plt.show()#deikse to plot