import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
carsinfo = pd.read_csv('cars.csv')
X = carsinfo[['Weight', 'Volume']] #aneksartites metavlites panta
Y = carsinfo['CO2']
reg = linear_model.LinearRegression()#antikeimeno modelou gia linear regression
reg.fit(X, Y)#fittaro sto modelo ta dedomena
#provlepo poso co2 tha ekpebei ena amaxi me varos 2300 kila kai ogko 1300 kyvika
print(reg.predict([[2300, 1300]]))
print(reg.coef_)#deixnei gia kathe aneksartiti metavliti poso tha allaksei to apotelesma an ayksithei kathe aneksartiti metavliti antistixa kata 1
print(reg.predict([[3300, 1300]]))#co2=107.20+1000*0.0075=114.75
scale = StandardScaler()#metasximatismos
x_ = scale.fit_transform(X)#metasximatise thn eisodo
reg.fit(x_, Y)#fitare/xose ta dedomena sto modelo
scaled = scale.transform([[2300, 1300]])#metasximatise th mia mono provlepsi diafora transform/fit transform
print(reg.predict([scaled[0]]))#typose thn provlepsi
#anti reg = LinearRegression tha exo reg = DecisionTreeClassifier meta reg.fit ...