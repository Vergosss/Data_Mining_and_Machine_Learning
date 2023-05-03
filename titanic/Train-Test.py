#80 train 20 test
#train->dimiourgia modelou
#test->elegxos apotelesmaton
import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
numpy.random.seed(2)
x = numpy.random.normal(3,1,100)#100 times kanonikis katanomis me mesh timh 3 kai diaspora 1
y = numpy.random.normal(150, 40, 100)/x#100 times kanonikis katanomis me mesh timh 150 kai diaspora 100
plt.scatter(x,y)#diespeire ta data ston x kai y axones
plt.show()#deikse th grafiki
train_x = x[:80]#to 80% ton timon eisodou os training
train_y = y[:80]#to 80% ton timon eksodou os training
test_x = x[80:]#to 20% ton timon eisodou os test
test_y = y[80:]#to 20% ton timon eksodou os test
plt.scatter(train_x, train_y)
plt.show()
#elegxo an to modelo(ekpaideyetai) ikanopoioi KAI TA DYO dataset kai train kai test
plt.scatter(test_x, test_y)
plt.show()# x kai y idiou megethous opote kanena thema
#polyonimi palindromisi
model = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))#polionimiki palindromisi me modelo polionimo 4ou vathmou
axonas_x = numpy.linspace(0,6,100)
plt.scatter(train_x, train_y)
plt.plot(axonas_x, model(axonas_x))#/plotare ton x kai xose sto modelo ayta ta x
plt.show()
#antistixa gia to r^2 apo 0-1 0:katholou sisxetisi 1 full sysxetisi
print(r2_score(train_y, model(train_x)))#diafora alithinis eksodou-ektimisi(provlepsi(eisodou))
print(r2_score(test_y, model(test_x)))#kali sisxetisi
#provlepseis genika modelo(timi-list) klp..
print(model(5))
