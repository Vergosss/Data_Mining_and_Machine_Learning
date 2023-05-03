#psaxno sysxetisi metaxy metavliton
#ayth h sysxetish xrisimeyei sthn provlepsi gegonoton
#sth grammikh palindromisi einai mia eytheia grammh y=ax+b
import matplotlib.pyplot as plt #gia grafikes parastaseis
from scipy import stats #gia grammiki palindromisi
x = [5,7,8,7,2,17,2,9,4,11,12,9,6]#ilikia amaaxiou
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]#taxythta
plt.scatter(x, y)#diespeire ta data ston 2d xrono
plt.show()#deikse th grafiki
slope, intercept,  r, p,  std_err = stats.linregress(x,y)#ypologise th sysxetish metaxy ton dio metavliton (grammikh palindromisi),y=a*x+b=>a=slope klisi, intercept=b
#to r deixnei th sxesh metaxy tou x kai y. apo -1 eos 1. to 0 deixnei katholou sysxetisi to 1,-1 teleia sysxetisi
def calculate(x):
    return slope*x + intercept #y=a*x+b=>a=slope klisi, intercept=b
model = list(map(calculate, x))#proto orisma synartisi h opoia tha treksei se kathe stixio tou synolou x kai kanei 1-1 antistixisi
plt.scatter(x, y)#kane scatter to x kai y
plt.plot(x, model)# plot to x kai to model
plt.show()
print(r)#deikse ti sisxetisi -0.76 kalh sxetika
print(calculate(10))#provlepsi taxythtas gia 10 xronon amaksi
x_ = [89,43,36,36,95,10,66,34,38,20,26,29,48,64,6,5,36,66,72,40]
y_ = [21,46,3,35,67,95,53,72,58,10,26,34,90,33,38,20,56,2,47,15]
slope2, intercept2,  r2, p2,  std_err2 = stats.linregress(x_, y_)#ypologise th sysxetish metaxy ton dio metavliton (grammikh palindromisi),y=a*x+b=>a=slope klisi, intercept=b
def calculate2(x):
    return slope2*x + intercept2
model2 = list(map(calculate2, x_))
plt.scatter(x_, y_)
plt.plot(x_, model2)
plt.show()
print(r2)#0.01 poli mikro r2 - sysxetisi den sysxetizontai katholou x_,y_. Dyskolo na provlepso melontikes times


