import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score #opos sth gramiki exo to r edo exo to r^2 gia th sysxetisi ton
x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]
plt.scatter(x, y)#diespeire stous x kai y ta dedomena
plt.show()#deikse to plot
model = np.poly1d(np.polyfit(x, y, 3))#to trito orisma einai o vathmos tou polyonymou
axonas_x = np.linspace(1, 22, 100)#axonas ton x ena linespace
plt.scatter(x, y)#diespeire ston x
plt.plot(axonas_x, model(axonas_x))#plotare ston x ta dedomena me th kabylh tou modelou
plt.show()#deikse th grafiki
print(r2_score(y, model(x)))#2 orismata eksodos-modelo(eisodos)-poli konta sto 1 kalh sysxetisi
#provlepsi: vazo mia timi sto model
print(model(17))
#dedomena pou de tairiazoun sto polionimo regression
x_ = [89,43,36,36,95,10,66,34,38,20,26,29,48,64,6,5,36,66,72,40]
y_ = [21,46,3,35,67,95,53,72,58,10,26,34,90,33,38,20,56,2,47,15]
model2 = np.poly1d(np.polyfit(x_, y_, 3))#modelo polionimikis palindromisis
axonas_x = np.linspace(2, 95, 100)#axonas x linspace
plt.scatter(x_, y_)#diespeire ston x kai y ta dedomena
plt.plot(axonas_x, model2(axonas_x))#plotare ton axona x(ena linspace) me to modelo
plt.show()#deikse th grafiki
print(r2_score(y_, model2(x_)))#eksodos-modelo(eisodos)-mikri konta sto 0 mikrh sysxetisi oxi kali gia provlepsis