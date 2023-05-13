from sklearn import metrics
import matplotlib.pyplot as plt
import numpy
ideal = numpy.random.binomial(1,0.9,size=1000)
predicted = numpy.random.binomial(1,0.9,size=1000)
confusion_matrix = metrics.confusion_matrix(ideal, predicted)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False,True])
cm_display.plot()
plt.show()
accuracy = metrics.accuracy_score(ideal, predicted)#(true positive+ true negative)/provlepseis -> pote einai ontos(ontos mesa i provlepsi) sosto to modelo(aneksartitos lathous)
print(accuracy)
precision = metrics.precision_score(ideal, predicted)#precision=(true positive)/(true positive+false positive) apo ta alithi pou PROVLEFTHISAN OS ETSI. posa einai ontos alithina?
print(precision)
sensitivity_recall = metrics.recall_score(ideal, predicted)#true positive/(true positive+false negative) apo ola ta positives pou EINAI POSITIVES
print(sensitivity_recall)
specifity = metrics.recall_score(ideal,predicted,pos_label=0)#to antitheto tou sensitivity diladi true negative/(true negative+ false positive)->apo ta pseydi poia einai ontos pseydi
fscore = metrics.f1_score(ideal, predicted)#2*precision*sensitivity/(precision+sensitivity)-'mesos' metaxy precision kai sensitivity
print(fscore)
