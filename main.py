from sklearn.ensemble import RandomForestClassifier
from func import imputation_uncertainty,multiple_imputation,missing_value_generator,produce_NA,importance,class_var,trainmodel,testmodel
import pandas as pd
from sklearn import model_selection
import heapq
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import random

data_df = pd.read_csv('./dataset/makedata.csv', delimiter=',', header=None)
data = data_df.values

#split dataset
c_train,c_test = model_selection.train_test_split(data,test_size = 0.3)

traindata=c_train
new_train,new_test = model_selection.train_test_split(traindata,test_size = 0.3)

X_train1 = new_train[:,1:]
Y_train1 = new_train[:,0]
X_test1=new_test[:,1:]
Y_test1 = new_test[:,0]
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train1, Y_train1)




X_test = c_test[:,1:]
Y_test = c_test[:,0]
#Calculate the accurancy without miss data
Y_pred = classifier.predict(X_test)
j=0
for i in range(len(Y_test)):
    if Y_test[i]==Y_pred[i]:
        j+=1
acc_full=j/len(Y_test)
print("Accurancy with full data:",j/len(Y_test))

#generate some miss data in testset
#X_miss=missing_value_generator(X_test,30,1)
X_miss=produce_NA(X_test, 0.3, mecha="MCAR", opt=None, p_obs=None, q=None)

#get the accurancy with miss value
X_imputed, imputed_list, mice=multiple_imputation(X_miss, 2, 1)

Y_predwithmiss = classifier.predict(X_imputed)

j=0
for i in range(len(Y_test)):
    if Y_test[i]==Y_predwithmiss[i]:
        j+=1
print("Accurancy with missing value:",j/len(Y_test))


#get trainmodel and testmodel
testmodel=testmodel(X_miss,classifier)
trainmodel=trainmodel(X_test1,Y_test1,classifier)

#input()
#using linerRegression to predict the importance
trainmodel= pd.read_csv('trainmodel.csv', delimiter=',', header=None)
trainmodel.drop(index=0, inplace=True)
trainmodel = trainmodel.values
testmodel= pd.read_csv('testmodel.csv', delimiter=',', header=None)

#testmodel.drop(index=0, inplace=True)
testmodel = testmodel.values

importance=importance(trainmodel,testmodel)
#print(importance)

tr_importance=imputation_uncertainty(imputed_list)
tr_importance=tr_importance.tolist()



class_importance=class_var(imputed_list,classifier)
class_importance=class_importance.tolist()



#maximp = list(map(importance.index, heapq.nlargest(1000, importance)))



plt.ylim(0.1,0.6)
plt.ylim(0.65,0.9)

plt.legend('DATASET')  #显示上面的label
plt.xlabel('The proportion of data obtained manually') #x_label
plt.ylabel('Accuracy')#y_label

x_axis_data = []
y_axis_data = []
y_axis_data1 = []
y_axis_data2 = []
y_axis_data3 = []
y_axis_data4 = []
y_axis_data5 = []
k=0
while k<=0.6:
    X_imputed1=X_imputed.copy()
    maximp = list(map(importance.index, heapq.nlargest(int(len(importance)*k), importance)))
    for i in maximp:
    
        X_imputed1[i]=X_test[i]


    Y_predwithmiss = classifier.predict(X_imputed1)

    j=0
    for i in range(len(Y_test)):
        if Y_test[i]==Y_predwithmiss[i]:
            j+=1
    #print("Accurancy with ac:",j/len(Y_test))
    accuracy=j/len(Y_test)
    x_axis_data.append(k)
    y_axis_data.append(accuracy)
    k+=0.1
    
    
plt.plot(x_axis_data, y_axis_data, 'b*-', alpha=0.5, linewidth=2, label='Most important')
#plt.show()
k=0
while k<=0.6:
    X_imputed1=X_imputed.copy()
    maximp = list(map(tr_importance.index, heapq.nlargest(int(len(importance)*k), tr_importance)))

    for i in maximp:
    
        X_imputed1[i]=X_test[i]


    Y_predwithmiss = classifier.predict(X_imputed1)

    j=0
    for i in range(len(Y_test)):
        if Y_test[i]==Y_predwithmiss[i]:
            j+=1
    #print("Accurancy with ac:",j/len(Y_test))
    accuracy=j/len(Y_test)
    #x_axis_data.append(k)
    y_axis_data4.append(accuracy)
    k+=0.1
    
    
plt.plot(x_axis_data, y_axis_data4, 'y*-', alpha=0.5, linewidth=2, label='Only imputer_var')






k=0
while k<=0.6:
    X_imputed1=X_imputed.copy()
    maximp = list(map(importance.index, heapq.nsmallest(int(len(importance)*k), importance)))
    for i in maximp:
    
        X_imputed1[i]=X_test[i]


    Y_predwithmiss = classifier.predict(X_imputed1)

    j=0
    for i in range(len(Y_test)):
        if Y_test[i]==Y_predwithmiss[i]:
            j+=1
    #print("Accurancy with ac:",j/len(Y_test))
    accuracy=j/len(Y_test)
    #x_axis_data.append(k)
    y_axis_data1.append(accuracy)
    k+=0.1
    
    
plt.plot(x_axis_data, y_axis_data1, 'ro-', alpha=0.5, linewidth=2, label='Least important')

k=0
while k<=0.6:
    X_imputed1=X_imputed.copy()
    maximp = list(map(importance.index, random.sample(importance,int(len(importance)*k))))
    for i in maximp:
    
        X_imputed1[i]=X_test[i]


    Y_predwithmiss = classifier.predict(X_imputed1)

    j=0
    for i in range(len(Y_test)):
        if Y_test[i]==Y_predwithmiss[i]:
            j+=1
    print("Accurancy with ac:",j/len(Y_test))
    accuracy=j/len(Y_test)
    #x_axis_data.append(k)
    y_axis_data2.append(accuracy)
    k+=0.1
    
plt.plot(x_axis_data, y_axis_data2, 'gs-', alpha=0.5, linewidth=2, label='Random choose')

k=0
while k<=0.6:
    y_axis_data3.append(acc_full)
    k+=0.1
    
plt.plot(x_axis_data, y_axis_data3, 'c.-', alpha=0.5, linewidth=2, label='Full data')


plt.legend()
plt.show()

"""X_imputed1=X_imputed.copy()
X_imputed2=X_imputed.copy()


maximp = list(map(importance.index, heapq.nlargest(int(len(importance)*0.5), importance)))

for i in maximp:
    
    X_imputed1[i]=X_test[i]


Y_predwithmiss = classifier.predict(X_imputed1)

j=0
for i in range(len(Y_test)):
    if Y_test[i]==Y_predwithmiss[i]:
        j+=1
print("Accurancy with ac:",j/len(Y_test))"""


"""minimp = list(map(importance.index, heapq.nsmallest(int(len(importance)*0.5), importance)))
for i in minimp:
    
    X_imputed2[i]=X_test[i]
    
#print(X_imputed)

Y_predwithmin = classifier.predict(X_imputed2)

j=0
for i in range(len(Y_test)):
    if Y_test[i]==Y_predwithmin[i]:
        j+=1
print("Accurancy with ac:",j/len(Y_test))"""



"""X = X_test
missing_rate=30
seed = 1
X_missing=missing_value_generator(X, missing_rate, seed)


m_imputations=5
X_imputed, imputed_list, mice=multiple_imputation(X_missing, m_imputations, seed)

class_var=class_var(imputed_list,classifier)
#print(class_var[:100],max(class_var))
#input()




Y_predwithmiss = classifier.predict(X_imputed)

j=0
for i in range(len(Y_test)):
    if Y_test[i]==Y_predwithmiss[i]:
        j+=1
print("Accurancy with missing value:",j/len(Y_test))


'''Train the liner regressive model and do the prediction'''


trainmodel = trainmodel.values
testmodel = testmodel.values
train_X=trainmodel[:,1:]
train_Y=trainmodel[:,0]
test_X=testmodel[:,1:]
test_Y=testmodel[:,0]
print('1')
input()

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(train_X , train_Y)

preimp=model.predict(test_X)
print(preimp)
input()


#importance
importance=imputation_uncertainty(imputed_list)+class_var*100
importance=importance.tolist()
#maximp 存储着最大importance的索引



maximp = list(map(importance.index, heapq.nlargest(1000, importance)))
for i in maximp:

    
    X_imputed[i]=X_test[i]
    
#print(X_imputed)

Y_predwithmiss = classifier.predict(X_imputed)

j=0
for i in range(len(Y_test)):
    if Y_test[i]==Y_predwithmiss[i]:
        j+=1
print("Accurancy with ac:",j/len(Y_test))"""
