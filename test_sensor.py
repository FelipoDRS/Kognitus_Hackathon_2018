# -*- coding: utf-8 -*-
"""
Scripting for simulating vibration data to evaluate Machine Learning model 
in well damage identification task 
@author: Group Klepper
"""
import csv
import numpy as np
import sklearn.svm as SVM
import matplotlib.pyplot as plt
import sklearn.model_selection as ml
n=np.random.rand(60*100) #noise
nm=np.sin(np.linspace(0,1,num=60*100)*10*np.pi)#motor noise
#x=np.genfromtxt('sensor.csv',delimiter=';')
#x=np.genfromtxt('dados_iniciais.csv',delimiter=';')
x1=np.copy(n+nm)
y=np.zeros(np.shape(x1))
for i in range(len(x1)):
    if np.random.rand(1)<0.01:
        x1[i]=x1[i]+10
        y[i]=1
         
plt.plot(np.linspace(0,1,num=60*100),x1)

MVS=SVM.SVC()
x_train, x_test, y_train, y_test=ml.train_test_split(np.reshape(x1,(6000,1)),np.reshape(y,(6000,)),test_size=0.5)
MVS.fit(x_train, y_train)

print(MVS.score(x_test, y_test))

x2=np.copy(n+nm)
x2=x2+np.sin(np.linspace(0,1,num=60*100)*200*np.pi)*10
y2=np.ones(np.shape(x2))*2
x0=np.copy(n+nm)+np.random.rand(60*100)*0.1 #noise
y0=np.zeros(np.shape(x0))
X=np.concatenate((x0,x1,x2))
Y=np.concatenate((y0,y,y2))

MVS2=SVM.SVC()
x_train, x_test, y_train, y_test=ml.train_test_split(np.reshape(X,(18000,1)),np.reshape(Y,(18000,)),test_size=0.5)
MVS2.fit(x_train, y_train)
print(MVS2.score(x_test, y_test)) #96.5% accuracy

np.savetxt('sem_ruido.csv',x0,delimiter=',')#Sharing data
np.savetxt('batida.csv',x1,delimiter=',')
np.savetxt('raspagem.csv',x2,delimiter=',')

