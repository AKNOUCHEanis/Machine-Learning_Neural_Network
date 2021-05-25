# -*- coding: utf-8 -*-

"""
Created on Fri May 21 16:23:34 2021

@author: DELL VOSTRO
"""

from projet_etu import Linear, MSELoss, Sigmoide, TanH, Softmax, CE, Softmax_stable, CElogSoftMax, CELoss, LogSoftmax 
from Sequentiel import Sequentiel
from Optim import Optim
import numpy as np
import matplotlib.pyplot as plt


from mltools import load_usps, get_usps, show_usps

def oneHotEncoding(dataY):
    nbClasses=len(np.unique(dataY))
    
    res=np.zeros((dataY.shape[0],nbClasses))
    
    for i in range(len(dataY)):
        label=dataY[i]
        res[i,label]=1
        
    return res
    

class SGD():
    
    def __init__(self,optim):
        self.optim=optim
        
    def fit(self, dataX, dataY, max_iter):
        lossListe=[]
        for i in range(max_iter):
            yhat,loss=self.optim.step(dataX,dataY)
            lossListe.append(loss)
            
        return yhat,lossListe
            
    def predict(self,dataX):
        return self.optim.net.forward(dataX)
     
    def score(self,dataX,dataY):
        #utilisation de la loss
        return np.mean(self.optim.loss.forward(dataY,self.predict(dataX)))
    
    def accuracy(self, dataX, dataY):
        
        yhat=self.predict(dataX)
        yhatVector=[]
        
        
        for i in range(len(yhat)):
                label=list(yhat[i,:]).index(np.max(yhat[i,:]))
                yhatVector.append(label)
        
        
        return np.where(yhatVector==dataY,1,0).sum()/len(dataY)
        
    
            
    

"----------------------------Test sur les données USPS ----------------------------"

if __name__ =="__main__":
    uspsdatatrain = "../data/USPS_train.txt"
    uspsdatatest = "../data/USPS_test.txt"
    alltrainx,alltrainy = load_usps(uspsdatatrain)
    alltestx,alltesty = load_usps(uspsdatatest)
    
    """----------------Avec les 10 classes de chiffres-----------------------------"""
    
    dataX=alltrainx
    dataY=alltrainy
    nbClasses=10 #nombre de classes de chiffres
    dataY.reshape(-1,1)
    
    print("shape datax: ",np.shape(dataX))
    print("shape datay: ",np.shape(dataY))
    
    
    testX=np.array(alltestx)
    testY=np.array(alltesty)
    
    
    """-----------------Avec 2 classes de chiffres------------------------------------"""
    """
    cinq = 5
    six = 6
    nbClasses=2 #nombre de classes de chiffres
    dataX,dataY = get_usps([cinq,six],alltrainx,alltrainy)
    
    print("shape datax: ",np.shape(dataX))
    print("shape datay: ",np.shape(dataY))
    
    testX,testY = get_usps([cinq,six],alltestx,alltesty)
    
    dataY=np.where(dataY==5,1,0)
    testY=np.where(testY==5,1,0)
    """
    """---------------------------------------------------------------------------------"""
    
    dataY_OHE=oneHotEncoding(dataY)
    testY_OHE=oneHotEncoding(testY)
    
    seq=Sequentiel()
    ce=CE()

    seq.addModule(Linear(256,10))
    seq.addModule(Softmax_stable())
    
    optim=Optim(seq,ce,eps=1e-4)
    
    sgd=SGD(optim)
    
    max_iter=300
    
    #fit
    sgd.fit(dataX, dataY_OHE, max_iter)
    
    
    #accuracy pour le train
    yhat=sgd.predict(dataX)
    print("yhat =\n",yhat)
    
    print("Train Score =",sgd.score(dataX, dataY_OHE))
    print("Train accuracy =",sgd.accuracy(dataX,dataY))
    
    print("Test Score =",sgd.score(testX, testY_OHE))
    print("Test accuracy =",sgd.accuracy(testX, testY))
    
    
    #one hot encoding
    
    yhatVector=[]
    for i in range(len(yhat)):
            label=list(yhat[i,:]).index(np.max(yhat[i,:]))
            yhatVector.append(label)
    #print("yhatvector=  \n",yhatVector)
    
    
    
    plt.scatter(dataY,[i for i in range(len(dataY))],color='blue',label="Y")
    plt.scatter(yhatVector,[i for i in range(len(yhatVector))],color='red',label="Yhat")
    
    plt.title("Prédiction avec le réseau de neuronnes ")
    plt.legend()
    plt.show()
    
    
    
    
    