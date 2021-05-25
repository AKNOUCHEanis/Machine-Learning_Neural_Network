# -*- coding: utf-8 -*-
"""
Created on Mon May 24 22:11:20 2021

@author: DELL VOSTRO
"""
from projet_etu import Linear, MSELoss, Sigmoide, TanH, Softmax, CE, Softmax_stable, CElogSoftMax, CELoss, LogSoftmax
from Sequentiel import Sequentiel
from Optim import Optim
import numpy as np
import matplotlib.pyplot as plt
from SGD import SGD
from sklearn.metrics import confusion_matrix
import seaborn as sns

from mltools import load_usps, get_usps, show_usps

def oneHotEncoding(dataY):
    nbClasses=len(np.unique(dataY))
    
    res=np.zeros((dataY.shape[0],nbClasses))
    
    for i in range(len(dataY)):
        label=dataY[i]
        res[i,label]=1
        
    return res



"""------------------------ Test du multi-Classe ----------------------------------------"""

if __name__=="__main__":
    
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
 
    
    seq.addModule(Linear(256,nbClasses))
   
    seq.addModule(Softmax_stable())
    optim=Optim(seq,ce,eps=1e-4)
    
    sgd=SGD(optim)
    
    max_iter=300
    
    
  
    yhat, lossList=sgd.fit(dataX, dataY_OHE, max_iter)
    lossList=np.array(lossList).reshape(max_iter,dataX.shape[0])
    meanLossList=np.mean(lossList,axis=1)
    
    
    
    print("Train Score =",sgd.score(dataX, dataY_OHE))
    print("Train accuracy =",sgd.accuracy(dataX,dataY))
    
    print("Test Score =",sgd.score(testX, testY_OHE))
    print("Test accuracy =",sgd.accuracy(testX, testY))
    
    
     
    plt.plot([i for i in range(len(meanLossList))],meanLossList,color="Blue")
    plt.title("Cout cross-entropique ")
    plt.show()
    
    #Matrice de confusion
    yhat=sgd.predict(testX)
    
    yhatVector=[]
        
        
    for i in range(len(yhat)):
        label=list(yhat[i,:]).index(np.max(yhat[i,:]))
        yhatVector.append(label)
        
    cf_matrix=confusion_matrix( testY, yhatVector)
    sns.heatmap(cf_matrix, annot=True)
    
    
    