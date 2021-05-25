# -*- coding: utf-8 -*-
"""
Test du Modèle lineaire et optimisation avec une déscente de gradient

"""

import numpy as np
from projet_etu import MSELoss, Linear
import matplotlib.pyplot as plt
import random
from mltools import gen_arti, plot_frontiere, plot_data
from SGD import SGD
from Optim import Optim
from Sequentiel import Sequentiel
from sklearn.metrics import accuracy_score

"""
def plot():
    plot_frontiere(datatest,seq.predict,step=20)
    plot_data(datatest,labels=labeltest)
    plt.show()
    
    plot_frontiere(data,seq.predict,step=20)
    plot_data(data,labels=label)
    plt.show()
"""    


if __name__=="__main__":
  
    """---------------------------------------------Test------------------------------------"""
    """   
    dataX=np.array([[1],[3.5],[5],[6],[3],[8]])
    dataY=np.array([[1],[4],[4],[7],[3.2],[9.3]])
    
    
    mse=MSELoss()
    linear=Linear(1,1)
    
    iter_max=10000
    for i in range(iter_max):
        yhat=linear.forward(dataX)
        loss=mse.forward(dataY,yhat)
        lossGrad=mse.backward(dataY, yhat)
        linear.backward_update_gradient(dataX, lossGrad)
        linear.update_parameters()
        
     #plot_frontiere(data, f)   
       
        
       
     
    w=linear._parameters
    
    plt.scatter(dataX,dataY)
    x=[i/10 for i in range(100) ]
    plt.plot(x,w[0]*x+w[1],color="blue")
    plt.title("Regression lineaire par descente de gradient ")
    plt.show()
    """
    """---------------------------------------------Test sur des données linéaires------------"""   


    dataX, dataY = gen_arti(centerx=1,centery=1,sigma=0.3,nbex=1000,data_type=0,epsilon=0.02)
    datatest, labeltest = gen_arti(centerx=1,centery=1,sigma=0.3,nbex=400,data_type=0,epsilon=0.02)
    
    
    seq=Sequentiel()
    mse=MSELoss()
    
    seq.addModule(Linear(2,1))
  
    optim=Optim(seq, mse, eps=1e-3)
    sgd=SGD(optim)
    
    yhat,lossListe=sgd.fit(dataX, dataY, max_iter=1000)
    
    lossListe=np.asarray(lossListe).reshape(1000,1000)
    def f(x):
        return np.where(sgd.predict(x)>0.5,1,-1)
    
    print("Train accuracy =",accuracy_score( dataY,np.where(sgd.predict(dataX)>0,1,-1)))
    print("Test accuracy =",accuracy_score(labeltest,np.where(sgd.predict(datatest)>0,1,-1)))
    
    plot_frontiere(datatest,f)
    plot_data(datatest)
    plt.show()
    
   
    
    
    
    
    
    
    
    



