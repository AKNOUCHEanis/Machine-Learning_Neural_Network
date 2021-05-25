# -*- coding: utf-8 -*-
"""
Test du Modèle non lineaire
"""

from projet_etu import TanH, Sigmoide, Linear, MSELoss, TanH
import numpy as np
import matplotlib.pyplot as plt

from mltools import gen_arti, plot_frontiere, plot_data
from SGD import SGD
from Optim import Optim
from Sequentiel import Sequentiel
from sklearn.metrics import accuracy_score




"""---------------------------------------Test sur des données non linéaires -------------"""

if __name__=="__main__":
    
    dataX, dataY = gen_arti(centerx=1,centery=1,sigma=0.3,nbex=1000,data_type=1,epsilon=0.02)
    datatest, labeltest = gen_arti(centerx=1,centery=1,sigma=0.3,nbex=400,data_type=1,epsilon=0.02)
    
    
    seq=Sequentiel()
    mse=MSELoss()
    
    seq.addModule(Linear(2,1))
    seq.addModule(TanH())
  
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


"""---------------------------------Test sur des données en XOR--------------------------"""
"""
if __name__=="__main__":
    
    dataX, dataY = gen_arti(centerx=1,centery=1,sigma=0.3,nbex=1000,data_type=2,epsilon=0.02)
    datatest, labeltest = gen_arti(centerx=1,centery=1,sigma=0.3,nbex=1000,data_type=2,epsilon=0.02)
    
    
    seq=Sequentiel()
    mse=MSELoss()
    
    seq.addModule(Linear(2,1))
    seq.addModule(TanH())
    seq.addModule(Linear(1,1))
    seq.addModule(TanH())
  
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
"""
