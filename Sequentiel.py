# -*- coding: utf-8 -*-
"""
Created on Fri May 21 13:53:39 2021

@author: DELL VOSTRO
"""

from projet_etu import Linear, MSELoss, Sigmoide, TanH
import numpy as np
import matplotlib.pyplot as plt

class Sequentiel():
    
    def __init__(self):
        self.modules=[] #Les modules du réseau
        self.data=[]       #Les données en entrée de chaque module
        
    def addModule(self, module):
        """
        Parameters
        ----------
        module : Module 
        -------
        """
        
        self.modules.append(module)
        
    def forward(self, dataX):
        """
        Parameters
        ----------
        dataX : données en entrée du réseau de neuronnes

        Returns
        -------
        data : données en sortie du réseau de neuronnes

        """
        data=dataX
        for module in self.modules:
            self.data.append(data)
            data=module.forward(data)
            
        return data
    
    def backward(self, gradLoss):
        """
        Parameters
        ----------
        gradLoss : Loss gradient

        Returns
        -------
        delta : delta du premier module du réseau de neuronnes
        """
        delta=gradLoss
        for i in range(len(self.modules)):
            if self.modules[-i-1].module_activation==False:
                self.modules[-i-1].backward_update_gradient(self.data[-i-1],delta)
            delta=self.modules[-i-1].backward_delta(self.data[-i-1],delta)
        
        return delta
    
    def updateParameters(self,gradient_step=1e-3):
        """
        Parameters
        ----------
        gradient_step : Pas de mise à jour des paramètres, optional
            The default is 1e-3.

        Returns
        -------
        None.
        """
        
        for module in self.modules :
            if module.module_activation==False :
                module.update_parameters()
            
   
"""--------------------------------------Test-------------------------------------------"""
if __name__=="__main__":
    
    dataX=np.array([[0,0],[15,20],[1,2],[11,10],[13,9],[3,4],[14,19],[6,2],[4,2],[17,13]]).reshape(10,2)
    dataY=np.array([0,1,0,1,1,0,1,0,0,1]).reshape(10,1)
    
    seq=Sequentiel()
    mse=MSELoss()
    
    seq.addModule(Linear(2,1))
    seq.addModule(TanH())
    seq.addModule(Linear(1,1))
    seq.addModule(Sigmoide())
    

    for i in range(1000):
        yhat=seq.forward(dataX)
        gradLoss=mse.backward(dataY, yhat)
        delta=seq.backward(gradLoss)
        seq.updateParameters()
            
    
    print(yhat)
    print("MSE =",np.mean(mse.forward(dataY,yhat)))


   
  
    plt.scatter(dataY,[i for i in range(len(dataY))],color='blue',label="Y")
    plt.scatter(yhat,[i for i in range(len(yhat))],color='red',label="Yhat")
    
    plt.title("Prédiction avec le réseau de neuronnes ")
    plt.legend()
    plt.show()


    
    