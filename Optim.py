# -*- coding: utf-8 -*-
"""
Created on Fri May 21 15:40:15 2021

@author: DELL VOSTRO
"""


from projet_etu import Linear, MSELoss, Sigmoide, TanH
from Sequentiel import Sequentiel
import numpy as np
import matplotlib.pyplot as plt
from mltools import gen_arti, load_usps

class Optim():
    
    def __init__(self,net,loss,eps):
        self.net=net
        self.loss=loss
        self.eps=eps
    
    def step(self, batch_x, batch_y):
        
        yhat=self.net.forward(batch_x)
        loss=self.loss.forward(batch_y, yhat)
        gradLoss=self.loss.backward(batch_y,yhat)
        delta=self.net.backward(gradLoss)
        self.net.updateParameters(self.eps)
        
        return yhat,loss
    
        
        
        
        
