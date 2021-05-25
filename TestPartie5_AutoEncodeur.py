# -*- coding: utf-8 -*-
"""
Created on Mon May 24 02:46:48 2021

@author: DELL VOSTRO
"""


from projet_etu import Linear, MSELoss, Sigmoide, TanH, Softmax, CE, Softmax_stable, CElogSoftMax, CELoss, LogSoftmax, BCELoss
from Optim import Optim
from SGD import SGD
import numpy as np
import matplotlib.pyplot as plt
from mltools import load_usps 
from Sequentiel import Sequentiel

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split    
from sklearn.manifold import TSNE   
from sklearn.decomposition import PCA 
import seaborn as sns
import pandas as pd


class AutoEncodeur():
    
    def __init__(self,loss,eps):
        self.seqEncodeur=Sequentiel()
        self.seqDecodeur=Sequentiel()
        self.loss=loss
        self.eps=eps
        
        
    
    def addEncodeurModule(self,module):
        self.seqEncodeur.addModule(module)
        
    def addDecodeurModule(self,module):
        self.seqDecodeur.addModule(module)
        
    def fit(self,dataX,iter_max):
        
        for i in range(iter_max):
            
            
            yhatEncod=self.seqEncodeur.forward(dataX)
           
            yhatDecod=self.seqDecodeur.forward(yhatEncod)
            
            gradLossDecod=self.loss.backward(dataX,yhatDecod)
            deltaDecod=self.seqDecodeur.backward(gradLossDecod)
            
            deltaEncod=self.seqEncodeur.backward(deltaDecod)
            
            self.seqDecodeur.updateParameters(gradient_step=self.eps)
            self.seqEncodeur.updateParameters(gradient_step=self.eps)
            
        
            
    def compression(self, dataX):
        return self.seqEncodeur.forward(dataX)
    
    def decompression(self, dataCompressed):
        return self.seqDecodeur.forward(dataCompressed)
            
    def scoreLoss(self,dataX):
        
        yhatEncod=self.seqEncodeur.forward(dataX)
        yhatDecod=self.seqDecodeur.forward(yhatEncod)
        loss=self.loss.forward(dataX,yhatDecod)
        
        return loss
        

   
"""------------------------Test Auto-Encodeur----------------------------------------"""

if __name__=="__main__":
    """
    mnist = load_digits()

    datax, datay = (mnist.data, mnist.target)


    X_train, X_test, y_train, y_test = train_test_split(datax, datay, test_size=0.3)
  
    """
    uspsdatatrain = "../data/USPS_train.txt"
    uspsdatatest = "../data/USPS_test.txt"
    alltrainx,alltrainy = load_usps(uspsdatatrain)
    alltestx,alltesty = load_usps(uspsdatatest)
    
    
    """----------------Avec les 10 classes de chiffres-----------------------------"""
    
    dataX=alltrainx[:200]
    dataY=alltrainy[:200]
    nbClasses=10 #nombre de classes de chiffres
    dataY.reshape(-1,1)
    
    print("shape datax: ",np.shape(dataX))
    print("shape datay: ",np.shape(dataY))
    
    
    testX=np.array(alltestx)
    testY=np.array(alltesty)
    
    
    """-----------------------Creation de l'Auto-Encodeur-----------------------------"""
    input_dim=dataX.shape[1]
    #print("shape X :\n",X_train.shape)
    
    loss=BCELoss()
    
    autoEncodeur=AutoEncodeur(loss,eps=1e-3)
    
    autoEncodeur.addEncodeurModule(Linear(input_dim,100))
    autoEncodeur.addEncodeurModule(TanH())
   
    autoEncodeur.addDecodeurModule(Linear(100,input_dim))
    autoEncodeur.addDecodeurModule(Sigmoide())
        
    #dataX=X_train
    iter_max=1000
    
    autoEncodeur.fit(dataX, iter_max)
    
    print("Score loss :\n",autoEncodeur.scoreLoss(dataX))
    
    #Exemple de compression et décompression d'une image correspondant au label number
    number=list(dataY).index(9)
    
    print("Image originale :\n")
    plt.imshow(dataX[number].reshape(16,16))
    plt.show()
    
    print("Image compressée :\n")
    dataCompressed=autoEncodeur.compression(dataX)
    plt.imshow(dataCompressed[number].reshape(25,4))
    plt.show()
    
    print("Image décompressée :\n")
    dataDecompressed=autoEncodeur.decompression(dataCompressed)
    plt.imshow(dataDecompressed[number].reshape(16,16))
    plt.show()
    
    """--------------Visualisation avec PCA -------------------------------"""

    """
    #dataX=dataDecompressed #pour une pca sur l'espace projeté
    feat_cols = [ 'pixel'+str(i) for i in range(dataX.shape[1]) ]
    df = pd.DataFrame(dataX,columns=feat_cols)
    df['y'] = dataY
    df['label'] = df['y'].apply(lambda i: str(i))   
        
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(df[feat_cols].values)
    df['pca-one'] = pca_result[:,0]
    df['pca-two'] = pca_result[:,1] 
    df['pca-three'] = pca_result[:,2]
    
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    #Explained variation per principal component: [0.09746116 0.07155445 0.06149531]  
    rndperm = np.random.permutation(df.shape[0])
    
    #Plot 2D PCA
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="pca-one", y="pca-two",
        hue="y",
        palette=sns.color_palette("hls", 10),
        data=df.loc[rndperm,:],
        legend="full",
        alpha=0.3
    )
        
    #Plot 3D PCA
    ax = plt.figure(figsize=(16,10)).gca(projection='3d')
    ax.scatter(
        xs=df.loc[rndperm,:]["pca-one"], 
        ys=df.loc[rndperm,:]["pca-two"], 
        zs=df.loc[rndperm,:]["pca-three"], 
        c=df.loc[rndperm,:]["y"], 
        cmap='tab10'
    )
    ax.set_xlabel('pca-one')
    ax.set_ylabel('pca-two')
    ax.set_zlabel('pca-three')
    plt.show()
    """        
        
        
        
        
       