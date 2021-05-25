import numpy as np



class Loss(object):
    def forward(self, y, yhat):
        pass

    def backward(self, y, yhat):
        pass


class Module(object):
    
    def __init__(self,dim_entree,dim_sortie):
        self.dim_entree=dim_entree
        self.dim_sortie=dim_sortie
        #le biais est placé à la derniere colonne de X
        #dans les parametres il represente la derniere ligne
        self._parameters= np.random.normal(0,1/(self.dim_entree+1),(self.dim_entree+1,self.dim_sortie) )
        #self._parameters = np.random.rand(self.dim_entree+1,self.dim_sortie) 
        self._gradient=None
        self.zero_grad()

    def zero_grad(self):
        ## Annule gradient
        self._gradient=np.zeros(self._parameters.shape)
        

    def forward(self, X):
        ## Calcule la passe forward
        pass

    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        self._parameters -= gradient_step*self._gradient

    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        pass

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        pass
    

class MSELoss(Loss):
    
    def forward(self,y,yhat):
        return (y-yhat)**2
    
    def backward(self,y,yhat):
        return -2*(y-yhat)

class Linear(Module):
    
    def __init__(self,dim_entree,dim_sortie):
        self.module_activation=False
        self.dim_entree=dim_entree
        self.dim_sortie=dim_sortie
        #le biais est placé à la derniere colonne de X
        #dans les parametres il represente la derniere ligne
        self._parameters= np.random.normal(0,1/(self.dim_entree+1),(self.dim_entree+1,self.dim_sortie) )

        self._gradient=None
        self.zero_grad()
    
    def forward(self, X):
        
        assert X.shape[1] == self.dim_entree 
        #Ajout du biais
        
        bias=np.ones((X.shape[0],1))
        X=np.hstack((X,bias))
       
        return np.dot(X, self._parameters)
    
    def backward_update_gradient(self, input_, delta):
        ## Met a jour la valeur du gradient
        #Ajout du biais
        bias=np.ones((input_.shape[0],1))
        input_=np.hstack((input_,bias))
        self._gradient+=np.dot(input_.T,delta)/input_.shape[0]



    def backward_delta(self, input_, delta):
        ## Calcul la derivee de l'erreur
        
        assert self.dim_sortie==delta.shape[1]
        return np.dot(delta,self._parameters.T)[:,:-1]
    
   
    
class TanH(Module):
    
    def __init__(self):
        self.module_activation=True
        
    
    def forward(self, X):
        """
        Parameters
        ----------
        X : input de taille (n,m)

        Returns : une matrice de taille (n,m) Tanh(X)
        -------

        """
        return np.tanh(X)
    
    
    def backward_delta(self, input_, delta):
        """
        Parameters
        ----------
        input_ : matrice de taille (n,m)
        delta : delta de la couche suivante

        Returns : une matrice qui correspond au gradient du cout par rapport aux entrées
        ------- 
        """
        
        sortie = self.forward(input_)
        
        return (1-np.square(sortie))*delta
        
      
    

class Sigmoide(Module):
    
    def __init__(self):
        self.module_activation=True
    
    def forward(self, X):
        """
        Parameters
        ----------
        X : input de taille (n,m)

        Returns : une matrice de taille (n,m) Sigmoid(X)
        -------
        """
        
        return 1. / (1. + np.exp(-X))
    
    def backward_delta(self, input_, delta):
        """
        Parameters
        ----------
        input_ : matrice de taille (n,m)
        delta : delta de la couche suivante

        Returns : une matrice qui correspond au gradient du cout par rapport aux entrées
        -------
        """
        
        sortie = self.forward(input_)
        return (sortie * (1. - sortie))* delta
    
    
class Softmax(Module):
    
    def __init__(self):
        #C'est un module d'activation qu'on va utiliser à la sortie de notre réseau
        self.module_activation=True
    
    def forward(self, input_):
        """
        Parameters
        ----------
        input_ : matrice de taille (n,m) 

        Returns 
        -------
        softmax(input_) matrice de taille (n, m)

        """
        
        res=np.exp(input_)*(1. /np.sum(np.exp(input_),axis=1)).reshape(-1,1)
        
      
        return res
    
    def backward_delta(self, input_, delta):
        """
        Parameters
        ----------
        input_ : matrice de taille (n,m)
        delta : delta de la couche suivante

        Returns : une matrice qui correspond au gradient du cout par rapport aux entrées
        -------
        """
        
        delta= self.forward(input_)*(1-self.forward(input_))*delta
       
        return delta
    
class Softmax_stable(Softmax):
    
    def forward(self, input_):
        """
        On va régulariser l'expo pour eviter un overflow

        Parameters
        ----------
        input_ : matrice de taille (n,m) 

        Returns 
        -------
        softmax(input_) matrice de taille (n, m)

        """
        
        input_=input_ - np.max(input_,axis=1,keepdims=True)
        sum_exp=np.sum(np.exp(input_),axis=1,keepdims=True)
        return np.exp(input_)/sum_exp
    
class LogSoftmax(Softmax):
    
    def forward(self,x):
        c = x.max()
        logsumexp = np.log(np.sum(np.exp(x - c),axis=1)).reshape(-1,1)
        return x - c - logsumexp
    
class CE(Loss):
    
    def forward(self,y,yhat):
        assert(y.shape==yhat.shape)
    
        return np.sum(-y*np.maximum(np.log(yhat+1e-3),-100),axis=1, keepdims=True)
        
    def backward(self,y,yhat):
        assert(y.shape==yhat.shape)
        
        return yhat-y
        
class CELoss(Loss):
    
    def forward(self, y, yhat):
        assert(y.shape==yhat.shape)
    
        return 1 -np.sum(yhat*y, axis=1)
    
    def backward(self, y, yhat):
        assert(y.shape==yhat.shape)
    
        return -y



class CElogSoftMax(Loss):
    
    def forward(self, y, yhat):
        assert(y.shape == yhat.shape)

        return np.log(np.sum(np.exp(yhat),axis=1)) -np.sum(y*yhat,axis=1)

    def backward(self, y, yhat):
        assert(y.shape==yhat.shape)
        
        exp=np.exp(yhat)
        return exp/np.sum(exp,axis=1).reshape((-1,1)) -y
        


  
    
class BCELoss(Loss):
    
    def forward(self, y, yhat):
        
        return -(y * np.maximum(np.log(yhat+1e-3),-100)+ (1 - y) * np.maximum(np.log(1 - yhat+1e-3),-100))

    def backward(self, y, yhat):
        
        return -(y / (yhat+1e-3) - (1 - y) / (1 - yhat + 1e-3))
    


    