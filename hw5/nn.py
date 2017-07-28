#!/bin/python3
import numpy as np

class NN:
    def __init__(self,nlayer,nneurals,activ_func,dactiv_func,weights=None):
        assert nlayer>1,"number of layers should not be smaller than 2"
        self.nlayer=nlayer
        assert len(nneurals)==nlayer,"number of layers does not match the dim of nneurals"
        self.nneurals=nneurals
        self.func=activ_func  #activation function
        self.dfunc=dactiv_func #derivative of the activation function
        if weights is None: #if weights are not given explicitly, assign randomly
            self.w=[]
            for l in range(nlayer-1):
                #construct (l+1)-th layer's weights
                #w[l][i][j]=weight for combining output of j-th preneurals of (l+1)-th layer's i-th neural
                w_tmp=[] #w_tmp will be w[l]
                for i in range(nneurals[l]):
                    w_tmp.append(np.random.random_sample((nneurals[l+1],)))
                w_tmp=np.array(w_tmp)
                for i in range(nneurals[l+1]):
                    w_tmp[:,i]=w_tmp[:,i]/sum(w_tmp[:,i])
                print(len(w_tmp))
                self.w.append(w_tmp)
            self.w=np.array(self.w)
        else:
            self.w=weights
    def layereval(self,inp,l):
        if l==0:
            inp=np.array(inp)
            return [inp,inp]
        assert l>0
        #print(len(inp),len(self.w[l-1]))
        a=np.dot(inp,self.w[l-1])
        dfa=np.vectorize(self.dfunc)(a)
        out=np.vectorize(self.func)(a)
        if l<self.nlayer-1: #for hidden layers, the last neuron always return 1
            out[-1]=1;dfa[-1]=0
        return [out,dfa]
    def predict(self,inp):
        self.dfa=[];self.x=[]
        for l in range(self.nlayer):
            inp,dfa=self.layereval(inp,l)
            self.x.append(inp)
            if l>0:
                self.dfa.append(dfa)
        return  inp
    def train(self,inp,y,rate):
        assert self.nneurals[-1]==len(y)
        self.delta=[]
        for l in range(self.nlayer-1): #initialize delta
            self.delta.append(np.zeros(self.nneurals[l+1]))
        x=self.predict(inp) #prediction
        self.delta[-1]=(x-y)*self.dfa[-1]
        for dl in range(self.nlayer-2):
            l=self.nlayer-2-dl
            self.delta[l-1]=self.dfa[l-1]*np.dot(self.w[l],self.delta[l])
        for l in range(self.nlayer-1):
            self.w[l]-=rate*np.outer(self.x[l],self.delta[l])
