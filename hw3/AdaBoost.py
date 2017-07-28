#!/bin/python

import numpy as np
from WeakLearner import WeakLearner
from functools import partial
from multiprocessing import Pool

class AdaBoost:
    def __init__(self,iimages,labels):
        self.iimages,self.labels=iimages,np.array(labels)
        self.nimage=len(iimages)
        self.depth=0
        #self.weights=np.ones(self.nimage)/self.nimage
        self.weights=np.ones(self.nimage)
        faceindex=np.where(self.labels >0)[0]
        backindex=np.where(self.labels <0)[0]
        if len(faceindex) !=0:
            self.weights[faceindex]/=(2*len(faceindex))
        else:
            self.weights=2*self.weights
        if len(backindex) !=0:
            self.weights[backindex]/=(2*len(backindex))
        else:
            self.weights=2*self.weights
        self.weaklearners=[]; self.alphas=[]
    def addlearner(self,featurepool,times=1):
        for t in range(times):
           self.depth+=1
           bestweaklearner=None; epsilon=self.nimage
#serial version:
#           for f in featurepool:
#               tmpweaklearner=WeakLearner(f,self)
#               if tmpweaklearner.epsilon <= epsilon:
#                   bestweaklearner=tmpweaklearner
#                   epsilon=bestweaklearner.epsilon
           func=partial(WeakLearner,adaboost=self)
           with Pool() as p:
               candidates=p.map(func,featurepool)
           for l in candidates:
               if l.epsilon <= epsilon:
                   bestweaklearner=l
                   epsilon=bestweaklearner.epsilon
           self.weaklearners.append(bestweaklearner)
           if epsilon==0:
               self.weaklearners=[bestweaklearner]
               self.alphas=np.array([1.0])
               return None
           else:
               alpha=1/2*np.log(1/epsilon-1)
           self.alphas.append(alpha)
           z=2*np.sqrt((epsilon*(1-epsilon)))
           predicts=[]
           for iimage in self.iimages:
               predicts.append(bestweaklearner.predict(iimage))
           self.weights=self.weights*np.exp(self.labels*np.array(predicts)*(-alpha))/z
    def predict(self,iimage):
        f=0
        for alpha,l in zip(self.alphas,self.weaklearners):
            f+=alpha*l.predict(iimage)
        if f>=0:
            return 1
        else:
            return -1
