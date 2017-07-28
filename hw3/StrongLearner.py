#!/bin/python

import numpy as np

class StrongLearner:
    def __init__(self,adaboost):
        faceindex=np.where(adaboost.labels > 0)[0]
        self.Theta=adaboost.alphas[0]
        for i in faceindex:
            f=0
            for alpha,l in zip(adaboost.alphas,adaboost.weaklearners):
                f+=alpha*(l.predict(adaboost.iimages[int(i)]))
            if f< self.Theta:
                self.Theta=f
        self.Theta=-self.Theta
        self.alphas,self.weaklearners=[adaboost.alphas,adaboost.weaklearners]
        backindex=np.where(adaboost.labels < 0)[0]
        p=[]
        for i in backindex:
            p.append(self.predict(adaboost.iimages[int(i)]))
        if len(backindex) !=0:
            self.error=sum(np.array(p)+1)/2/len(backindex)
        else:
            self.error=None
    def h(self,iimage):
        f=self.Theta
        for alpha,l in zip(self.alphas,self.weaklearners):
            f+=alpha*l.predict(iimage)
        return f
    def predict(self,iimage):
        f=self.h(iimage)
        if f>0:
            return 1
        else:
            return -1

