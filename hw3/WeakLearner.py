#!/bin/python

import numpy as np
import Feature

class WeakLearner:
    def __init__(self,f,adaboost):
        self.f=f
        if len(adaboost.iimages) == 0:
            return None
        values=[]
        for iimage in adaboost.iimages:
            values.append(f.compute_value(iimage))
        sigma=np.argsort(values)
        tplus=0; tminus=0
        for label,weight in zip(adaboost.labels,adaboost.weights):
            tplus+=(label+1)*weight/2
            tminus+=(1-label)*weight/2
        #initialize theta at smallest feature value over data set
        self.theta=values[sigma[0]]
        if tplus < tminus:
            self.p=-1; self.epsilon=tplus
        else:
            self.p=1; self.epsilon=tminus
        splus=0; sminus=0
        #for j in sigma[:-1]:
        for k in range(adaboost.nimage-1):
            j=sigma[k]
            splus+=(adaboost.labels[j]+1)*adaboost.weights[j]/2
            sminus+=(1-adaboost.labels[j])*adaboost.weights[j]/2
            tmp1,tmp2=[splus+tminus-sminus,sminus+tplus-splus]
            if tmp1 < self.epsilon:
                self.epsilon=tmp1
                #self.theta=(values[j]+values[j+1])/2
                self.theta=(values[j]+values[sigma[k+1]])/2
                self.p=1
            elif tmp2 < self.epsilon:
                self.epsilon=tmp2
                self.theta=(values[j]+values[sigma[k+1]])/2
                self.p=-1
    def predict(self,iimage):
        fvalue=self.f.compute_value(iimage)
        if self.p * (fvalue - self.theta) >=0:
            return 1
        else:
            return -1
