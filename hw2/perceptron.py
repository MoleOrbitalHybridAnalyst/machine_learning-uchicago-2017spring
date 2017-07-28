#!/bin/python

import numpy as np

class Perceptron:
    def __init__(self,data,labels):
        self.data=data
        self.labels=labels
        self.n=len(self.data)
        self.dim=len(self.data[0])+1
        #self.num=number of data points already fed
        self.num=0
        self.w=np.zeros(self.dim)
    def perceptron(self,data=None,labels=None):
        if data is None or labels is None:
            data=self.data
            labels=self.labels
            n=self.n
        else:
            n=len(data)
            self.data=np.append(self.data,data,axis=0)
            self.labels=np.append(self.labels,labels)
            self.n=len(self.data)
        data=np.append(data,np.ones([n,1]),axis=1)
        data=data/np.sqrt(np.sum(data**2,axis=1)).reshape([n,1])
        for xi,yi in zip(data,labels):
            if np.dot(self.w,xi)>=0:
                yhat=1
            else:
                yhat=-1
            if yhat*yi<0:
                self.w+=yi*xi
            self.num+=1
    def predict(self,tdata,tlabels=None):
        nt=len(tdata)
        plabels=[]
        tdata=np.append(tdata,np.ones([nt,1]),axis=1)
        tdata=tdata/np.sqrt(np.sum(tdata**2,axis=1)).reshape([nt,1])
        for i,txi in enumerate(tdata):
            if np.dot(self.w,txi)>=0:
                plabels.append(1)
            else:
                plabels.append(-1)
        if tlabels is None:
            return plabels
        else:
            diff=np.abs(tlabels-plabels)/2
            return [plabels,int(np.sum(diff)),np.sum(diff)/nt]
