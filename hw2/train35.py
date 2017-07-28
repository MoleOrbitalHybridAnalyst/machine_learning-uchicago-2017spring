#!/bin/python

import numpy as np
import perceptron as ppt            #import my perceptron subrotine

def testoneblock(data,M,label,block_size,index):
    #train M times then test on the index-th block
    mask=np.ones(len(data),dtype=np.bool)
    mask[block_size*index:block_size*(index+1)]=False
    clf=ppt.Perceptron(data[mask],label[mask])
    clf.perceptron()
    for i in range(M-1):
        clf.perceptron(data[mask],label[mask])
    plabels,mistakes,prob=clf.predict(data[block_size*index:block_size*(index+1)],\
label[block_size*index:block_size*(index+1)])
    return mistakes

def readdata(filename):
    with open(filename,"r") as fp:
        fulldata=[]
        for line in fp:
            xi=[]
            for xij in line.split():
                xi.append(float(xij))
            fulldata.append(np.array(xi))
        fulldata=np.array(fulldata)
    return fulldata

block_size=50
fulldata=readdata("train35.digits")
with open("train35.labels","r") as fp_labels:
    fulllabels=[]
    for line in fp_labels:
        fulllabels.append(float(line))
    fulllabels=np.array(fulllabels)
n=len(fulldata)
error=[]
for M in range(1,10):
    tmp=0
    for index in range(int(n/block_size)):
        tmp+=testoneblock(fulldata,M,fulllabels,block_size,index)
    error.append(tmp/float(n))
print("The optimal M < 10:")
M=np.argmin(error)+1
print(M)
for i,e in enumerate(error):
    print("%d\t%f"%(i+1,e))
clf=ppt.Perceptron(fulldata,fulllabels)
clf.perceptron()
for i in range(M-1):
    clf.perceptron(fulldata,fulllabels)
tdata=readdata("test35-1.digits")
plabels=clf.predict(tdata)
with open("test35.predictions","w") as fp:
    for label in plabels:
        print("%d"%label,file=fp)



