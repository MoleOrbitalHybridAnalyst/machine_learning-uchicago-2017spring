#!/bin/python
#python3 gendata.py a k n
#a: box size
#k: number of clusters
#n: number of points in each cluster

import numpy as np
import kmeans
from sys import argv

dim=10
a=float(argv[1])
k=int(argv[2])
n=int(argv[3])
centers=np.random.uniform(-a/2,a/2,[k,dim])
data=[]
cov=np.zeros([dim,dim])
for i in range(dim):
    cov[i,i]=1
for center in centers:
    for i in range(n):
        data.append(np.random.multivariate_normal(center,cov,1)[0])
for x in data:
    for xi in x:
        print("%f"%xi,end="\t")
    print()
