#!/bin/python3
#python3 kmenas.py data k mode max_iter output
# data:     input data
# k:        number of clusters
# mode:     mode selection
#               0: normal kmeans
#               1: kmeans++
# max_iter: max number of iterations
# output:   output file name

import random
import numpy as np
from sys import argv

debug=2

def read_data(filename):        #read data, return a numpy array
    data=[]
    with open(filename,"r") as fp_data:
        for line in fp_data:
            ele=[]
            for x in line.split():
                ele.append(float(x))
            data.append(ele)
    return np.array(data)

def initialize(data,k,mode):    #assign initial clustering centers
    if mode==0:                 #ordinary kmeans
        ref=[]
        for d in data.transpose():
            ref.append(np.random.uniform(np.amin(d),np.amax(d),k))
        return np.dstack(ref)[0]
    else:                       #kmeans++ chosen
        refs=[]
        refs.append(data[random.randrange(len(data))])
        mindist2=np.sum((data-refs[0])**2,axis=1)
        index=range(len(data))
        for i in range(1,k):
            dist2=np.sum((data-refs[i-1])**2,axis=1)
            mindist2=np.minimum(mindist2,dist2)
            prob=mindist2/sum(mindist2)
            refs.append(data[np.random.choice(index,p=prob)])
        return refs


def update_gamma(data,k,refs):  #given centers, assign clusters
    gammas=[]
    for x in data:
        small=sum((x-refs[0])**2)
        gamma=0
        index=1
        for ref in refs[1:]:
            dist2=sum((x-ref)**2)
            if dist2<=small:
                small=dist2
                gamma=index
            index+=1
        gammas.append(gamma)
    return gammas

def update_refs(data,k,gamma):  #given cluster assignment, update centers
    refs=np.zeros((k,len(data[0])))
    counts=np.zeros(k)          #number of points in one cluster
    for index,x in enumerate(data):
        refs[gamma[index]]+=x
        counts[gamma[index]]+=1
    for index in range(k):
        if counts[index]!=0:
            refs[index]/=counts[index]
        else:                   #this could happen in kmeans
            ref=[]
            for d in data.transpose():
                ref.append(np.random.uniform(np.amin(d),np.amax(d)))
            refs[index]=ref
    return refs

def distortion(data,gamma,refs):
    javg=0
    for index,x in enumerate(data):
        javg+=sum((x-refs[gamma[index]])**2)
    return javg

def kmeans(data,k,mode,max_iter,output):            #wrapper for kmeans
    refs=initialize(data,k,mode)
    if debug==1:    print(refs)
    #the first iteration
    gamma=update_gamma(data,k,refs)
    refs_prev=update_refs(data,k,gamma)
    if debug==1:
        print("after 1 iteration")
        print(gamma)
        print(refs_prev)
    elif debug==2:
        print("%d\t%f"%(1,distortion(data,gamma,refs_prev)))
    count=1                     #count for iterations
    while count<max_iter:
        gamma=update_gamma(data,k,refs_prev)
        refs=update_refs(data,k,gamma)
        if sum(sum((refs_prev-refs)**2)/k) <= 1e-15:
            print("converged")
            break
        refs_prev=refs
        count+=1
        if debug==1:
            print("after %d iterations"%count)
            print(gamma)
            print(refs)
        elif debug==2:
            print("%d\t%f"%(count,distortion(data,gamma,refs)))
    if(output!=None):
        with open(output,"w") as fp_out:
            for index,cluster in enumerate(gamma):
                for x in data[index]:
                    fp_out.write("%f\t"%x)
                fp_out.write("%d\n"%cluster)
    return [refs,gamma]

if __name__=="__main__":
    data=read_data(argv[1])
    if argv[5]=="None":
        kmeans(data,int(argv[2]),int(argv[3]),int(argv[4]),None)
    else:
        kmeans(data,int(argv[2]),int(argv[3]),int(argv[4]),argv[5])
