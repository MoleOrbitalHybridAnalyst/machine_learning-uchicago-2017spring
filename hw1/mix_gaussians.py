#!/bin/python
#python3 mix_gaussians.py data k mode max_iter output
#mode:
#   0: random initials
#   1: take kmeans++ as initials

import random
import numpy as np
from scipy import stats
import kmeans                   #import my kmeans subroutine for initialization
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

def initialize(data,k,mode):
    n=len(data)
    if mode==0:                         #random initials
        #assign initial centers (mu) like kmeans++
        refs=[]
        refs.append(data[random.randrange(n)])
        mindist2=np.sum((data-refs[0])**2,axis=1)
        index=range(n)
        for i in range(1,k):
            dist2=np.sum((data-refs[i-1])**2,axis=1)
            mindist2=np.minimum(mindist2,dist2)
            prob=mindist2/sum(mindist2)
            refs.append(data[np.random.choice(index,p=prob)])
        #assign uniform pi
        pi=np.ones(k)
        pi/=k
        #assign diagonal covar
        dim=len(data[0])
        covars=[]
        covar=np.cov(data.transpose())/k    #use the covariance of the whole
        for i in range(k):                  #data set as estimations for the
            covars.append(covar)            #covariance for each cluster
    else:                               #use 100 steps of kmeans++ as initials
        print("kmeans involved..")
        refs,gamma=kmeans.kmeans(data,k,1,100,None)
        p=np.zeros([n,k])
        for index,j in enumerate(gamma):
            p[index,j]=1
        pi,refs,covars=m_step(data,k,p)
    #return
    return [refs,pi,np.array(covars)]

def e_step(data,k,refs,covars,pi):          #update the p_ij
    n=len(data)
    p=np.zeros([n,k])
    for i in range(n):
        for j in range(k):
            p[i,j]=pi[j]*stats.multivariate_normal.pdf(data[i],refs[j],covars[j])
        p[i,:]=p[i,:]/sum(p[i,:])
    return p

def m_step(data,k,p):                       #update pi,refs,covars
    n=len(data)
    dim=len(data[0])
    pi=np.sum(p,axis=0)/n                   #update pi
    refs=np.dot(p.transpose(),data)
    for j in range(k):
        refs[j]/=(n*pi[j])
    covars=np.zeros([k,dim,dim])
    for j in range(k):
        for i in range(n):
            covars[j]+=p[i,j]*np.outer(data[i]-refs[j],data[i]-refs[j])
        covars[j]/=n
        covars[j]/=pi[j]
    #return
    return [pi,refs,covars]

def assign_cluster(data,p):                 #assign data point to cluster
    gamma=[]                                # by the largest probability
    for p_i in p:
        gamma.append(p_i.argmax())
    return gamma

def distortion(data,gamma,refs):            #compute j_{avg^2} based on
    javg=0                                  #maximum prob assignment of
    for index,x in enumerate(data):
        javg+=sum((x-refs[gamma[index]])**2)
    return javg

def distortionE(data,p,refs):                #compute the E[Javg]
    javg=0
    for i,pj in enumerate(p):
        for j,pij in enumerate(pj):
            javg+=pij*sum((data[i]-refs[j])**2)
    return javg

def llkE(data,k,refs,pi,covars,p):              #compute the E[likelihood]
    n=len(data)
    l=0
    for i in range(n):
        for j in range(k):
            l+=p[i,j]*(np.log(pi[j])+np.log(stats.multivariate_normal.pdf\
(data[i],refs[j],covars[j])) )
    return l

if __name__=="__main__":
    data=kmeans.read_data(argv[1])
    n=len(data)
    refs,pi,covars=initialize(data,int(argv[2]),int(argv[3]))
    if debug==1:
        print("initial:")
        print("pi:")
        print(pi)
        print("mu:")
        print(refs)
        print("sigma:")
        print(covars)
    p=e_step(data,int(argv[2]),refs,covars,pi)
    p_prev=p
    gamma=assign_cluster(data,p)
    pi,refs,covars=m_step(data,int(argv[2]),p)
    l_prev=llkE(data,int(argv[2]),refs,pi,covars,p)
    if debug==1:
        print("after 1 iteration:")
        print("p_ij:")
        print(p)
        print("pi:")
        print(pi)
        print("mu:")
        print(refs)
        print("sigma:")
        print(covars)
        print("cluster assignment:")
        print(gamma)
    elif debug==2:
        print("%d\t%f"%(1,l_prev))
    count=1
    while count<int(argv[4]):
        p=e_step(data,int(argv[2]),refs,covars,pi)
        pi,refs,covars=m_step(data,int(argv[2]),p)
        l=llkE(data,int(argv[2]),refs,pi,covars,p_prev)
        p_prev=p
        if ((l-l_prev)/l)**2 <= 1e-20:
            print("converged")
            break
        l_prev=l
        gamma=assign_cluster(data,p)
        count+=1
        if debug==1:
            print("after %d iterations:"%count)
            print("p_ij:")
            print(p)
            print("pi:")
            print(pi)
            print("mu:")
            print(refs)
            print("sigma:")
            print(covars)
            print("cluster assignment:")
            print(gamma)
        elif debug==2:
            print("%d\t%f"%(count,l))
    with open(argv[5],"w") as fp_out:
        for (index, cluster) in enumerate(gamma):
            for c in data[index]:
                fp_out.write("%f\t"%c)
            fp_out.write("%d\n"%cluster)
