#!/bin/python
#python3 dimreduc.py datafile dim_to_reduce num_of_neighbors

import numpy as np
from sys import argv

class DimReduc:
    #members:
    #pim: reduced dimension; M: matrix to be diagonalized
    #w: eigenvalues; v: eigenvectors; y: resulting points
    #n: number of data points; sort_index: index which
    #sorts the eigenvalues increasingly; center: center of
    #the data; nblist: k-nn list
    data=None;dim=None;pim=None;M=None;w=None;v=None;y=None
    n=None;sort_index=None;center=None;nblist=None;
    #methods:
    def __init__(self,data,pim):
        self.data=data
        self.pim=pim
        self.dim=len(data[0])
        self.n=len(data)
        self.y=[]
        self.center=[]
        for i in range(self.dim):
            self.center.append(np.mean(data[:,i]))
        self.center=np.array(self.center)
    def diago(self):
        self.w,self.v=np.linalg.eigh(self.M)
        self.v=self.v.T
        self.sort_index=np.argsort(self.w)
    def norm2(self):                    #contruct D matrix using Euclidean
        self.D=np.zeros([self.n,self.n])#distances
        for i,xi in enumerate(self.data):
            for j,xj in enumerate(self.data[i:]):
                self.D[i,j+i]=sum((xi-xj)**2)
        for i in range(self.n):
            for j in range(i):
                self.D[i,j]=self.D[j,i]
    def knn(self):                      #construct neighbour list
        self.nblist=[]
        for Di in self.D:
            self.nblist.append(np.argsort(Di)[:self.k+1])
    def symmetrize(self):
        #symmetrize the M matrix, this is needed sometimes
        self.M=(self.M+self.M.T)/2
    def print_results(self,labels,fp_out):
        for i,yi in enumerate(self.y):
            for yij in yi:
                fp_out.write("%f\t"%yij)
            fp_out.write("%d\n"%labels[i])

class Pca(DimReduc):            #principle component analysis
    #members:
    #no additional members for PCA
    #methods:
    def __init__(self,data,pim):
        super(Pca,self).__init__(data,pim)
        self.M=np.cov(data.T,bias=True)
        self.diago()
        for x in data:
            yi=[]
            for vec in self.v[self.sort_index[-self.pim:]]:
                yi.append(np.dot(x-self.center,vec))
            self.y.append(np.array(yi))

class Mds(DimReduc):
    #members:
    #dist: function to construct D;D: matrix used to construct G
    #P: projection matrix
    dist=None;D=None;P=None
    #methods:
    def __init__(self,data,pim,dist):
        super(Mds,self).__init__(data,pim)
        self.dist=dist
        self.dist(self)
        self.P=np.identity(self.n)-np.ones([self.n,self.n])/self.n
        self.M=-1/2*np.dot(np.dot(self.P,self.D),self.P)
        self.symmetrize()
        self.diago()
        for Qi in self.v.T:
            xi=[]
            for j,Qij in enumerate(Qi[self.sort_index[-pim:]]):
                xi.append(np.sqrt(self.w[self.sort_index[self.n-self.pim+j]])\
*Qij)
            self.y.append(np.array(xi))

class Isomap(Mds):
    #members:
    #k: number of neighbours;IntD: intermediate D calculated based on
    #Euclidean distance
    k=None;IntD=None
    #methods:
    def __init__(self,data,pim,k):
        self.k=k
        super(Isomap,self).__init__(data,pim,Isomap.floyd_warshall)
    def floyd_warshall(self):               #contruct D matrix using
        self.norm2()                        #floyd_warshall alg
        self.knn()                          #construct knn list
        self.IntD=self.D
        self.D=np.ones([self.n,self.n])*np.Inf
        for i in range(self.n):
            for j in self.nblist[i]:
                self.D[i,j]=np.sqrt(self.IntD[i,j])
        for k in range(self.n):
            for i in range(self.n):
                for j in range(self.n):
                    if self.D[i,j]>self.D[i,k]+self.D[k,j]:
                        self.D[i,j]=self.D[i,k]+self.D[k,j]
        self.D=(self.D)**2

class Lle(DimReduc):
    #members:
    #W: weight matrix; reg: regulization factor
    D=None;k=None;W=None;reg=-1e-6
    #methods:
    def __init__(self,data,pim,k):
        super(Lle,self).__init__(data,pim)
        self.k=k
        self.norm2()
        self.knn()
        self.W=np.zeros([self.n,self.n])
        for i in range(self.n):
            K=[]
            for j in self.nblist[i][1:]:
                Kj=[]
                for k in self.nblist[i][1:]:
                    Kj.append(np.dot(data[i]-data[j],data[i]-data[k]))
                K.append(np.array(Kj))
            Wi=np.linalg.solve(K+self.reg*np.identity(self.k),np.ones(self.k))
            Wi/=sum(Wi)
            for j,Wij in enumerate(Wi):
                self.W[i,self.nblist[i][j+1]]=Wij
        self.M=np.dot((np.identity(self.n)-self.W).T,np.identity(self.n)-self.W)
        self.diago()
        for i in self.sort_index[1:pim+1]:
            self.y.append(self.v[i])
        self.y=np.array(self.y).T

if __name__=="__main__":
    filename=argv[1]
    pim=int(argv[2])
    k=int(argv[3])
    data=[]; labels=[]
    with open(filename,"r") as fp_data:
        for line in fp_data:
            xi=[]
            for ele in line.split()[:-1]:
                xi.append(float(ele))
            data.append(xi); labels.append(int(line.split()[-1]))
    data=np.array(data)
    with open("pca_result.dat","w") as fp_pca:
        pca=Pca(data,pim)
        pca.print_results(labels,fp_pca)
   with open("mds_result.dat","w") as fp_mds:
        mds=Mds(data,pim,Mds.norm2)
        mds.print_results(labels,fp_mds)
    with open("isomap_result.dat","w") as fp_isomap:
        isomap=Isomap(data,pim,k)
        isomap.print_results(labels,fp_isomap)
    with open("lle_result.dat","w") as fp_lle:
        lle=Lle(data,pim,k)
        lle.print_results(labels,fp_lle)

