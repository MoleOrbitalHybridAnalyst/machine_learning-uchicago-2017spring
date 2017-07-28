#!/bin/python3
#python3 10digit.py eta Nlayer Nhidden Nepoch PredTest1.dat PredTest2.dat errfile.dat
import numpy as np
from sys import argv
from nn import NN

def sig(x):
    return 1/(1+np.exp(-x))

def dsig(x):
    return np.exp(-x)/(1+np.exp(-x))**2

def int2vec(x):
    vec=np.zeros(10)
    vec[x]=1
    return vec

#def trainonce(nn,rate):
#    for trainx,trainy in zip(trainxs,trainys):
#        y=int2vec(trainy)
#        nn.train(np.append(trainx,[1]),y,rate)
#    return nn
#
#def error(nn):
#    error=0
#    for testx,testy in zip(testxs,testys):
#        y=int2vec(testy)
#        error+=sum((nn.predict(np.append(testx,[1]))-y)**2)
#    return error/len(testx)

if __name__=="__main__":
    trainxs=np.loadtxt("./TrainDigitX.csv.gz",delimiter=',')
    trainys=np.loadtxt("./TrainDigitY.csv.gz",delimiter=',',dtype=int)
    testxs=np.loadtxt("./TestDigitX.csv.gz",delimiter=',')
    testys=np.loadtxt("./TestDigitY.csv.gz",delimiter=',',dtype=int)
    test2xs=np.loadtxt("./TestDigitX2.csv.gz",delimiter=',')
    rate=float(argv[1])
    errfile=open(argv[7],"w")
    nneurals=[785]
    for i in range(int(argv[2])-2):
        nneurals.append(int(argv[3])+1)
    nneurals.append(10)
    nn=NN(int(argv[2]),nneurals,sig,dsig)
    errors=[];count=1
    for n in range(int(argv[4])): #loop for epoch
        for trainx,trainy in zip(trainxs,trainys):
            y=int2vec(trainy)
            nn.train(np.append(trainx,1),y,rate)
            error=sum((nn.predict(np.append(trainx,1))-y)**2)
            print(count,"\t",error,file=errfile)
            errors.append(error)
            count+=1
        print(error)
        pred=open(argv[5]+"_"+str(n)+".dat","w")
        pred2=open(argv[6]+"_"+str(n)+".dat","w")
        testmis=0
        for testx,testy in zip(testxs,testys):
            y=int2vec(testy)
            p=np.argmax(nn.predict(np.append(testx,1)))
            print(p,file=pred)
            if p!=testy:
                testmis+=1
        print("mistake rate on test set=",testmis/len(testxs))
        for testx in test2xs:
            p=np.argmax(nn.predict(np.append(testx,1)))
            print(p,file=pred2)
        pred.close()
        pred2.close()
    errfile.close()
