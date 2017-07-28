#!/bin/python

import numpy as np
from PIL import Image,ImageDraw
from Feature import *
from Iimage import Iimage
from WeakLearner import WeakLearner
from AdaBoost import AdaBoost
from StrongLearner import StrongLearner
from Cascade import Cascade

wmax2=31
hmax2=63
wmax3=21
hmax3=63
wmax4=31
hmax4=31
imagesize=64
stridex=4
stridey=4
adaboostconverg=0.00001
cascadeconverg=0.002
maxweak=5
maxstrong=5

if __name__ == "__main__":
    iimages=[]
    labels=[]
    for i in range(2000):
        im=Image.open("./faces/face"+str(i)+".jpg")
        pix=im.load()
        im.close()
        iimage=Iimage(pix,0,0,64,64)
        iimages.append(iimage)
        labels.append(1)
        im=Image.open("./background/"+str(i)+".jpg")
        pix=im.load()
        im.close()
        iimage=Iimage(pix,0,0,64,64)
        iimages.append(iimage)
        labels.append(-1)
    labels=np.array(labels)
    features=[]
    for w in range(1,wmax2,3):
        for h in range(1,hmax2,3):
            for x in range(0,imagesize-2*w,stridex):
                for y in range(0,imagesize-h,stridey):
                    feature=Feature2(x,y,w,h,0)
                    features.append(feature)
    for w in range(1,hmax2,3):
        for h in range(1,wmax2,3):
            for x in range(0,imagesize-w,stridex):
                for y in range(0,imagesize-2*h,stridey):
                    feature=Feature2(x,y,w,h,1)
                    features.append(feature)
    for w in range(1,wmax3,3):
        for h in range(1,hmax3,3):
            for x in range(0,imagesize-3*w,stridex):
                for y in range(0,imagesize-h,stridey):
                    feature=Feature3(x,y,w,h,0)
                    features.append(feature)
    for w in range(1,hmax3,3):
        for h in range(1,wmax3,3):
            for x in range(0,imagesize-w,stridex):
                for y in range(0,imagesize-3*h,stridey):
                    feature=Feature3(x,y,w,h,1)
                    features.append(feature)
    for w in range(1,wmax4,3):
        for h in range(1,hmax4,3):
            for x in range(0,imagesize-2*w,stridex):
                for y in range(0,imagesize-2*h,stridey):
                    feature=Feature4(x,y,w,h)
                    features.append(feature)
    print("%d features in the pool"%len(features))
    passindex=range(len(iimages))
    cascade=Cascade()
    for i in range(maxstrong):
        print("constructing strong classifier # %d"%(i+1))
        subiimages=[iimages[k] for k in passindex]
        sublabels=[labels[k] for k in passindex]
        adaboost=AdaBoost(subiimages,sublabels)
        for j in range(maxweak):
            print("\tconstructing weak classifier # %d"%(j+1))
            adaboost.addlearner(features)
            stronglearner=StrongLearner(adaboost)
            if stronglearner.error !=None:
                print("\terror=%f"%stronglearner.error)
            else:
                print("no backgrounds left")
            if stronglearner.error < adaboostconverg or stronglearner.error is None:
                break
        cascade.addlearner(stronglearner)
        passindex=[]; error=0
        for k,(iimage,label) in enumerate(zip(iimages,labels)):
            if cascade.predict(iimage) > 0:
                passindex.append(k)
                if label < 0:
                    error+=1
            elif label > 0:
                error+=1
        error=error/len(iimages)
        print("error=%f"%error)
        if error < cascadeconverg:
            break
        if passindex == []:
            cascade.stronglearners=cascade.stronglearners[:-1]
            break
    im=Image.open("class.jpg")
    pix=im.load()
    facepatches=[]
    draw=ImageDraw.Draw(im)
    for x in range(0,im.size[0]-imagesize,10):
        for y in range(0,im.size[1]-imagesize,10):
            if cascade.predict(Iimage(pix,x,y,x+imagesize,y+imagesize,1))>0:
                facepatches.append([x,y])
                draw.line([(x,y),(x+imagesize,y),(x+imagesize,y+imagesize),(x,y+imagesize),(x,y)],width=1,fill='white')
    im.save("class_det.png","PNG")
    im.show()

