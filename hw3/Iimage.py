#!/bin/python

import numpy as np
#from PIL import Image

def rgb2gray(rgb):
    return 0.299*rgb[0]+0.587*rgb[1]+0.114*rgb[2]

class Iimage:
    def __init__(self,pix,x1,y1,x2,y2,gray=0):
        self.width=x2-x1
        self.height=y2-y1
        s=np.zeros([self.width,self.height])
        self.ii=np.zeros([self.width,self.height])
        ii=0;jj=0
        for i in range(x1,x2):
            jj=0
            for j in range(y1,y2):
                if jj==0:
                    if gray==0:
                        s[ii,jj]=rgb2gray(pix[i,j])
                    else:
                        s[ii,jj]=pix[i,j]
                else:
                    if gray==0:
                        s[ii,jj]+=s[ii,jj-1]+rgb2gray(pix[i,j])
                    else:
                        s[ii,jj]+=s[ii,jj-1]+pix[i,j]
                if ii==0:
                    self.ii[ii,jj]=s[ii,jj]
                else:
                    self.ii[ii,jj]+=self.ii[ii-1,jj]+s[ii,jj]
                jj+=1
            ii+=1
    def compute_block(self,x,y,w,h):
        #compute the sum within one block
        return self.ii[x+w,y+h]+self.ii[x,y]-self.ii[x+w,y]-self.ii[x,y+h]

