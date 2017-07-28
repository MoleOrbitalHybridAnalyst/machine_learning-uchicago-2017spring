#!/bin/python

class Feature2:
    #class for two rectangle feature
    def __init__(self,x,y,w,h,mode):
        #iimage: integral image; mode: type of feature, 0: two-rectangles;
        #x1,y1,x2,y2: starting coordinate; w,h: size of the rectangle
        self.x1,self.y1,self.w,self.h=[x,y,w,h]
        if mode==0:
            #horizontal feature
            self.x2,self.y2=[x+w,y]
        else:
            #vertical feature
            assert mode==1, "unknown type for Feature2"
            self.x2,self.y2=[x,y+h]
    def compute_value(self,iimage):
        return (iimage.compute_block(self.x2,self.y2,self.w,self.h)-\
        iimage.compute_block(self.x1,self.y1,self.w,self.h))/self.w/self.h

class Feature3:
    def __init__(self,x,y,w,h,mode):
        self.x1,self.y1,self.w,self.h=[x,y,w,h]
        if mode==0:
            self.x2,self.y2,self.x3,self.y3=[x+w,y,x+2*w,y]
        else:
            assert mode==1, "unknown type for Feature3"
            self.x2,self.y2,self.x3,self.y3=[x,y+h,x,y+2*h]
    def compute_value(self,iimage):
        return (iimage.compute_block(self.x3,self.y3,self.w,self.h)+\
        iimage.compute_block(self.x1,self.y1,self.w,self.h)/2-\
        iimage.compute_block(self.x2,self.y2,self.w,self.h))/self.w/self.h

class Feature4:
    def __init__(self,x,y,w,h):
        self.x1,self.y1,self.w,self.h=[x,y,w,h]
        self.x2,self.y2,self.x3,self.y3,self.x4,self.y4=[x+w,y,x,y+h,x+w,y+h]
    def compute_value(self,iimage):
        return (iimage.compute_block(self.x1,self.y1,self.w,self.h)+\
        iimage.compute_block(self.x4,self.y4,self.w,self.h)-\
        iimage.compute_block(self.x2,self.y2,self.w,self.h)-\
        iimage.compute_block(self.x3,self.y3,self.w,self.h))/self.w/self.h
