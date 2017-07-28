#!/bin/python

from StrongLearner import StrongLearner

class Cascade:
    def __init__(self):
        self.stronglearners=[]
    def addlearner(self,stronglearner):
        self.stronglearners.append(stronglearner)
    def predict(self,iimage):
        for stronglearner in self.stronglearners:
            if stronglearner.predict(iimage) <= 0:
                return -1
        return 1

