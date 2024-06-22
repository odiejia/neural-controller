# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 09:26:38 2021

@author: lenovo
"""

import numpy as np

from myClass.myTrainingPointFile2D import TrainingPoints_MCMC
from myClass.myPredictPointsFile2D import predictionPoints2D

class myDataSet_MCMC():
    def __init__(self,parameters,myGaussianFPKequation):

        self.Trainingdata = TrainingPoints_MCMC(parameters.axis_par,myGaussianFPKequation)
        self.Predictdata = predictionPoints2D(parameters.axis_par)



