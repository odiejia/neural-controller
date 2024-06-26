# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 12:15:32 2020

@author: WantaoJia
"""

import tensorflow as tf
from myClass.myPINNFile import myPINN_manual

class myML_manual(): # 手动训练，但是数据传入使用dataSet传递
    def __init__(self,dataSet,parameters,myPDE,myControl):
#######################################################################
        self.dataSet = dataSet
#######################################################################
        self.myPINNmodel=myPINN_manual(self.dataSet,parameters,myPDE,myControl)
        self.history = None
        
    def modelTrain(self):
            
        self.history = self.myPINNmodel.myPINNtrain()
        
    def modelPrediction(self,x1_data,x2_data):
            
            x_data = tf.concat([x1_data,x2_data],1)
            
            y_pred=self.myPINNmodel.predict(x_data)
            return y_pred
        