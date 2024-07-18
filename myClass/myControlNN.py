# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

##############################################################################################################################################  
class myControlNNAuto(tf.keras.Model):
    def __init__(self,parameters):
        super().__init__()
        
        self.NN_parameters = NNparameters(parameters)
        self.x_l = np.array([parameters.axis_par["x1_lb"],parameters.axis_par["x2_lb"]])
        self.x_u = np.array([parameters.axis_par["x1_ub"],parameters.axis_par["x2_ub"]])
        
        self.NN_net_list = self.NN_layers(self.NN_parameters)

    @tf.function
    def call(self, input):
        x = input
        x = 2.0*(x-self.x_l)/(self.x_u-self.x_l)-1.0
        for i_layer in range(0,self.NN_parameters.NumOflayer):
            x = self.NN_net_list[i_layer](x)
            
        output = x
        return output

    def NN_layers(self,NN_parameters):
        
        NN_net_list = []
        
        for i_layer in range(0,NN_parameters.NumOflayer):
            
            numOfNeronsOfthislayer =  NN_parameters.NumOfNeuronEachLayer[i_layer]
            nameActiveFun = NN_parameters.ActivateFunOfEachLayer[i_layer]
            
            if nameActiveFun == 'tanh':
                activefun = tf.keras.activations.tanh
            elif nameActiveFun == 'exp':
                activefun = tf.keras.activations.exponential
            elif nameActiveFun == 'softmax':
                activefun = tf.keras.activations.softmax
            elif nameActiveFun == 'exp':
                activefun = tf.keras.activations.exponential
            elif nameActiveFun == 'None':
                activefun = None
            else:
                print('Please add this acitive function')
            
            denselayer = tf.keras.layers.Dense(units=numOfNeronsOfthislayer,
                                               activation=activefun)
            NN_net_list.append(denselayer)
            
        return NN_net_list


class NNparameters():
    def __init__(self,parameters):
        self.NumOflayer = parameters.NN_Control["NumOflayer"]
        self.NumOfNeuronEachLayer = parameters.NN_Control["NumOfNeuronEachLayer"]
        self.ActivateFunOfEachLayer = parameters.NN_Control["ActivateFunOfEachLayer"]
