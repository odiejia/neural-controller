# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from pyDOE import lhs
    
class TrainingPoints_MCMC():
    def __init__(self,axis_par,pde):

        self.x1_lb = axis_par["x1_lb"]
        self.x1_ub = axis_par["x1_ub"]
        self.x1_f_step = axis_par["x1_f_step"]
        
        self.x2_lb = axis_par["x2_lb"]
        self.x2_ub = axis_par["x2_ub"]
        self.x2_f_step = axis_par["x2_f_step"]

        self.Num_training_LHS = axis_par["Num_training_LHS"]     
		
        self.GenerateTrainingPoints_LHS()
        
    def GenerateTrainingPoints_LHS(self):
		
        rng_mat = lhs(2,self.Num_training_LHS)        
        X1_f_lhs_arr = (rng_mat[:,0]*(self.x1_ub-self.x1_lb)).reshape([-1,1])+self.x1_lb
        X2_f_lhs_arr = (rng_mat[:,1]*(self.x2_ub-self.x2_lb)).reshape([-1,1])+self.x2_lb		
		
        self.X1_f_LHS_arr = tf.cast(np.vstack((X1_f_lhs_arr)),dtype=tf.float32)
        self.X2_f_LHS_arr = tf.cast(np.vstack((X2_f_lhs_arr)),dtype=tf.float32)
		
        self.u_f_LHS_arr = tf.zeros(self.X1_f_LHS_arr.shape)
        
        return 
