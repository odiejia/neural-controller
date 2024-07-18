# -*- coding: utf-8 -*-

import numpy as np

class predictionPoints2D():
    def __init__(self,axis_par):
        
############################################################################################    
        self.x1_pred_lb = axis_par["x1_pred_lb"]
        self.x1_pred_ub = axis_par["x1_pred_ub"]
        self.x2_pred_lb = axis_par["x2_pred_lb"]
        self.x2_pred_ub = axis_par["x2_pred_ub"]
        self.x1_pred_step = axis_par["x1_pred_step"]
        self.x2_pred_step = axis_par["x2_pred_step"]
############################################################################################        
        self.X1_pred_2d_mat = None
        self.X2_pred_2d_mat = None

        self.X1_pred_2d_mat2arr = None
        self.X2_pred_2d_mat2arr = None
        
        self.GeneratePredPoints()
############################################################################################        
    def GeneratePredPoints(self):
        x1_pred_arr = np.arange(self.x1_pred_lb,self.x1_pred_ub+self.x1_pred_step,self.x1_pred_step)
        x2_pred_arr = np.arange(self.x2_pred_lb,self.x2_pred_ub+self.x2_pred_step,self.x2_pred_step)
        
        self.X1_pred_2d_mat,self.X2_pred_2d_mat = np.meshgrid( x1_pred_arr, x2_pred_arr)
        
        self.X1_pred_2d_mat2arr = self.X1_pred_2d_mat.reshape([-1,1])
        self.X2_pred_2d_mat2arr = self.X2_pred_2d_mat.reshape([-1,1])
        
        return
