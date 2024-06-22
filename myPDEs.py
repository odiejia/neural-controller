# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 09:38:22 2021

@author: lenovo
"""

import numpy as np
import tensorflow as tf
import scipy as sc
from scipy import integrate
################################################################################
class myGaussianFPK2D():
    def __init__(self,parameters):
##################################################################################################################
        self.parameters = parameters    
        self.K1 = tf.constant(parameters.system_par["K1"])
        self.K2 = tf.constant(parameters.system_par["K2"])
        self.K3 = tf.constant(parameters.system_par["K3"])
        self.K4 = tf.constant(parameters.system_par["K4"])
        self.A = tf.constant(parameters.system_par["A"])        
        self.B = tf.constant(parameters.system_par["B"])    
        self.D11 = tf.constant(parameters.system_par["D11"])   
        self.D12 = tf.constant(parameters.system_par["D12"]) 
        self.D21 = tf.constant(parameters.system_par["D21"]) 
        self.D22 = tf.constant(parameters.system_par["D22"]) 
##################################################################################################################
        self.x1_lb = parameters.axis_par["x1_lb"]
        self.x1_ub = parameters.axis_par["x1_ub"]
        self.x2_lb = parameters.axis_par["x2_lb"]
        self.x2_ub = parameters.axis_par["x2_ub"]
        
        self.C_normal_target = self.normalize_constant_fun(self.exp_phifun_targetfun)

################################################################################
    def normalize_constant_fun(self,exp_phi):
        C_temp = sc.integrate.dblquad(exp_phi,self.x2_lb, self.x2_ub, self.x1_lb, self.x1_ub)

        return C_temp[0]

################################################################################    
    def yf_addloss(self,x1_f,x2_f,myControl):
        
        c= myControl(tf.concat([x1_f,x2_f],1),training=True)
        psi1I = tf.reshape(c[:,0],shape=(-1,1))
        psi2I = tf.reshape(c[:,3],shape=(-1,1))
        
        ptarget = self.targetPDF(x1_f,x2_f)
###############################################################################        
        x1_fun_tar = (x1_f-self.parameters.tragetPDF_par["miu1"])/(self.parameters.tragetPDF_par["sigma1"]**2)
        
        f1 = self.K1*self.A - self.K2*self.B*x1_f + self.K3*x1_f**2*x2_f - self.K4*x1_f
        b11 = 2*self.D11 + 2*self.D12*x2_f**2

             
        temp10 = f1+psi1I
        temp11 = 0.5*b11*x1_fun_tar

        f_val1 = (temp10 +temp11)*ptarget
###############################################################################
        x2_fun_tar = (x2_f-self.parameters.tragetPDF_par["miu2"])/(self.parameters.tragetPDF_par["sigma2"]**2)
        
        f2=self.K2*self.B*x1_f - self.K3*x1_f**2*x2_f 

        b22 = 2*self.D21 + 2*self.D22*x1_f**2
             
        temp20 = f2+psi2I
        temp21 = 0.5*b22*x2_fun_tar

        f_val2 = (temp20 +temp21)*ptarget
  
        return f_val1,f_val2    

################################################################################
    def drift_coef(self,x1_f,x2_f):
        
        f1=self.K1*self.A - self.K2*self.B*x1_f + self.K3*x1_f**2*x2_f - self.K4*x1_f
        f2=self.K2*self.B*x1_f - self.K3*x1_f**2*x2_f 
        
        return f1,f2
    
    def diffusion_coef(self,x1_f,x2_f):
        
        g11 = tf.ones(x2_f.shape,dtype=tf.float32)
        g12 = x2_f
        
        g21 = tf.ones(x2_f.shape,dtype=tf.float32)
        g22 = x1_f
        
        return g11,g12,g21,g22

################################################################################
    def phifun_targetfun(self,x,y):
        miu1=self.parameters.tragetPDF_par["miu1"]
        miu2=self.parameters.tragetPDF_par["miu2"]
        sigma1=self.parameters.tragetPDF_par["sigma1"]
        sigma2=self.parameters.tragetPDF_par["sigma2"]
        phifun_target = -((x-miu1)**2/(2*sigma1**2)+(y-miu2)**2/(2*sigma2**2))
        return phifun_target
    
    def exp_phifun_targetfun(self,x,y):
        exp_phifun_target = np.exp(self.phifun_targetfun(x,y))
        return exp_phifun_target
    
    def targetPDF(self,x,y):
        pdf_garget=self.exp_phifun_targetfun(x,y)/self.C_normal_target
        
        return pdf_garget
        
    
        
