# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

axis_par={
    
    "x1_lb":0.4,"x1_ub":0.8,
    "x2_lb": 0.8,"x2_ub":1.2,
    
    "x1_f_step":0.01,
    "x2_f_step":0.01,

    "Num_training_LHS": 10000,
    
    "x1_pred_lb":0.4,"x1_pred_ub":0.8,
    "x2_pred_lb":0.8,"x2_pred_ub":1.2,
    "x1_pred_step":0.01,
    "x2_pred_step":0.01,
    
    }

system_par={
    "K1": 0.5,
    "K2": 0.3,
    "K3": 0.7,
    "K4": 1.0,
    "A": 1.0,
    "B": 1.5,

    "D11": 0.01,
    "D12": 0.001,
    "D21": 0.01,
    "D22": 0.001,
    }

stationary_par = {
    "alpha":1.0,
    "beta":1.0,
    "gamma":0.001,
    "Lambda_big":0.001,
    "Gamma_big_lb":2.0,
    "Gamma_big_ub":10.0,
    "Num_9":100,
    "Num_10":100,
    }

tragetPDF_par={
    "miu1":0.6,
    "miu2":1.0,
    "sigma1":0.08,
    "sigma2":0.08
    }

ml_parameter={
    "batch_size": 10000,
    "buffer_size": 10000,
    "epochs":20000,
    "learning_rate":0.005,
    "weight_f":1.0,
    "weight_uf":1.0,
    "weight_a1":1.0,
    "weight_a2":1.0,    
    "add_1":1.0,
    "add_2":1.0,
    }

NN_Control = {
    "NumOflayer":3,
    "NumOfNeuronEachLayer":[60,60,4],
    "ActivateFunOfEachLayer":['tanh','tanh','None'],
    }

