# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 15:24:24 2020
 
@author: WantaoJia
"""

import tensorflow as tf
import scipy.io as scio
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d.axes3d import Axes3D
from myClass.myMLFile import myML_manual_v3
from myClass.myGenerateDataSet import myDataSet_MCMC
import parameters
from myPDEs import myGaussianFPK2D
from myClass.myControlNN import myControlNNAuto

import time

time_start=time.time()

scio.savemat('initial_time.mat',{'t':0})
################################################################################
myGaussianFPKequation=myGaussianFPK2D(parameters)
################################################################################
dataSet=myDataSet_MCMC(parameters,myGaussianFPKequation)

saveinitialData = 'training_sample_initial.mat'
scio.savemat(saveinitialData, {'x1_f':dataSet.Trainingdata.X1_f_MCMC_arr.numpy(),
                           'x2_f':dataSet.Trainingdata.X2_f_MCMC_arr.numpy(),
                           'x1_s':dataSet.Sampledata.X1_Sample_target_MCMC_arr.numpy(),
                           'x2_s':dataSet.Sampledata.X2_Sample_target_MCMC_arr.numpy(),
                           'y_s':dataSet.Sampledata.u_Sample_target_MCMC_arr})
################################################################################
parameters.ml_parameter["lossFun"]= tf.keras.losses.MeanSquaredError()
################################################################################
myControl = myControlNNAuto(parameters)
################################################################################
myMLModel=myML_manual_v3(dataSet,parameters,myGaussianFPKequation,myControl)

myMLModel.modelTrain()

history=myMLModel.history
#history=myMLModel.history.history

y_f_pred=myMLModel.modelPrediction(
    dataSet.Predictdata.X1_pred_2d_mat2arr,
    dataSet.Predictdata.X2_pred_2d_mat2arr
    )
   
u_f_pred_tf=tf.reshape(y_f_pred,dataSet.Predictdata.X1_pred_2d_mat.shape)

c_f_pred=myControl.predict(
    tf.concat([dataSet.Predictdata.X1_pred_2d_mat2arr,
    dataSet.Predictdata.X2_pred_2d_mat2arr],1)
    )

control_f_pred = tf.reshape(c_f_pred,dataSet.Predictdata.X1_pred_2d_mat.shape)


time_end=time.time()

time_spend=time_end-time_start

savePDFData = 'pdf.mat'
scio.savemat(savePDFData, {'X1_mat':dataSet.Predictdata.X1_pred_2d_mat,
                           'X2_mat':dataSet.Predictdata.X2_pred_2d_mat,
                           'pdf_mat':u_f_pred_tf.numpy()})

savePDFData = 'control.mat'
scio.savemat(savePDFData, {'X1_mat':dataSet.Predictdata.X1_pred_2d_mat,
                           'X2_mat':dataSet.Predictdata.X2_pred_2d_mat,
                           'control_mat':control_f_pred.numpy()})


saveHistory='history.mat'
scio.savemat(saveHistory,{'loss':history["loss"],
                              'lossf':history["loss_f"],
                              'lossu':history["loss_u"],
                              'losslayer':history["losslayer"],
                              'loss_sample':history["loss_sample"],
                              'loss_normalize':history["loss_normalize"],
                              'x1_f_adaptive':history["x1_f_adaptive"],
                              'x2_f_adaptive':history["x2_f_adaptive"],
                              'x1_s_adaptive':history["x1_s_adaptive"],
                              'x2_s_adaptive':history["x2_s_adaptive"],
                              'y_s_adaptive':history["y_s_adaptive"]
                              })

timeSpendData='time_spend.mat'
scio.savemat(timeSpendData,{'time_spend':time_spend,'time_start':time_start,'time_end':time_end})


myMLModel.myPINNmodel.save_weights("saved/myPINNmodel_weight/myPINNmodel_weight")
myControl.save_weights("saved/myControl_weight/myControl_weight")

tf.saved_model.save(myControl, "savedmodel/myControl/myControl")
tf.saved_model.save(myMLModel.myPINNmodel, "savedmodel/myPINN/myPINN")



