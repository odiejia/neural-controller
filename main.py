# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 15:24:24 2020
 
@author: WantaoJia
"""

import tensorflow as tf
import scipy.io as scio
from myClass.myMLFile import myML_manual
from myClass.myGenerateDataSet import myDataSet_MCMC
import parameters
from myPDEs import myGaussianFPK2D
from myClass.myControlNN import myControlNNAuto

import time

time_start=time.time()

################################################################################
myGaussianFPKequation=myGaussianFPK2D(parameters)
################################################################################
dataSet=myDataSet_MCMC(parameters,myGaussianFPKequation)

################################################################################
parameters.ml_parameter["lossFun"]= tf.keras.losses.MeanSquaredError()
################################################################################
myControl = myControlNNAuto(parameters)
################################################################################
myMLModel=myML_manual(dataSet,parameters,myGaussianFPKequation,myControl)

myMLModel.modelTrain()

history=myMLModel.history

c_f_pred=myControl.predict(
    tf.concat([dataSet.Predictdata.X1_pred_2d_mat2arr,
    dataSet.Predictdata.X2_pred_2d_mat2arr],1)
    )


control_1_f_pred = tf.reshape(c_f_pred[:,0],dataSet.Predictdata.X1_pred_2d_mat.shape)

control_2_f_pred = tf.reshape(c_f_pred[:,3],dataSet.Predictdata.X1_pred_2d_mat.shape)

time_end=time.time()

time_spend=time_end-time_start

savePDFData = 'control_1.mat'
scio.savemat(savePDFData, {'X1_mat':dataSet.Predictdata.X1_pred_2d_mat,
                           'X2_mat':dataSet.Predictdata.X2_pred_2d_mat,
                           'control_1_mat':control_1_f_pred.numpy()})

savePDFData = 'control_2.mat'
scio.savemat(savePDFData, {'X1_mat':dataSet.Predictdata.X1_pred_2d_mat,
                           'X2_mat':dataSet.Predictdata.X2_pred_2d_mat,
                           'control_2_mat':control_2_f_pred.numpy()})


saveHistory='history.mat'
scio.savemat(saveHistory,{'loss':history["loss"],
                              'lossf':history["loss_f"],
                              'losslayer':history["losslayer"],
                              })


tf.saved_model.save(myControl, "savedmodel/myControl")
tf.saved_model.save(myMLModel.myPINNmodel, "savedmodel/myMLModel")




