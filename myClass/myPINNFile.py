# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 15:24:24 2020

@author: WantaoJia
"""
import numpy as np
import tensorflow as tf

class myPINN_manual(tf.keras.Model): 
    #由四个边界构成的loss，采用dataSet传递数据，分别计算每个边的loss，手动训练
    def __init__(self,dataSet,parameters,myPDE,myControl):
        super().__init__(parameters)
###############################################################################
        self.dataSet = dataSet
        self.parameters = parameters
        self.myControl = myControl

        self.PDE = myPDE
###############################################################################
        self.weight_f = parameters.ml_parameter["weight_f"]
        self.weight_uf = parameters.ml_parameter["weight_uf"]
        
        self.add_1 = parameters.ml_parameter["add_1"]
        self.add_2 = parameters.ml_parameter["add_2"]
        
        self.weight_a1 =  parameters.ml_parameter["weight_a1"]
        self.weight_a2 =  parameters.ml_parameter["weight_a2"]        
        
        self.loss = []
        self.loss_f = []
        self.losslayer = []

        self.lr = []
        self.loss_add_1 = []
        self.loss_add_2 = []
###############################################################################
        self.learning_rate=parameters.ml_parameter["learning_rate"]
        self.buffer_size=parameters.ml_parameter["buffer_size"]
        self.batch_size=parameters.ml_parameter["batch_size"]
        self.epochs=parameters.ml_parameter["epochs"]
###############################################################################
        self.lossfunNet = parameters.ml_parameter["lossFun"]
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

###############################################################################
        self.x_mat_stationary = tf.random.uniform([self.parameters.stationary_par["Num_9"],2],
                          minval=self.parameters.stationary_par["Gamma_big_lb"],
                          maxval=self.parameters.stationary_par["Gamma_big_ub"],
                          dtype=tf.float32)
        self.index_mat_stationary = np.random.uniform(0,1,[self.parameters.stationary_par["Num_9"],2])
        self.index_mat_stationary[self.index_mat_stationary>=0.5]=1
        self.index_mat_stationary[self.index_mat_stationary<0.5]=-1 #这两行代码不能交换
        
        self.x_mat_stationary = self.x_mat_stationary * self.index_mat_stationary
        self.x1_arr_stationary = tf.reshape(self.x_mat_stationary[:,0],[-1,1])
        self.x2_arr_stationary = tf.reshape(self.x_mat_stationary[:,1],[-1,1])
###############################################################################
        self.x_mat_assum2 = tf.random.uniform([self.parameters.stationary_par["Num_10"],2],
                                              minval=(self.parameters.axis_par["x1_lb"],self.parameters.axis_par["x2_lb"]),
                                              maxval=(self.parameters.axis_par["x1_ub"],self.parameters.axis_par["x2_ub"]),
                                              dtype=tf.float32)
        self.x1_mat_assum2 = tf.reshape(self.x_mat_assum2[:,0],[-1,1])
        self.x2_mat_assum2 = tf.reshape(self.x_mat_assum2[:,1],[-1,1])
        
        self.y_mat_assum2 = tf.random.uniform([self.parameters.stationary_par["Num_10"],2],
                                              minval=(self.parameters.axis_par["x1_lb"],self.parameters.axis_par["x2_lb"]),
                                              maxval=(self.parameters.axis_par["x1_ub"],self.parameters.axis_par["x2_ub"]),
                                              dtype=tf.float32)
        self.y1_mat_assum2 = tf.reshape(self.y_mat_assum2[:,0],[-1,1])
        self.y2_mat_assum2 = tf.reshape(self.y_mat_assum2[:,1],[-1,1])

###############################################################################
        self.x1_f=self.dataSet.Trainingdata.X1_f_LHS_arr
        self.x2_f=self.dataSet.Trainingdata.X2_f_LHS_arr
                
        self.x_f_dataset=tf.data.Dataset.from_tensor_slices((self.x1_f,self.x2_f))
###############################################################################     
    def myPINNtrain(self):
        
        for i in range(1,self.epochs+1):
            self.x_f_dataset_p=self.x_f_dataset.shuffle(self.buffer_size).batch(self.batch_size)
            
            if (i % 5000 == 0 ):
                self.optimizer.learning_rate= self.optimizer.learning_rate / 5
            for step, (x1_batch_tf,x2_batch_tf) in enumerate(self.x_f_dataset_p):
                with tf.GradientTape() as tape:
                    loss, lossf,loss_add_1,loss_add_2,loss_a1, loss_a2 =self.lossfun_regular(x1_batch_tf,x2_batch_tf)
        
                trainable_vars = self.trainable_variables
                grads=tape.gradient(loss,trainable_vars)
                self.optimizer.apply_gradients(grads_and_vars=zip(grads,trainable_vars))
            
            if (i % 500 ==0 ):
                print("---------------------------------")
                print("epochs_num:",i)
                print("loss:",loss.numpy())
                print("lossf:",lossf.numpy())
                print("loss_add_1:",loss_add_1.numpy())
                print("loss_add_2:",loss_add_2.numpy())
                print("loss_a1:",loss_a1.numpy())
                print("loss_a2:",loss_a2.numpy())
                print("learning_rate:",self.optimizer.get_config()['learning_rate'])
                
                self.loss.append(loss.numpy())
                self.loss_f.append(lossf.numpy())
                self.loss_add_1.append(loss_add_1.numpy())
                self.loss_add_2.append(loss_add_2.numpy())
                self.lr.append(self.optimizer.get_config()['learning_rate'])
                 
        return {"loss":np.array(self.loss),"loss_f":np.array(self.loss_f),"learning_rate":np.array(self.lr),
                "losslayer": np.array(self.losslayer), "loss_add_1":np.array(self.loss_add_1),"loss_add_2":np.array(self.loss_add_2)
                }
    
###############################################################################    
    def lossfun_regular(self,x1_batch_tf,x2_batch_tf):
            
        lossf =  self.loss_f_fun(x1_batch_tf,x2_batch_tf,self.myControl)

###############################################################################   
        loss_add_1,loss_add_2 = self.loss_add_fun(x1_batch_tf,x2_batch_tf,self.myControl)
###############################################################################            
        loss_a1 = self.loss_assum1(self.myControl)
        loss_a2 = self.loss_assum2(self.myControl)
###############################################################################            
        loss =  self.weight_f*lossf + self.weight_a1*loss_a1+ self.weight_a2*loss_a2 + \
             self.add_1*loss_add_1 + self.add_2*loss_add_2

        return loss, lossf, loss_add_1, loss_add_2, loss_a1,loss_a2

###############################################################################  
    def loss_add_fun(self,x1_batch_tf,x2_batch_tf,myControl):
        ffun1,ffun2=self.PDE.yf_addloss(x1_batch_tf,x2_batch_tf,myControl)
        ffun_zeros = tf.zeros(x1_batch_tf.shape,dtype = tf.float32)
        loss_add_1 = self.lossfunNet(ffun1,ffun_zeros)
        
        loss_add_2= self.lossfunNet(ffun2,ffun_zeros)
        return loss_add_1,loss_add_2

###############################################################################                
    def loss_f_fun(self,x1_batch_tf,x2_batch_tf,myControl):
            
            with tf.GradientTape(persistent=True) as tape_o:
                tape_o.watch(x1_batch_tf)
                tape_o.watch(x2_batch_tf)
                loss_add_1,loss_add_2 =  self.PDE.yf_addloss(x1_batch_tf,x2_batch_tf,myControl)
                
            dg1dx1=tape_o.gradient(loss_add_1,x1_batch_tf)
            dg2dx2=tape_o.gradient(loss_add_2,x2_batch_tf)

            del tape_o

            ffun=dg1dx1+dg2dx2
            ffun_zeros = tf.zeros(x1_batch_tf.shape,dtype = tf.float32)
            lossf = self.lossfunNet(ffun,ffun_zeros)

            return lossf

##############################################################################
    def loss_assum1(self,myControl):
        
        control = myControl(self.x_mat_stationary)
        u1_arr = tf.reshape(control[:,0],(-1,1))
        u2_arr = tf.reshape(control[:,3],(-1,1))
        
        f1_arr,f2_arr = self.PDE.drift_coef(self.x1_arr_stationary,self.x2_arr_stationary)
        g11,g12,g21,g22 = self.PDE.diffusion_coef(self.x1_arr_stationary,self.x2_arr_stationary)
        
        part11=tf.reshape((1.0+tf.math.reduce_sum(self.x_mat_stationary**2,1)),(-1,1))
        
        part121 = 2.0*(self.x1_arr_stationary*(f1_arr+u1_arr)+self.x2_arr_stationary*(f2_arr+u2_arr))
        part122 = g11**2 + g12**2 + g21**2 + g22**2
        
        part12 = part121+part122
        
        part1 = part11 * part12
        
        part2 = -(2.0-self.parameters.stationary_par["alpha"])*((g11*self.x1_arr_stationary+g21*self.x2_arr_stationary)**2
                                                               +(g12*self.x1_arr_stationary+g22*self.x2_arr_stationary)**2)
        part3 = self.parameters.stationary_par["Lambda_big"]*part11**2
        
        part = part1+part2+part3
        
        loss_a1 = tf.reduce_max(tf.concat([part[:,0],[0.0]],axis=0))

        return loss_a1
##############################################################################
    def loss_assum2(self,myControl):
        
        f1_x_arr,f2_x_arr = self.PDE.drift_coef(self.x1_mat_assum2,self.x2_mat_assum2)
        gx11,gx12,gx21,gx22 = self.PDE.diffusion_coef(self.x1_mat_assum2,self.x2_mat_assum2)
        controlx = myControl(self.x_mat_assum2)
        u1_x_arr = tf.reshape(controlx[:,0],(-1,1))
        u2_x_arr = tf.reshape(controlx[:,3],(-1,1))
        
        f1_y_arr,f2_y_arr = self.PDE.drift_coef(self.y1_mat_assum2,self.y2_mat_assum2)
        gy11,gy12,gy21,gy22 = self.PDE.diffusion_coef(self.y1_mat_assum2,self.y2_mat_assum2)
        controly = myControl(self.y_mat_assum2)
        u1_y_arr = tf.reshape(controly[:,0],(-1,1))
        u2_y_arr = tf.reshape(controly[:,3],(-1,1))
        
        part11 = tf.reshape(tf.reduce_sum((self.x_mat_assum2-self.y_mat_assum2)**2,1),(-1,1))
        
        part121 = 2.0*((self.x1_mat_assum2-self.y1_mat_assum2)*(f1_x_arr+u1_x_arr-f1_y_arr-u1_y_arr)
                       +(self.x2_mat_assum2-self.y2_mat_assum2)*(f2_x_arr+u2_x_arr-f2_y_arr-u2_y_arr))
        part122 = (gx11-gy11)**2 + (gx12-gy12)**2+(gx21-gy21)**2+(gx22-gy22)**2
        
        part12 = part121 + part122
        
        part1 = part11 * part12
        
        part21 = ((self.x1_mat_assum2-self.y1_mat_assum2)*(gx11-gy11)+(self.x2_mat_assum2-self.y2_mat_assum2)*(gx21-gy21))**2
        part22 = ((self.x1_mat_assum2-self.y1_mat_assum2)*(gx12-gy12)+(self.x2_mat_assum2-self.y2_mat_assum2)*(gx22-gy22))**2
        part2 = -(2.0-self.parameters.stationary_par["beta"])*(part21+part22)
        
        part3 = self.parameters.stationary_par["gamma"]*part11**2
        
        part = part1 + part2 +part3
        
        loss_a2 = tf.reduce_max(tf.concat([part[:,0],[0.0]],axis=0))
        return loss_a2
   
