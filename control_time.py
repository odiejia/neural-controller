# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 14:50:19 2023

@author: dell
"""

"""
MC simulation
"""
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' 


###系统参数
alpha=-0.5
beta=0.1
omega=1.4
D1=0.01   
D2=0.01 #噪声1
S1=0.001
S2=0.001   #噪声2
K1=0.5
K2=0.3
K3=0.7
K4=1.0
A=1.0
B=1.5

###坐标
x_l = 0.5
x_u = 0.7
y_l = 0.9
y_u = 1.1
step_pdf = 0.01


###时间
t_ini = 0          # tmin
t_end = 100.0       # tmax
t_h = 0.01         # 步进长度
t = np.linspace(t_ini, t_end, int((t_end-t_ini)/t_h+1))

x = t.copy()
y = t.copy()
u1 = t.copy()
u2 = t.copy()

###初值
x[0] = 0.6
y[0] = 1.0

rate_x = 0.0
rate_y = 0.0
rate_xy = 0.0


def gwnoise(intensity,step,noise_num):
    normal = np.random.randn(noise_num)
    return np.sqrt(2*intensity/step)*normal


def fun1(q,p,gnoise1,gnoise2,u1,u2):
    return K1*A - K2*B*q + K3*q**2*p - K4*q + u1 + gnoise1+p*gnoise2

def fun2(q,p,gnoise3,gnoise4,u1,u2):
    return K2*B*q - K3*q**2*p + u2+ gnoise3+q*gnoise4


myControl1=tf.saved_model.load("savedmodel/myControl_1")    
    
myControl2=tf.saved_model.load("savedmodel/myControl_2")   
for i in range(t.shape[0]-1):
    step = t[i+1] - t[i]
    
    gnoise1 = gwnoise(2*D1,step,t.shape[0])
    gnoise2 = gwnoise(2*S1,step,t.shape[0])
    gnoise3 = gwnoise(2*D2,step,t.shape[0])
    gnoise4 = gwnoise(2*S2,step,t.shape[0])    

    #c_f_pred=myControl.predict(tf.concat([x[i],y[i]],1))   
    x1 = tf.constant(x[i].reshape(-1,1))
    y1 = tf.constant(y[i].reshape(-1,1))
    input_control1 = tf.cast(tf.concat([x1,y1],1),dtype=tf.float32)
    u1[i]=myControl1.call(input_control1)
    input_control2 = tf.cast(tf.concat([x1,y1],1),dtype=tf.float32)
    u2[i]=myControl2.call(input_control2)    
    k1_x = fun1(x[i], y[i], gnoise1[i], gnoise2[i], u1[i], u2[i])
    k1_y = fun2(x[i], y[i], gnoise3[i], gnoise4[i], u1[i], u2[i])
    
    temp1 = x[i] + 0.5*step*k1_x
    temp2 = y[i] + 0.5*step*k1_y
    
    k2_x = fun1(temp1, temp2,(gnoise1[i]+gnoise1[i+1])/2.0, (gnoise2[i]+gnoise2[i+1])/2.0, u1[i], u2[i])
    k2_y = fun2(temp1, temp2,(gnoise3[i]+gnoise3[i+1])/2.0, (gnoise4[i]+gnoise4[i+1])/2.0, u1[i], u2[i])
    
    temp1 = x[i] + 0.5*step*k2_x
    temp2 = y[i] + 0.5*step*k2_y
    
    k3_x = fun1(temp1, temp2,(gnoise1[i]+gnoise1[i+1])/2.0, (gnoise2[i]+gnoise2[i+1])/2.0, u1[i], u2[i])
    k3_y = fun2(temp1, temp2,(gnoise3[i]+gnoise3[i+1])/2.0, (gnoise4[i]+gnoise4[i+1])/2.0, u1[i], u2[i])
    
    temp1 = x[i] + step*k3_x
    temp2 = y[i] + step*k3_y
           
    k4_x = fun1(temp1, temp2,(gnoise1[i]+gnoise1[i+1])/2.0, (gnoise2[i]+gnoise2[i+1])/2.0, u1[i], u2[i])
    k4_y = fun2(temp1, temp2,(gnoise3[i]+gnoise3[i+1])/2.0, (gnoise4[i]+gnoise4[i+1])/2.0, u1[i], u2[i])
    
    x[i+1] = x[i] + step*(k1_x + 2*k2_x + 2*k3_x + k4_x)/6
    y[i+1] = y[i] + step*(k1_y + 2*k2_y + 2*k3_y + k4_y)/6  
    
    #x_value = tf.constant(x[i+1].reshape(-1,1))
    #y_value = tf.constant(y[i+1].reshape(-1,1))    
    #state = tf.cast(tf.concat([x_value,y_value],1),dtype=tf.float32) 
    #u[i+1] = myControl.call(state)
    #x_index = np.ceil((x[i] - x_l - 0.5*step_pdf)/step_pdf)
    #y_index = np.ceil((y[i] - y_l - 0.5*step_pdf)/step_pdf)

    #rate_x[x_index,i]= rate_x[x_index,i] + 1.0
     
#def control(q,p):
#    u1 = (-omega**2+1)*q
#    u2 = (2*D - alpha - 6*S*omega**2*q**4)*p + (beta - 6*D*omega**2 + 2*S)*q**2*p - (6*D+6*S*q**2)*p**3
#    u = u1+u2 
#    return u     


#u = control(x,y)    

saveControldata = 'control_time_hmc1.mat'
scio.savemat(saveControldata, {'t':t,
                           'control_time_hmc1':u1})
saveControldata = 'control_time_hmc2.mat'
scio.savemat(saveControldata, {'t':t,
                           'control_time_hmc2':u2})
###统计概率密度


plt.figure(dpi=600,figsize=(10,8)) 
#font = {'family' : 'Times New Roman',
#         'weight' : 'normal',
#         'size' : 20,
#         }

 
#plt.subplot(1,3,1)
#plt.plot(t,x)
#plt.xlabel('t',font)
#plt.ylabel('x',font)
#plt.xticks(fontsize=20)
#plt.yticks(fontsize=20) 
#plt.xlim(0,50)
#plt.ylim(-1.5,1.5)

#plt.subplot(1,3,2)
#plt.plot(t,y)
#plt.xlabel('t',font)
#plt.ylabel('y',font)
#plt.xticks(fontsize=20)
#plt.yticks(fontsize=20) 
#plt.xlim(0,50)
#plt.ylim(-1.5,1.5)

#plt.subplot(1,3,3)
plt.plot(t[1:t.shape[0]-1],u1[1:u1.shape[0]-1])
plt.xlabel('t',fontsize=30,fontproperties="Times New Roman")
plt.ylabel('u1',fontsize=30,fontproperties="Times New Roman")
plt.plot(t[1:t.shape[0]-1],u2[1:u2.shape[0]-1])
plt.xlabel('t',fontsize=30,fontproperties="Times New Roman")
plt.ylabel('u2',fontsize=30,fontproperties="Times New Roman")
plt.xticks(fontsize=25,fontproperties="Times New Roman")
plt.yticks(fontsize=25,fontproperties="Times New Roman") 
plt.xlim(0,50)
#plt.ylim(-1.5,1.5)

plt.savefig('./control_time.png')
