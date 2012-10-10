# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 10:21:25 2012

@author: rohan
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 19:15:40 2012

@author: Rohan
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as sp



    
    
nn=1001
dx=2*math.pi/nn
sig=2*math.pi/36
sig1=2*math.pi/22 #mexican hat
C=0.09
pat=np.zeros((nn,nn))
h=0.0 # Just now it is set to  0.0 later I will give its some input
tau_inv=0.1


def weights(sig):
    f=np.zeros((nn,))    
    f=((nn-1)/2-np.arange(nn))
   
    f=f*dx
    f=1./(np.sqrt(2*math.pi)*sig)*np.exp(-(f**2/(2*sig**2)))    

    return f



def plot(figno,time):
    fig = plt.figure(figno)
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(np.arange(nn),np.arange(time))

    surf = ax.plot_surface(X, Y,u_history)
    ax.w_zaxis.set_major_locator(LinearLocator(10))
    ax.w_zaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_zlabel("Excitation")
    ax.set_xlabel("Node")
    ax.set_ylabel("Time")
  
    

def update(u,I):
    
    r=1/(1+np.exp(-u))

   
    
    t=np.convolve(np.append(np.append(r[nn/2+1:],r),r[:-nn/2]),w, 'valid')
     
    sum1=t*dx
   
    u=u+tau_inv*((-u+sum1+I+h))
        
        
    return u


def gauss_pbc(locx,sig):
        w=np.zeros((nn,))
        for i in range(nn):
                d=min([abs(i*dx-locx) , 2*math.pi-abs(i*dx-locx)])
               
                w[i]=1./(np.sqrt(2*math.pi)*sig)*np.exp(-(d**2/(2*sig**2)))
                
        return w

#Update With Localised Input

I_ext=np.zeros((nn,))

for k in range(450,550):
    I_ext[k]=1

for k in range(250,350):
    I_ext[k]=1
t0=weights(sig)
t1=weights(sig*1.44)
w=50*((t0-t1)-C)

time=50
u_history=np.zeros((time,nn))

u=np.zeros((nn,))
for k in range(time):
    u=update(u,I_ext)
    r=1/(1+np.exp(-u))
    u_history[k]=r
plot(1,time)


time=50
u_history=np.zeros((time,nn))

I_ext=np.zeros((nn,))
for k in range(time):
    u=update(u,I_ext)
    r=1/(1+np.exp(-u))
    u_history[k]=r
plot(2,time)




'''
time=100
u_history=np.zeros((time,nn))
I_ext=np.zeros((nn,))
for k in range(250,350):
    I_ext[k]=1
for k in range(time):
    u=update(u,I_ext)
    r=1/(1+np.exp(-u))
    u_history[k]=r
plot(3,time)


time=150
u_history=np.zeros((time,nn))

I_ext=np.zeros((nn,))
for k in range(time):
    u=update(u,I_ext)
    r=1/(1+np.exp(-u))
    u_history[k]=r
plot(4,time)
'''
plt.show()


