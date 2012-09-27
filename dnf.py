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




    
nn=101
dx=2*math.pi/nn
sig=2*math.pi/10
C=0.5
pat=np.zeros((nn,nn))
h=0.4
tau_inv=0.1

#Training Weight Matrix

w = 4 * (np.exp(-dx**2*((nn-1.)/2-np.arange(nn))**2/(2*pow(sig,2))) - C) 
'''
f=((nn-1)/2-np.arange(nn))
f=f*dx
f=-(f**2)
f=f/(2*(sig**2))
f=np.exp(f)
f=f-C
f=4*f
for loc in range(0,nn):
    i=np.arange(0,nn)
    #i=np.transpose(i)
    
    dis=np.minimum(abs(i-loc),nn-abs(i-loc))
    print np.exp(-np.power((dis*dx),2)/(2*np.power(sig,2)))
    pat[:,loc]= np.exp((-np.power(dis*dx,2))/(2*pow(sig,2)))

#w=np.dot(pat,np.transpose(pat)) # why do we do this?
#w=w/w[0][0]
w=4*(pat[0:100][50]-C)
'''


def plot(figno,time):
    fig = plt.figure(figno)
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(np.arange(nn),np.arange(time))

    surf = ax.plot_surface(X, Y,u_history)
    ax.w_zaxis.set_major_locator(LinearLocator(10))
    ax.w_zaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_zlabel("Excitation")
    ax.set_xlabel("Node")
    ax.set_ylabel("Time")

def update(u,I):
    
    r=1/(1+np.exp(-u))

    #print u.shape
    
    t=np.convolve(np.append(np.append(r[-1:nn/2:-1],r),r[:-nn/2]),w, 'valid')
    #t=np.dot(r,w)  
    sum1=t*dx
   
    u=u+tau_inv*((-u+sum1+I+h))
        
        
    return u

#Update With Localised Input

I_ext=np.zeros((nn,))
for k in range(int(nn/2-np.floor(nn/10)),int(nn/2+np.floor(nn/10))+1):
    I_ext[k]=2



#!print u.shape

time=50
u_history=np.zeros((time,nn))

u=np.zeros((nn,))
for k in range(time):
    u=update(u,I_ext)
    r=1/(1+np.exp(-u))
    u_history[k]=r
plot(1,time)

time=100
u_history=np.zeros((time,nn))

I_ext=np.zeros((nn,))
for k in range(time):
    u=update(u,I_ext)
    r=1/(1+np.exp(-u))
    u_history[k]=r
plot(2,time)




#I=np.ones((nn,))
#update(50,u,I)
#plot(2,50)
#r=1/1+exp(-I)
#for k in range(50):
#        #print u.shape
#        r=1/(1+np.exp(-u))
#        t=np.convolve(np.append(np.append(r[-1:nn/2:-1],r),r[:-nn/2]),w, 'valid')
#        #t=np.dot(r,w)  
#        sum1=t*dx
#   
#        u=u+dt*(-u+sum1+I)
#        r=1/(1+np.exp(-u))
        #u_history[k]=r
#u=update(50,u,I_ext)
#plt.show()

#I_ext=np.zeros((nn,))

  


