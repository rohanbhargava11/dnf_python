# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 15:17:28 2012

@author: rohan
"""

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
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import scipy.signal as sp




    
nn=101
dx=2*math.pi/nn
dy=2*math.pi/nn
sig=2*math.pi/36
sig1=2*math.pi/32 #mexican hat
C=0.09
pat=np.zeros((nn,nn))
h=0.0 # Just now it is set to  0.0 later I will give its some input
tau_inv=0.1


X,Y=np.mgrid[-nn/2:nn/2,-nn/2:nn/2]
def weights(sig):
    f=np.zeros((nn,nn))    
    f=-((dx*X)**2+(dy*X)**2)
    f=f/2*sig**2
    f=np.exp(f)
    f=f*(1./(np.sqrt(2*math.pi)*sig))
    return f
    
    #sig=sig**2
    #f=f*dx
    #f=-(f**2)
    #f=f/(2*(sig**2))
    #f=sp.norm.pdf(f,0,sig)  
    #f=f*dx
    #f=1./(np.sqrt(2*math.pi)*sig)*np.exp(-(f**2/(2*sig**2)))    
    #print f.shape
#f=np.exp(f)
    #f=np.dot(f,f.transpose())
    #f=np.dot(f,f)
    #return f

w0=1./(np.sqrt(2*math.pi)*sig)*(np.exp((-(((dx*X)**2)+((dy*Y)**2)))/(2*sig**2)))

w1=1./(np.sqrt(2*math.pi)*sig1)*((np.exp((-(((dx*X)**2)+((dy*Y)**2)))/(2*sig1**2))))

#w0=weights(sig)

#w1=weights(sig*3)
w=50*((w0-w1))


def plot(figno,u_history):
    fig = plt.figure(figno)
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(np.arange(nn),np.arange(nn))

    surf = ax.plot_surface(X, Y,u_history,cmap=cm.jet,linewidth=0, antialiased=True)
    ax.w_zaxis.set_major_locator(LinearLocator(10))
    ax.w_zaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    
    ax.set_zlabel("Excitation")
    ax.set_xlabel("Node X")
    ax.set_ylabel("Node Y")
    fig.colorbar(surf, shrink=0.5, aspect=5) 
    

def update(u,I):
    
    r=1/(1+np.exp(-u))

    #print u.shape
    
    t=sp.convolve2d(r,w,'same','wrap')
    #t=np.dot(r,w)  
    sum1=t*dx
   
    u=u+tau_inv*((-u+sum1+I))
        
        
    return u


def gauss_pbc(locx,locy,sig):
        z=np.zeros((nn,nn))
        for i in range(nn):
            for j in range(nn):
                d=min([abs(i*dx-locx) , 2*math.pi-abs(i*dx-locx)]) 
                d2=min([abs(j*dy-locy), 2*math.pi-abs(j*dy-locy)]) 
                z[j][i]=1./(np.sqrt(2*math.pi)*sig)*np.exp(-(d**2/(2*sig**2)+(d2**2/(2*sig**2))))
        return z    
#Update With Localised Input

'''
I_ext=np.zeros((nn,nn))
for k in range(int(nn/2-np.floor(nn/20)),int(nn/2+np.floor(nn/20))+1):
    I_ext[k]=1
'''
I_ext=np.zeros((nn,nn))
#I_ext[int(nn/2-np.floor(nn/20)):int(nn/2+np.floor(nn/20))+1,
 #    int(nn/2-np.floor(nn/20)):int(nn/2+np.floor(nn/20))+1] = 1
I_ext[47:52,47:52]=1
#I_ext=gauss_pbc(3*math.pi/2,3*math.pi/2,sig)


#!print u.shape

time=20
u_history=np.zeros((nn,nn))

u=np.zeros((nn,nn))
for k in range(time):
    
    u=update(u,I_ext)
    r=1/(1+np.exp(-u))
    u_history=r
    
plot(1,u_history)

u_history=np.zeros((nn,nn))
I_ext=np.zeros((nn,nn))
time=20
for k in range(time):
    u=update(u,I_ext)
    r=1/(1+np.exp(-u))
    u_history=r
    
plot(2,u_history)

plt.show()
