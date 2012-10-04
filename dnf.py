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

#Training Weight Matrix


#w0 =(np.exp(-(dx*((nn-1.)/2-np.arange(nn))**2)/(2*pow(sig1,2)))) 
#w0 =(np.exp(-(dx*((nn-1.)/2-np.arange(nn))**2)/(2*pow(sig,2))))  #mexican hat
#w=(w0-w1)

def weights(sig):
    f=np.zeros((nn,))    
    f=((nn-1)/2-np.arange(nn))
    #sig=sig**2
    #f=f*dx
    #f=-(f**2)
    #f=f/(2*(sig**2))
    #f=sp.norm.pdf(f,0,sig)  
    f=f*dx
    f=1./(np.sqrt(2*math.pi)*sig)*np.exp(-(f**2/(2*sig**2)))    
    #print f.shape
#f=np.exp(f)
    #f=np.dot(f,f.transpose())
    #f=np.dot(f,f)
    return f
    #f=f-C
    #f=aw*f
#w0=4*weights(sig)
#w0=np.dot(w0,w0.transpose())
#w1=3*weights(sig1)
#w=(w0-w1)

'''
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
    ax.w_zaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_zlabel("Excitation")
    ax.set_xlabel("Node")
    ax.set_ylabel("Time")
  
    

def update(u,I):
    
    r=1/(1+np.exp(-u))

    #print u.shape
    
    t=np.convolve(np.append(np.append(r[nn/2+1:],r),r[:-nn/2]),w, 'valid')
    #t=np.dot(r,w)  
    sum1=t*dx
   
    u=u+tau_inv*((-u+sum1+I+h))
        
        
    return u


def gauss_pbc(locx,sig):
        w=np.zeros((nn,))
        for i in range(nn):
                d=min([abs(i*dx-locx) , 2*math.pi-abs(i*dx-locx)])
                #d=d*dx
                w[i]=1./(np.sqrt(2*math.pi)*sig)*np.exp(-(d**2/(2*sig**2)))
                #w[i]=np.exp(-(d**2/(2*sig**2)))
        return w

#Update With Localised Input

I_ext=np.zeros((nn,))

for k in range(420,450):
    I_ext[k]=1

#for k in range(600,800):
 #   I_ext[k]=1

#for k in range(int(nn/2-np.floor(nn/10)),int(nn/2+np.floor(nn/10))+1):
 #  I_ext[k]=1


#!print u.shape
#I_ext=gauss_pbc(math.pi,sig/200)
#I_ext=np.reshape(I_ext,(nn,))

def hebbMulti():
        
        w=np.zeros((nn,nn))   
        for i in range(nn):
            r=gauss_pbc(i*dx,sig/3) - gauss_pbc(i*dx,sig)
            #plt.plot(r)
            w=w+np.dot(r,r.transpose())
            print w.shape
        return w/nn
#w=hebbMulti() 
#w=1000*(hebbMulti()-C)
#v0=gauss_pbc(50,sig/3)

#v1=gauss_pbc(50,sig)
#v=w[50]
#v2=1000*(((v0-v1)/nn)-C)
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


time=200
u_history=np.zeros((time,nn))

I_ext=np.zeros((nn,))
for k in range(time):
    u=update(u,I_ext)
    r=1/(1+np.exp(-u))
    u_history[k]=r
plot(2,time)





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

#plt.show()

'''
for k in range(140,200):
    I_ext[k]=1

time=25
u_history=np.zeros((time,nn))

u=np.zeros((nn,))
for k in range(time):
    u=update(u,I_ext)
    r=1/(1+np.exp(-u))
    u_history[k]=r
plot(5,time)
'''
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




