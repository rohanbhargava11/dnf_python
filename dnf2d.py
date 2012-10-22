

from __future__ import division
import numpy as np
import math
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import scipy.signal as sp
import cv2

from time import *
print 'Hello'
    
nn=101
dx=2*math.pi/nn
dy=2*math.pi/nn
sig=2*math.pi/15
sig1=sig*4#mexican hat
C=0.09
pat=np.zeros((nn,nn))
h=0.0 # Just now it is set to  0.0 later I will give its some input
tau_inv=0.1


X,Y=np.mgrid[-nn/2:nn/2,-nn/2:nn/2]
def weights(sig):
    f=np.zeros((nn,nn))    
    f=-(((dx*X)**2/2*sig**2)+((dy*Y)**2/2*sig**2))
    #f=f/2*sig**2
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
#w=4*(w-C)
w1=1./(np.sqrt(2*math.pi)*sig1)*(np.exp((-(((dx*X)**2)+((dy*Y)**2)))/(2*sig1**2)))

#w0=weights(sig)

#w1=weights(sig1)
w=100*((w0-w1)-C)


def plot(figno,u_history):
    fig = plt.figure(figno)
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(np.arange(nn),np.arange(nn))

    surf = ax.plot_surface(X, Y,u_history)#,cmap=cm.jet,
            #linewidth=0, antialiased=True)
    ax.w_zaxis.set_major_locator(LinearLocator(10))
    ax.w_zaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    
    ax.set_zlabel("Excitation")
    ax.set_xlabel("Node X")
    ax.set_ylabel("Node Y")
    #fig.colorbar(surf, shrink=0.5, aspect=5) 
    

def update(u,I):
    
    r=1/(1+np.exp(-u))

    #print u.shape
    
    t=sp.convolve2d(r,w,'same','wrap')
    #t=np.dot(r,w)  
    sum1=t*dx*dx
   
    u=u+tau_inv*((-u+sum1+50*I)+h)
        
        
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
#cap=cv2.VideoCapture(0)
while True:
    #ret,im=cap.read()    
    #im=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    input_image=cv2.imread('/home/rohan/Documents/dnf_python/test1.jpg',0)
    input_image=cv2.resize(input_image,(101,101))
    for i in range (101):
        for j in range (101):
            if input_image[i][j]>140:
                input_image[i][j]=0
            else:
                input_image[i][j]=1
    I_ext=input_image#/np.max(input_image)
    #cv2.imshow('test',input_image)
    #I_ext[int(nn/2-np.floor(nn/20)):int(nn/2+np.floor(nn/20))+1,
     
    #     int(nn/2-np.floor(nn/20)):int(nn/2+np.floor(nn/20))+1] = 1
    #I_ext[50:60,50:60]=1
    #I_ext=gauss_pbc(3*math.pi/2,3*math.pi/2,sig)
    
    
    #!print u.shape
    
    #I_ext[10:30,10:30] = 1
    
    time=20
    u_history=np.zeros((nn,nn))
    
    u=np.zeros((nn,nn))
    #for k in range(time):
        
    u=update(u,I_ext)
    r=1/(1+np.exp(-u))
    u_history=r
    print 'I am going to plot' 
    plt.hold(True)
    plot(1,u_history)
    plt.show()
    #sleep()
    #plt.close()
#plot(7,I_ext)
'''
u_history=np.zeros((nn,nn))
I_ext=np.zeros((nn,nn))
#I_ext[10:30,10:30] = 1
time=20
for k in range(time):
    u=update(u,I_ext)
    r=1/(1+np.exp(-u))
    u_history=r
    
#plot(2,u_history)

u_history=np.zeros((nn,nn))
I_ext=np.zeros((nn,nn))
I_ext[10:30,10:30] = 1
time=20
h=5.0 # changes the h to control the decay in the bubble without the input

# without this put the time as 8 and you can see the decay


for k in range(time):
    u=update(u,I_ext)
    r=1/(1+np.exp(-u))
    u_history=r
    
plot(3,u_history)
'''
'''
time =20
I_ext=np.zeros((nn,nn))
u_history=np.zeros((nn,nn))
for k in range(time):
    
    u=update(u,I_ext)
    r=1/(1+np.exp(-u))
    u_history=r
    
plot(4,u_history)


I_ext=np.zeros((nn,nn))
u_history=np.zeros((nn,nn))
for k in range(time):
    
    u=update(u,I_ext)
    r=1/(1+np.exp(-u))
    u_history=r
    
plot(4,u_history)

'''
'''
I_ext[10:20,10:20] = 1
time=20
for k in range(time):
    u=update(u,I_ext)
    r=1/(1+np.exp(-u))
    u_history=r
plot(4,u_history)
I_ext=np.zeros((nn,nn))
time=20
for k in range(time):
    u=update(u,I_ext)
    r=1/(1+np.exp(-u))
    u_history=r
plot(5,u_history)
'''
#plot(6,w)


plt.show()
