# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 10:55:40 2012

@author: rohan
"""

import dana ,numpy


n=60
dt=0.1
alpha=10.0
tau=1.0
h=0.0

I=dana.zeros((n,n))
U=dana.zeros((n,n))

U.connect(I.V,numpy.ones((1,1,)),'I',sparse=True)

K = 1.25*dana.gaussian((2*n+1,2*n+1),0.1)- 0.7*dana.gaussian((2*n+1,2*n+1),1)

U.connect(U.V,K,'L',shared=True)

U.dV = U.dV = '-V + maximum(V+dt/tau*(-V+(L/(N*N)*10*10+I+h)/alpha),0)'

I.V = dana.gaussian((N,N), 0.2, ( 0.5, 0.5))
I.V += dana.gaussian((N,N), 0.2, (-0.5,-0.5))
I.V += (2*numpy.random.random((N,N))-1)*.05

for i in range(250):
    focus.compute(dt)


dana.pylab.view([input.V,focus.V]).show()