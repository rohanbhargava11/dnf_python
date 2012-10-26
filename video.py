# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 11:55:05 2012

@author: rohan
"""

import cv2
#win=cv2.namedWindow("video0")
cap=cv2.VideoCapture(0)
#im=cap.read()
while True:
  ret,im=cap.read()
  print im
  #print im.shape()
  im=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
  #input_image=cv2.imread('/home/rohan/Documents/dnf_python/test1.jpg',0)
  #image=cv2.resize(gray,(101,101))
  cv2.imshow('video test',im)
  cv2.waitKey(10)
  