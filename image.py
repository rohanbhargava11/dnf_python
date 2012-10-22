import cv2

input_image=cv2.imread('/home/rohan/Documents/dnf_python/test1.jpg')
input_image=cv2.cvtColor(input_image,cv2.COLOR_RGB2GRAY)
input_image=cv2.resize(input_image,(101,101))
while True:
        
    cv2.imshow('hello',input_image)
    cv2.waitKey(10)