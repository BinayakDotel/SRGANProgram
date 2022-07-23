import cv2 as cv
import os

total_images= 900
path= "dataset/hrc/"
img_list= os.listdir(path)[:total_images]

scale=4

for img in img_list:
  if img.lower().endswith(('.png', '.jpg', '.jpeg')):
    hr_img= cv.imread("dataset/hrc/"+img)
    
    h,w,c= hr_img.shape
    #print(hr_img.shape)
    
    w=w//scale
    h=h//scale
    
    dim=(w,h)
    #print(dim)
    
    lr_img= cv.resize(hr_img, dim, interpolation=cv.INTER_AREA)

    cv.imwrite("dataset/lrc/"+img, lr_img)
    