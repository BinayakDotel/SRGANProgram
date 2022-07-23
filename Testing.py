from pickle import TRUE
from keras.models import load_model
from numpy.random import randint
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os

'''images_count= 10
hr_path= "dataset/HR"
lr_path= "dataset/LR"

hr_list= os.listdir(hr_path)[:images_count]

for img in hr_list:
    if img.lower().endswith((".jpg", ".jpeg", ".png")):
        print(img)'''


'''imgHR= cv.imread(f"{path}/0005.png")

h,w,c= imgHR.shape

width = int(w/4)
height = int(h/4)
dim = (width, height)
      
# resize image
imgLR = cv.resize(imgHR, dim, interpolation = cv.INTER_AREA)

cv.imwrite(f"{path}/0005LR.png", imgLR)
cv.imshow("imageHR", imgHR)
cv.imshow("imageLR", imgLR)

key=cv.waitKey(0)
if(key == 'ESC'):
    cv.destroyAllWindows()'''

enhancer= load_model('models/gen_e_100.h5')

imgLR= cv.imread(f"dataset/HR/0010.png")

width = int(96)
height = int(96)
dim = (width, height)
print(dim)
      
# resize image
imgLR = cv.resize(imgLR, dim, interpolation = cv.INTER_AREA)
imgLR = cv.cvtColor(imgLR, cv.COLOR_BGR2RGB)
imgLR = imgLR/255.
imgLR = np.expand_dims(imgLR, axis=0)
print(imgLR.shape)

gen_im= enhancer.predict(imgLR)

print(gen_im.shape)

cv.imshow("gen", gen_im[0,:,:,:])
cv.imshow("lr", imgLR[0,:,:,:])

key=cv.waitKey(0)
if key == 'ESC':
    cv.destroyAllWindows()