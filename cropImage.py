import os
import cv2 as cv
from tqdm import tqdm


total_images = 900

hr_list= os.listdir("dataset/HR")[:total_images]
lr_list= os.listdir("dataset/LR")[:total_images]

for img in tqdm(hr_list):
    im= cv.imread("dataset/HR/"+img)
    h,w,c= im.shape
    cropped= im[(h//2-384//2):(h//2+384//2), (w//2-384//2):(w//2+384//2)]
    cv.imwrite("dataset/hrc/"+img, cropped)
    
'''for img in lr_list:
    im= cv.imread("dataset/LR/"+img)
    h,w,c= im.shape
    cropped= im[(h//2-96//2):(h//2+96//2), (w//2-96//2):(w//2+96//2)]
    cv.imwrite("dataset/LR_cropped/"+img, cropped)
'''

key= cv.waitKey(0)

if key=='ESC':
    cv.destroyAllWindows()