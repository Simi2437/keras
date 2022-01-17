import os.path
import numpy as np
import cv2
import glob

dict1="/home/stefan/Downloads/seg_mask_keras/Images_png_resized/"
dict2="/home/stefan/Downloads/seg_mask_keras/Category_ids_greyscale_resized/"

image_list = []
    #compression_para= cv2.IMWRITE_PNG_COMPRESSION[0]
#for filename in glob.glob('/home/stefan/Downloads/seg_mask_keras/Images/*.bmp'):
for filename in glob.glob(dict1 + "*.png"):
    basename = os.path.basename(filename)
    newpath = dict2 + os.path.splitext(basename)[
        0] + ".png"
    try:
        f= open(newpath)
    except IOError:
        print("file missing"+filename)

for filename in glob.glob(dict2+ '*.png'):
    basename = os.path.basename(filename)
    newpath = dict1 + os.path.splitext(basename)[
        0] + ".png"
    try:
        f= open(newpath)
    except IOError:
        print("file missing"+filename)