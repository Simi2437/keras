import os.path
import numpy as np
import cv2
import glob


image_list = []
for filename in glob.glob('/home/stefan/Downloads/seg_mask_keras/Category_ids/*.png'):
    im=cv2.imread(filename)
    print( im.shape)
    img_dst=np.zeros(shape=[1200,1920,1],dtype=np.uint8)
    #img=cv2.createMat(im)

    cv2.cvtColor(im,cv2.COLOR_BGR2GRAY,img_dst)
    cv2.threshold(img_dst,98,1,cv2.THRESH_BINARY,img_dst)
    img_dst=img_dst+1

    image_list.append(im)
    basename= os.path.basename(filename)
    newpath="/home/stefan/Downloads/seg_mask_keras/Category_ids_greyscale/" + os.path.basename(filename)
    #cv2.imshow("josef",img_dst)
    #cv2.waitKey(1000)
    cv2.imwrite(newpath,img_dst)
