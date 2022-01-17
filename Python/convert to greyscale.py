import os.path
import numpy as np
import cv2
import glob

dst_img_width=480
dst_img_height=304

image_list = []
for filename in glob.glob('/home/stefan/Downloads/seg_mask_keras/aug/mask/*.png'):
    im=cv2.imread(filename)
    print( im.shape)
    img_dst=np.zeros(shape=[dst_img_height,dst_img_width,1],dtype=np.uint8)
    #img=cv2.createMat(im)

    #cv2.cvtColor(im,cv2.COLOR_BGR2GRAY,img_dst)
    #cv2.threshold(img_dst,98,1,cv2.THRESH_BINARY,img_dst)
    white=np.where((im[:,:,0]==255))
    black=np.where((im[:,:,0])<255)
    im[white]=(2,2,2)
    im[black]=(1,1,1)
    img_dst=im
    #img_dst=img_dst+1

    image_list.append(im)
    basename= os.path.basename(filename)
    newpath="/home/stefan/Downloads/seg_mask_keras/aug/mask_rescaled/" + os.path.basename(filename)
    #cv2.imshow("josef",img_dst)
    #cv2.waitKey(1000)
    cv2.imwrite(newpath,img_dst)
