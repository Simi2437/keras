import os.path
import numpy as np
import cv2
import glob
dst_img_width=480
dst_img_height=304

image_list = []
    #compression_para= cv2.IMWRITE_PNG_COMPRESSION[0]
for filename in glob.glob('/home/stefan/Downloads/seg_mask_keras/Images/*.bmp'):
#for filename in glob.glob('/home/stefan/Downloads/seg_mask_keras/Category_ids_greyscale/*.png'):
    im=cv2.imread(filename)
    #im=im+10
    #cv2.imshow("src",im)

    print( im.shape)
    img_dst=np.zeros(shape=[dst_img_height,dst_img_width,3],dtype=np.uint8)
    #img=cv2.createMat(im)
    dim=(dst_img_width,dst_img_height)

    #cv2.cvtColor(im,cv2.COLOR_BGR2GRAY,img_dst)
    #cv2.threshold(img_dst,98,1,cv2.THRESH_BINARY,img_dst)
    #img_dst=img_dst+1
    cv2.resize(im,dim,img_dst,interpolation=cv2.INTER_AREA)
    image_list.append(im)
    basename= os.path.basename(filename)
    newpath="/home/stefan/Downloads/seg_mask_keras/Images_png_resized/" + os.path.splitext(basename)[0]+".png"
    #newpath="/home/stefan/Downloads/seg_mask_keras/Category_ids_greyscale_resized/" + os.path.splitext(basename)[0]+".png"
    #cv2.imshow("josef",img_dst)
    #cv2.waitKey(10000)
    cv2.imwrite(newpath,img_dst,[int(cv2.IMWRITE_PNG_COMPRESSION),9])
