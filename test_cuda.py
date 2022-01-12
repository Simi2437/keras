import cv2


#from cv2 import Tracker

#cv2.cuda.printCudaDeviceInfo(0)


#print(dir(cv2.cuda))

im= cv2.imread("/home/stefan/Downloads/seg_mask_keras/Images/0_150128.bmp")
cv2.imshow("1",im)
cv2.waitKey(1000)
GPU_im=cv2.cuda_GpuMat()
GPU_dst=cv2.cuda_GpuMat()
GPU_im.upload(im)
GPU_im.download(im)


cv2.imshow("1",im)
cv2.waitKey(1000)
