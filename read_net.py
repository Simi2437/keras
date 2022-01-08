import cv2
import cv2.dnn

net= cv2.dnn.readNetFromTensorflow('/home/stefan/Downloads/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8/saved_model.pb',"/home/stefan/Downloads/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8/pipeline.config")
#net=cv2.dnn.re