from Keras_preproc import *

from keras_preprocessing.image import ImageDataGenerator


data_gen_args= dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

#myGenerator = trainGenerator(20,'data/membrane/train','image','label',data_gen_args,save_to_dir = "data/membrane/train/aug")

myGenerator = trainGenerator(5,'/home/stefan/Downloads/seg_mask_keras','Images_png_resized', 'Category_ids_greyscale_resized', data_gen_args,
                             save_to_dir_img = "/home/stefan/Downloads/seg_mask_keras/aug/img",
                             save_to_dir_mask = "/home/stefan/Downloads/seg_mask_keras/aug/mask",
                             mask_save_prefix="1",
                             image_save_prefix="1",
                             image_color_mode="rgb"
                             )


#you will see 60 transformed images and their masks in data/membrane/train/aug
num_batch = 855
for i,batch in enumerate(myGenerator):
    if(i >= num_batch):
        break

#image_arr,mask_arr = geneTrainNpy("/home/stefan/Downloads/seg_mask_keras/aug/img","/home/stefan/Downloads/seg_mask_keras/aug/mask")
#np.save("data/image_arr.npy",image_arr)
#np.save("data/mask_arr.npy",mask_arr)