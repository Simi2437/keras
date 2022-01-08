import os
import random
import glob

from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img
import PIL
from PIL import ImageOps

from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import layers

from PyQt5.QtCore import Qt, pyqtSignal, QObject, QThread, QSize
from PyQt5.QtWidgets import QLineEdit, QApplication, QPushButton, QLabel, QWidget, QRadioButton, QProgressBar, \
    QTextEdit, QApplication, QFileDialog
from fbs_runtime.application_context.PyQt5 import ApplicationContext
from PyQt5.QtGui import QMovie, QPixmap, QFont
import cv2


class OxfordPets(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths

    def __len__(self):
        return len(self.input_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        img = load_img(self.input_img_paths, target_size=self.img_size)
        x[0] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        return x, y


class window(QWidget):
    def resizeImages(self, img_path):
        dst_img_width = 480
        dst_img_height = 304

        im = cv2.imread(img_path)
        # im=im+10
        # cv2.imshow("src",im)

        print(im.shape)
        img_dst = np.zeros(shape=[dst_img_height, dst_img_width, 3], dtype=np.uint8)
        # img=cv2.createMat(im)
        dim = (dst_img_width, dst_img_height)

        # cv2.cvtColor(im,cv2.COLOR_BGR2GRAY,img_dst)
        # cv2.threshold(img_dst,98,1,cv2.THRESH_BINARY,img_dst)
        # img_dst=img_dst+1
        cv2.resize(im, dim, img_dst, interpolation=cv2.INTER_AREA)
        # newpath="/home/stefan/Downloads/seg_mask_keras/Category_ids_greyscale_resized/" + os.path.splitext(basename)[0]+".png"
        # cv2.imshow("josef",img_dst)
        # cv2.waitKey(10000)
        cv2.imwrite(self.newpath, img_dst, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

    def clickedInput(self):
        # self.input_dir = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.input_ctrl.setText(self.input_dir)
        self.input_img_paths = sorted(
            [
                os.path.join(self.input_dir, fname)
                for fname in os.listdir(self.input_dir)
                if fname.endswith(".bmp")
            ]
        )
        self.clickedSetconfig()
        print('file: ', self.input_dir)

    # def clickedTarget(self):
    #   self.target_dir = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
    #   self.target_ctrl.setText(self.target_dir)
    #   self.target_img_paths = sorted(
    #     [
    #       os.path.join(self.target_dir, fname)
    #       for fname in os.listdir(self.target_dir)
    #       if fname.endswith(".png") and not fname.startswith(".")
    #     ]
    #   )

    def clickedLoadModel(self):
        # Loads the weights
        # self.checkpoint_path = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.model_ctrl.setText(self.checkpoint_path)
        self.model = keras.models.load_model(self.checkpoint_path)

    def clickedSetconfig(self):
        # Build model
        self.model = self.get_model(self.img_size, self.num_classes)
        self.model.summary()
        self.index = 0
        # Split our img paths into a training and a validation set
        # val_samples = 20
        # random.Random(130).shuffle(self.input_img_paths)
        # random.Random(130).shuffle(self.target_img_paths)
        # self.train_input_img_paths = self.input_img_paths[:-val_samples]
        # self.train_target_img_paths = self.target_img_paths[:-val_samples]
        # self.val_input_img_paths = self.input_img_paths[-val_samples:]
        # self.val_target_img_paths = self.target_img_paths[-val_samples:]

        # Instantiate data Sequences for each split
        # self.train_gen = OxfordPets(
        #     self.batch_size, self.img_size, self.train_input_img_paths, self.train_target_img_paths
        # )
        # self.val_gen = OxfordPets(self.batch_size, self.img_size, self.val_input_img_paths, self.val_target_img_paths)

        # Configure the model for training.
        # We use the "sparse" version of categorical_crossentropy
        # because our target data is integers.
        self.model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")
        self.checkpoint_path = "/home/stefan/Downloads/seg_mask_keras/saved_models/"
        self.callbacks = [
            keras.callbacks.ModelCheckpoint(self.checkpoint_path, save_best_only=True)
        ]

    def __init__(self, parent=None):
        super(window, self).__init__(parent)
        self.setFixedWidth(700)
        self.setFixedHeight(508)

        self.newpath = "/home/stefan/Downloads/seg_mask_keras/temp.png"

        # Input Dir
        self.input_label = QLabel(self)
        self.input_label.setText('Input Directory:')
        self.input_ctrl = QLineEdit(self)
        self.input_btn = QPushButton('...', self)
        self.input_btn.clicked.connect(self.clickedInput)

        self.input_btn.resize(32, 32)
        self.input_btn.move(350, 20)
        self.input_ctrl.move(130, 20)
        self.input_ctrl.resize(200, 32)
        self.input_label.move(20, 20)

        # Target Dir
        # self.target_label = QLabel(self)
        # self.target_label.setText('Target Directory:')
        # self.target_ctrl = QLineEdit(self)
        # self.target_btn = QPushButton('...', self)
        # self.target_btn.clicked.connect(self.clickedTarget)

        # self.target_btn.resize(32,32)
        # self.target_btn.move(350, 60)
        # self.target_ctrl.move(130, 60)
        # self.target_ctrl.resize(200, 32)
        # self.target_label.move(20, 60)

        # Model Dir
        self.model_label = QLabel(self)
        self.model_label.setText('Model Directory:')
        self.model_ctrl = QLineEdit(self)
        self.model_btn = QPushButton('...', self)
        self.model_btn.clicked.connect(self.clickedLoadModel)

        self.model_btn.resize(32, 32)
        self.model_btn.move(350, 300)
        self.model_ctrl.move(130, 300)
        self.model_ctrl.resize(200, 32)
        self.model_label.move(20, 300)

        self.input_dir = "/home/stefan/Downloads/seg_mask_keras/Images"
        self.target_dir = "/home/stefan/Downloads/seg_mask_keras/aug/mask_rescaled/"

        self.img_size = (304, 480)
        self.num_classes = 3
        self.batch_size = 12

        # self.setconfig_btn = QPushButton('Set', self)
        # self.setconfig_btn.clicked.connect(self.clickedSetconfig)

        # self.setconfig_btn.resize(32,32)
        # self.setconfig_btn.move(100, 100)

        # self.train_btn = QPushButton('Train', self)
        # self.train_btn.clicked.connect(self.train)

        # self.train_btn.resize(32,32)
        # self.train_btn.move(100, 250)

        self.predict_btn = QPushButton('Predict', self)
        self.predict_btn.clicked.connect(self.predict)

        self.predict_btn.resize(100, 32)
        self.predict_btn.move(150, 350)
        # print("Number of samples:", len(self.input_img_paths))

        # Free up RAM in case the model definition cells were run multiple times
        keras.backend.clear_session()
        self.clickedInput()
        self.clickedLoadModel()

    # def train(self):
    #   # Train the model, doing validation at the end of each epoch.
    #   epochs = 2
    #   self.model.fit(self.train_gen, epochs=epochs, validation_data=self.val_gen, callbacks=self.callbacks)

    def predict(self):
        # Generate predictions for all images in the validation set
        print(self.index, " : ", self.input_img_paths)
        self.resizeImages(self.input_img_paths[self.index])

        # self.val_target_img_paths = self.target_img_paths[-val_samples:]
        self.val_gen = OxfordPets(1, self.img_size, self.newpath)
        self.val_preds = self.model.predict(self.val_gen)
        self.display_mask(0)  # Note that the model only sees inputs at 150x150.
        self.index = self.index + 1

    def display_mask(self, i):
        """Quick utility to display a model's prediction."""
        mask = np.argmax(self.val_preds[i], axis=-1)
        mask = np.expand_dims(mask, axis=-1)
        mask_g = np.array(mask * 255, dtype=np.uint8)
        img1 = cv2.imread(self.newpath)
        masked = cv2.bitwise_and(img1, img1, mask=mask_g)
        hi = len(mask)
        wi = len(mask[0])
        for i in range(hi):
            for j in range(wi):
                if mask[i][j] == 0:
                    masked[i][j][0] = 255
        dst = cv2.addWeighted(img1, 1, masked, 0.3, 0)

        img = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(dst))
        print("img1: ", img1)
        display(img)
        img.show()

    def get_model(self, img_size, num_classes):
        inputs = keras.Input(shape=img_size + (3,))

        ### [First half of the network: downsampling inputs] ###

        # Entry block
        x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        previous_block_activation = x  # Set aside residual

        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for filters in [64, 128, 256]:
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
                previous_block_activation
            )
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        ### [Second half of the network: upsampling inputs] ###

        for filters in [256, 128, 64, 32]:
            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.UpSampling2D(2)(x)

            # Project residual
            residual = layers.UpSampling2D(2)(previous_block_activation)
            residual = layers.Conv2D(filters, 1, padding="same")(residual)
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        # Add a per-pixel classification layer
        outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

        # Define the model
        model = keras.Model(inputs, outputs)
        return model
    # def validation(self):
    #   # Display results for validation image #10
    #   i = 10

    #   # Display mask predicted by our model

    #   i = 9

    #   # Display mask predicted by our model
    #   self.display_mask(i)  # Note that the model only sees inputs at 150x150.

    #   i = 8

    #   # Display mask predicted by our model
    #   self.display_mask(i)  # Note that the model only sees inputs at 150x150.

    #   i = 7

    #   # Display input image
    #   display(Image(filename=self.val_input_img_paths[i]))


def main():
    app = QApplication([])
    app.setStyle("Fusion")
    ex = window()
    ex.show()
    exit(app.exec_())


if __name__ == '__main__':
    main()
# input_dir = "/home/stefan/Downloads/seg_mask_keras/Images_png_resized/"

# input_dir = "images/"
# target_dir = "annotations/trimaps/"
# target_dir = "/home/stefan/Downloads/seg_mask_keras/Category_ids_greyscale_resized/"


# for input_path, target_path in zip(input_img_paths[:2], target_img_paths[:2]):
#     print(input_path, "|", target_path)


# -----------------------------------------------------------------------------------------------------------------------


# from PIL import Image

# filename= input_img_paths[9]
# # Display input image #7
# display(Image(filename=input_img_paths[9]))
# # Display auto-contrast version of corresponding target (per-pixel categories)
# img = PIL.ImageOps.autocontrast(load_img(target_img_paths[9]))
# display(img)
# img.show()
# -----------------------------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------------------------------


"""
callbacks = [
    keras.callbacks.ModelCheckpoint("oxford_segmentation.h5", save_best_only=True)
]
"""

# model.load_weights(checkpoint_path)
# -----------------------------------------------------------------------------------------------------------------------









