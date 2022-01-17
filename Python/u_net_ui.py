import os
import random

from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img
import PIL
from PIL import ImageOps

from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import layers
import pickle
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QThread, QSize
from PyQt5.QtWidgets import QLineEdit, QApplication, QPushButton, QLabel, QWidget, QRadioButton, QProgressBar, \
    QTextEdit, QApplication, QFileDialog
from fbs_runtime.application_context.PyQt5 import ApplicationContext
from PyQt5.QtGui import QMovie, QPixmap, QFont


class OxfordPets(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i: i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            y[j] -= 1
        print(y)
        return x, y


class window(QWidget):
    def clickedInput(self):
        self.input_dir = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.input_ctrl.setText(self.input_dir)
        self.input_img_paths = sorted(
            [
                os.path.join(self.input_dir, fname)
                for fname in os.listdir(self.input_dir)
                if fname.endswith(".png")
            ]
        )
        print('file: ', self.input_dir)

    def clickedTarget(self):
        self.target_dir = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.target_ctrl.setText(self.target_dir)
        self.target_img_paths = sorted(
            [
                os.path.join(self.target_dir, fname)
                for fname in os.listdir(self.target_dir)
                if fname.endswith(".png") and not fname.startswith(".")
            ]
        )

    def clickedLoadModel(self):
        # Loads the weights
        self.checkpoint_path = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.model_ctrl.setText(self.checkpoint_path)
        self.model = keras.models.load_model(self.checkpoint_path)
        # self.model = pickle.load(open(self.checkpoint_path+"/model.pkl", "rb"))

    def clickedSetconfig(self):
        # Build model
        self.model = self.get_model(self.img_size, self.num_classes)
        self.model.summary()

        # Split our img paths into a training and a validation set
        val_samples = 20
        random.Random(130).shuffle(self.input_img_paths)
        random.Random(130).shuffle(self.target_img_paths)
        self.train_input_img_paths = self.input_img_paths[:-val_samples]
        self.train_target_img_paths = self.target_img_paths[:-val_samples]
        self.val_input_img_paths = self.input_img_paths[-val_samples:]
        self.val_target_img_paths = self.target_img_paths[-val_samples:]

        # Instantiate data Sequences for each split
        self.train_gen = OxfordPets(
            self.batch_size, self.img_size, self.train_input_img_paths, self.train_target_img_paths
        )
        self.val_gen = OxfordPets(self.batch_size, self.img_size, self.val_input_img_paths, self.val_target_img_paths)

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
        self.target_label = QLabel(self)
        self.target_label.setText('Target Directory:')
        self.target_ctrl = QLineEdit(self)
        self.target_btn = QPushButton('...', self)
        self.target_btn.clicked.connect(self.clickedTarget)

        self.target_btn.resize(32, 32)
        self.target_btn.move(350, 60)
        self.target_ctrl.move(130, 60)
        self.target_ctrl.resize(200, 32)
        self.target_label.move(20, 60)

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

        self.input_dir = "/home/stefan/Downloads/seg_mask_keras/aug/img/"
        self.target_dir = "/home/stefan/Downloads/seg_mask_keras/aug/mask_rescaled/"

        self.img_size = (304, 480)
        self.num_classes = 3
        self.batch_size = 12

        self.setconfig_btn = QPushButton('Set', self)
        self.setconfig_btn.clicked.connect(self.clickedSetconfig)

        self.setconfig_btn.resize(32, 32)
        self.setconfig_btn.move(130, 100)

        self.train_btn = QPushButton('Train', self)
        self.train_btn.clicked.connect(self.train)

        self.train_btn.resize(100, 32)
        self.train_btn.move(130, 250)

        self.predict_btn = QPushButton('Predict', self)
        self.predict_btn.clicked.connect(self.predict)

        self.predict_btn.resize(100, 32)
        self.predict_btn.move(130, 350)
        # print("Number of samples:", len(self.input_img_paths))

        # Free up RAM in case the model definition cells were run multiple times
        keras.backend.clear_session()

    def train(self):
        # Train the model, doing validation at the end of each epoch.
        epochs = 15
        self.model.fit(self.train_gen, epochs=epochs, validation_data=self.val_gen, callbacks=self.callbacks)
        # pickle.dump(self.model, open( self.checkpoint_path+"model.pkl", "wb"))
        self.model.save(self.checkpoint_path)

    def predict(self):
        # Generate predictions for all images in the validation set
        val_gen = OxfordPets(self.batch_size, self.img_size, self.val_input_img_paths, self.val_target_img_paths)
        self.val_preds = self.model.predict(val_gen)
        self.validation()

    def display_mask(self, i):
        """Quick utility to display a model's prediction."""
        mask = np.argmax(self.val_preds[i], axis=-1)
        mask = np.expand_dims(mask, axis=-1)
        img = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))
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

    def validation(self):
        # Display results for validation image #10
        i = 10
        # Display input image
        display(Image(filename=self.val_input_img_paths[i]))

        # Display ground-truth target mask
        img = PIL.ImageOps.autocontrast(load_img(self.val_target_img_paths[i]))
        display(img)
        img.show()

        # Display mask predicted by our model
        self.display_mask(i)  # Note that the model only sees inputs at 150x150.

        i = 9

        # Display input image
        display(Image(filename=self.val_input_img_paths[i]))

        # Display ground-truth target mask
        img = PIL.ImageOps.autocontrast(load_img(self.val_target_img_paths[i]))
        display(img)
        img.show()

        # Display mask predicted by our model
        self.display_mask(i)  # Note that the model only sees inputs at 150x150.

        i = 8

        # Display input image
        display(Image(filename=self.val_input_img_paths[i]))

        # Display ground-truth target mask
        img = PIL.ImageOps.autocontrast(load_img(self.val_target_img_paths[i]))
        display(img)
        img.show()

        # Display mask predicted by our model
        self.display_mask(i)  # Note that the model only sees inputs at 150x150.

        i = 7

        # Display input image
        display(Image(filename=self.val_input_img_paths[i]))

        # Display ground-truth target mask
        img = PIL.ImageOps.autocontrast(load_img(self.val_target_img_paths[i]))
        display(img)
        img.show()

        # Display mask predicted by our model
        self.display_mask(i)  # Note that the model only sees inputs at 150x150.
        # -----------------------------------------------------------------------------------------------------------------------


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









