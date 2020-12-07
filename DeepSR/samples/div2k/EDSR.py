import numpy as np
from keras import losses, regularizers
from keras.layers import Input, Add, Lambda
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.core import Reshape
from keras.layers import concatenate, Activation
from keras.optimizers import Adam
import tensorflow as tf
from keras.models import Model
from keras import backend as K
import tensorflow as tf

from subpixel import Subpixel

#  provide the settings for the model EDSR.
# These settings can be overriden by command scripts.
settings = \
    {
        'augment': [],  # can be any combination of [90,180,270, 'flipud', 'fliplr', 'flipudlr' ], or []
        'backend': 'tensorflow', # keras is going to use tensorflow framework in processing.
        'batchsize':16,  # number of batches
        'channels': 3,  # color channels to be used in training . Only one channel in this case
        'colormode': 'RGB',  # the color space is RGB. 'RGB' or 'YCbCr'
        'crop': 0,  # do not crop from borders of images.
        'crop_test': 0,  # do not crop from borders of images in tests.
        'decay': 0.9,  # learning rate decay.
        'earlystoppingpatience': 20,  # stop after 5 epochs if the performance of the model has not improved.
        'epoch': 20,  # train the model for total 10 passes on training data.
        'inputsize': 48,  # size of input image patches is 33x33.
        'lrate': 0.0001,  # learning rate
        'lrateplateaupatience': 10,  # number of epochs to wait before reducing the learning rate.
        'lrateplateaufactor': 0.5,  # the ratio of decrease in learning rate value.
        'minimumlrate': 1e-7,  # learning rate can be reduced down to a maximum of this value.
        'modelname': "EDSR",  # modelname is the same as the name of this file.
        'metrics': ["ERGAS", "PAMSE", "PSNR", "SAM", "SCC", "SSIM", "UQI", "VIF"], # modelname is the same as the name of this file.
        'normalization': ['standard', 53.28, 40.732],  # apply standardization to input images (mean, std)
        'outputdir': '',  # sub directories automatically created.
        'saveimages': False,
        'scale': 2,  # magnification factor is 4.
        'stride': 23,  # give a step of 11 pixels apart between patches while cropping them from images for training.
        'target_channels': 3,  # color channels to be used in tests . Only one channel in this case
        'target_cmode': 'RGB',  # 'YCbCr' or 'RGB'
        'testpath': [r"D:\calismalarim\datasets\DIV2K\sub_test"],  # path to the folder in which test images are. Can be more than one.
        'traindir': r"D:\calismalarim\datasets\DIV2K\sub_train",  # path to the folder in which training images are.
        'upscaleimage': False,  # Input image to be upscaled by given interpolation method before handling it over the model
        'valdir': r'',  # path to the folder in which validation images are.
        'workingdir': '',  #
        'weightpath': ''  # path to model weights either for training to start with, or for test.
}

# EDSR is defined within this method.
# It is written with KERAS API
def build_model(self, testmode=False):


    def residual_block(inp, n_filters, residual_scaling=None, regularize= True):

        if regularize:
            temp = Conv2D(n_filters, (3, 3), padding='same', activation='relu',
                          activity_regularizer=regularizers.l1(1e-9))(inp)

            temp = Conv2D(n_filters, (3, 3), padding='same',
                          activity_regularizer=regularizers.l1(1e-9))(temp)


        else:
            temp = Conv2D(n_filters, (3,3), padding='same', activation='relu')(inp)
            temp = Conv2D(n_filters, (3,3), padding='same' )(temp)


        if residual_scaling:
            temp = Lambda(lambda z: z * residual_scaling)(temp)

        temp = Add()([temp, inp])

        return temp

    n_filters = 64
    # 16 residual blocks as in the Article for the baseline model.
    n_res_blocks = 16
    regularize = True
    residual_scaling = None

    if testmode:

        if hasattr(self, "img_height"):
            print("img_height: ", self.img_height)
            input_img = Input(shape=(self.img_height, self.img_width, self.channels))
        else:
            input_size = 48
            input_img = Input(shape=(input_size, input_size, self.channels))

    else:
        input_size = self.inputsize

        input_img = Input(shape=(input_size, input_size, self.channels))

    if regularize:
        x = Conv2D(n_filters, (3, 3),  padding='same', activity_regularizer=regularizers.l1(1e-9))(input_img)
    else:
        x = Conv2D(n_filters, (3, 3),  padding='same')(input_img)
    blk = x

    for i in range(n_res_blocks):
        x = residual_block(x, n_filters, residual_scaling=residual_scaling, regularize= regularize) #, residual_scaling=0.1, regularize=True)

    x = Conv2D(n_filters, (3,3), padding='same' )(x)
    x = Add()([x, blk])

    if not self.scale%2:
        for i in range(int(np.log2(self.scale))):

            if regularize:
                x = Subpixel(n_filters, 3, 2, padding='same', activity_regularizer=regularizers.l1(1e-9))(x)
            else:
                x = Subpixel(n_filters, 3, 2, padding='same')(x)

    else: # scale = 3
        if regularize:
            x = Subpixel(n_filters, 3, 3, padding='same', activity_regularizer=regularizers.l1(1e-9))(x)
        else:
            x = Subpixel(n_filters, 3, 3, padding='same')(x)
    if regularize:
        x = Conv2D(3, 3, padding="same", name='last_layer', activity_regularizer=regularizers.l1(1e-9) )(x)
    else:
        x = Conv2D(3, 3, padding="same", name='last_layer' )(x)

    m = Model(input_img, x)
    m.compile(Adam(self.lrate, self.decay, clipvalue=0.5), loss=losses.MAE)

    # m.summary()

    return m  # This method returns the compiled model of EDSR.


"""
    if self.scale == 2 or self.scale == 3:

        x = Conv2D(n_filters * (self.scale ** 2), (3,3), padding='same' )(x)

        a , h, w, c = K.get_variable_shape(x)
        shape = [K.shape(x)[k] for k in range(4)]

        if h is None or w is None:
            x = Reshape(( shape[1]*self.scale, shape[2]*self.scale, n_filters), name='Reshape_layer_1')(x)

        else:
            x = Reshape(( h * self.scale, w * self.scale, n_filters), name='Reshape_layer_1')(x)

    elif self.scale == 4:

        for i in range(2):

            x = Conv2D(n_filters * (2 ** 2), (3,3), padding='same' )(x)

            a , h, w, c = K.get_variable_shape(x)
            shape = [K.shape(x)[k] for k in range(4)]

            if h is None or w is None:
                x = Reshape(( shape[1]*2, shape[2]*2, int(n_filters / 4)), name='Reshape_layer_1')(x)

            else:
                x = Reshape(( h * 2, w * 2,  int(n_filters / 4)), name='Reshape_layer_2')(x)

    else:
        print("Unsupported scale: ", str(self.scale) )
        return  None
"""

