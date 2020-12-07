#  sample_model_1.py file. Returns a CNN model.#
from keras import metrics, losses
from keras.layers import Input, Dense
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from os.path import dirname, abspath, basename

settings = \
    {
        'augment': [],  # can be any combination of [90,180,270, 'flipud', 'fliplr', 'flipudlr' ], or []
        'backend': 'tensorflow', # keras is going to use tensorflow framework in processing.
        'batchsize':256,  # number of batches
        'channels': 1,  # color channels to be used in training . Only one channel in this case
        'colormode': 'RGB',  # the color space is RGB. 'RGB' or 'YCbCr'
        'crop': 6,  # do not crop from borders of images.
        'crop_test': 0,  # do not crop from borders of images in tests.
        'decay': 1e-6,  # learning rate decay.
        'earlystoppingpatience': 5,  # stop after 5 epochs if the performance of the model has not improved.
        'epoch': 10,  # train the model for total 10 passes on training data.
        'inputsize': 33,  # size of input image patches is 33x33.
        'lrate': 0.001,
        'lrateplateaupatience': 3,  # number of epochs to wait before reducing the learning rate.
        'lrateplateaufactor': 0.5,  # the ratio of decrease in learning rate value.
        'minimumlrate': 1e-7,  # learning rate can be reduced down to a maximum of this value.
        'modelname': basename(__file__).split('.')[0],  # modelname is the same as the name of this file.
        'metrics': ['PSNR', 'SSIM'], # modelname is the same as the name of this file.
        'normalization': ['standard', 53.28, 40.732],  # apply standardization to input images (mean, std)
        'outputdir': '',  # sub directories automatically created.
        'scale': 4,  # magnification factor is 4.
        'stride': 11,  # give a step of 11 pixels apart between patches while cropping them from images for training.
        'target_channels': 1,  # color channels to be used in tests . Only one channel in this case
        'target_cmode': 'RGB',  # 'YCbCr' or 'RGB'
        'testpath': [r'D:\test_images'],  # path to the folder in which test images are. Can be more than one.
        'traindir': r'D:\training_images',  # path to the folder in which training images are.
        'upscaleimage': True,  # Input image to be upscaled by given interpolation method before handling it over the model
        'valdir': r'c:\validation_images',  # path to the folder in which validation images are.
        'workingdir': '',  # all outputs to be yielded within this folder.
        'weightpath': '',  # path to model weights either for training to start with, or for test.
        'saveimages': True
}




#  a method returning CNN model
def build_model(self, testmode=False):
    if testmode:
        input_size = None

    else:
        input_size = self.inputsize

    input_shape = (input_size, input_size, self.channels)

    my_model = Sequential()
    my_model.add(Conv2D(64, (9, 9), kernel_initializer='glorot_uniform', activation='relu',
                        padding='valid', input_shape=input_shape))
    my_model.add(Conv2D(32, (1, 1), kernel_initializer='glorot_uniform', activation='relu',
                        padding='valid'))
    my_model.add(Conv2D(1, (5, 5), kernel_initializer='glorot_uniform', activation='relu',
                        padding='valid'))
    my_model.compile(Adam(self.lrate, self.decay), loss=losses.mean_squared_error)
    return my_model

