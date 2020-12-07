from DeepSR import DeepSR
from os.path import basename

settings = \
    {
        'augment': [90, 180, 'fliplr'],  # can be any combination of [90,180,270, 'flipud', 'fliplr', 'flipudlr' ], or []
        'backend': 'tensorflow', # keras is going to use tensorflow framework in processing.
        'batchsize':9,  # number of batches
        'channels': 1,  # color channels to be used in training . Only one channel in this case
        'colormode': 'RGB',  # the color space is RGB. 'RGB' or 'YCbCr'
        'crop': 0,  # do not crop from borders of images.
        'crop_test': 0,  # do not crop from borders of images in tests.
        'decay': 1e-6,  # learning rate decay.
        'earlystoppingpatience': 5,  # stop after 5 epochs if the performance of the model has not improved.
        'epoch': 2,  # train the model for total 10 passes on training data.
        'inputsize': 33,  # size of input image patches is 33x33.
        'lrate': 0.001,
        'lrateplateaupatience': 3,  # number of epochs to wait before reducing the learning rate.
        'lrateplateaufactor': 0.5,  # the ratio of decrease in learning rate value.
        'minimumlrate': 1e-7,  # learning rate can be reduced down to a maximum of this value.
        'modelname': basename(__file__).split('.')[0],  # modelname is the same as the name of this file.
        'metrics': ['PSNR', 'SSIM'],  # the model name is the same as the name of this file.
        'normalization': ['standard', 53.28, 40.732],  # apply standardization to input images (mean, std)
        'outputdir': '',  # sub directories automatically created.
        'scale': 4,  # maginification factor is 4.
        'stride': 11,  # give a step of 11 pixels apart between patches while cropping them from images for training.
        'target_channels': 1,  # color channels to be used in tests . Only one channel in this case
        'target_cmode': 'RGB',  # 'YCbCr' or 'RGB'
        'testpath': [r'c:\test_images'],  # path to the folder in which test images are. Can be more than one.
        'traindir': r'c:\training_images',  # path to the folder in which training images are.
        'upscaleimage': False,  # The model is going to upscale the given low resolution image.
        'valdir': r'',  # path to the folder in which validation images are.
        'workingdir': r'',  # path to the working directory. All outputs to be produced within this directory
        'weightpath': '',  # path to model weights either for training to start with, or for test.
    }

DSR = DeepSR.DeepSR(settings)  # instance of DeepSR object without the build_model method.

from keras import  losses
from keras.layers import Input
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.models import  Model
from keras.optimizers import Adam

#  a method returning an autoencoder model
def build_model(self, testmode=False):
    if testmode:
        input_size = None

    else:
        input_size = self.inputsize

    # encoder
    input_img = Input(shape=(input_size, input_size, self.channels))
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = Conv2D(16, (1, 1), activation='relu', padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(3, (1, 1), activation='relu', padding='same')(x)

    # decoder
    x = UpSampling2D((self.scale, self.scale))(x)  # upscale by the scale factor
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(16, (1, 1), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    decoded = Conv2D(self.channels, (3, 3), activation='relu', padding='same')(x)
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(Adam(self.lrate, self.decay), loss=losses.mean_squared_error)
    return autoencoder

DSR.set_model(build_model)  # set build_model function to compose a DL model in the class.

DSR.epoch =1 # model shall be trained only for 1 epoch, instead of 10 as defined in settings.

DSR.plot_model()
DSR.train()  # training procedure.

# evaluate the performance of the model
DSR.test(testpath=DSR.testpath, weightpath=DSR.weightpath, saveimages=False, plot=False)
