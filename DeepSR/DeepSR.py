"""
DeepSR.py

An open source framework for implementation of the tasks regarding Super Resolution
with Deep Learning architectures based on Keras framework.

The main goal of this framework is to empower users/researchers to focus on their studies
while pursuin g successful Deep Learning algorithms for the task of Super Resolution,
saving them from the workloads of programming, implementing and testing.

It offers several ways to use the framework, such as using the command line, using the DeepSR
object in another program, or using batch files to automate successive multiple jobs.
The DeepSR governs each steps in the workflow of Super Resolution task, such as pre-processing,
augmentation, normalization, training, post-processing, and testing, in such a simple, easy and
fast way that there would remain only a very small amount of work to users/researchers to
accomplish their experiments.


Developed  by
        Hakan Temiz             htemiz@artvin.edu.tr

Version : 0.0.51
History :

"""


import sys
mod_name = vars(sys.modules[__name__])['__package__']

if mod_name == "DeepSR":
    # print ("DeepSR is being called as module\n")
    from DeepSR.args import getArgs
else:
    # print ("DeepSR is being called as script\n")
    from args import getArgs


ARGS = getArgs()


if ARGS['gpu'] is not None:
    from tensorflow.config import list_physical_devices, set_visible_devices

    # config = tf.ConfigProto(device_count={'GPU': 1})
    try:
        physical_devices = list_physical_devices('GPU')

        # environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if len(physical_devices) >1:

            if ARGS['gpu'] == '0':
                # environ["CUDA_VISIBLE_DEVICES"]="0"
                # config.gpu_options.visible_device_list = '0'
                set_visible_devices(physical_devices[0], 'GPU')
                print('GPU (0) ALLOCATED FOR THIS JOB')

            elif ARGS['gpu'] == '1':
                # environ["CUDA_VISIBLE_DEVICES"] = "1"
                # config.gpu_options.visible_device_list = '1'
                set_visible_devices(physical_devices[1], 'GPU')
                print('GPU : (1) ALLOCATED FOR THIS JOB')

            elif ARGS['gpu'] == '2':
                # environ["CUDA_VISIBLE_DEVICES"] = "2"
                set_visible_devices(physical_devices[2], 'GPU')
                print('GPU : (2) ALLOCATED FOR THIS JOB')

            elif ARGS['gpu'] == '3':
                # environ["CUDA_VISIBLE_DEVICES"] = "3"
                set_visible_devices(physical_devices[3], 'GPU')
                print('GPU : (3) ALLOCATED FOR THIS JOB')

            elif ARGS['gpu'].lower() == 'all':
                # environ["CUDA_VISIBLE_DEVICES"] = "3"
                set_visible_devices(physical_devices, 'GPU')
                print('GPU : ALL GPUS ALLOCATED FOR THIS JOB')

    except:
        del physical_devices



from os import environ
import sys
from sys import argv
from os import  makedirs, rename
from os.path import isfile, join, exists, splitext, abspath, basename, dirname, isdir
import numpy as np
from scipy.io import loadmat
from inspect import getmembers, isfunction
from types import MethodType

if mod_name == "DeepSR": # called as module
    folderpath = dirname(argv[0])
    sys.path.append(folderpath)

    import DeepSR.utils as utils
    from DeepSR import LossHistory
else: # called as script
    import utils
    import LossHistory


from keras import backend as K

if ARGS['backend'] is not None:

    if K.backend() != ARGS['backend']:
        environ['KERAS_BACKEND'] = ARGS['backend']
        from importlib import reload
        reload(K)

if ARGS['seed'] is not None:
    np.random.seed(ARGS['seed'])
    if K.backend() == 'tensorflow':
        from tensorflow import random as tfrandom
        tfrandom.set_seed(ARGS['seed'])


import h5py
import time
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
from importlib import import_module
from tqdm import tqdm
from pathlib import Path

from skimage.color import rgb2ycbcr, rgb2gray, ycbcr2rgb
from skimage.io import imread, imsave

from keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.backend import image_data_format
from keras.layers import Activation
from keras.layers.advanced_activations import LeakyReLU, PReLU


image_extentions = ["*.bmp", "*.BMP", "*.jpg", "*.JPG", "*.png", "*.PNG", "*.jpeg", "*.JPEG", "*.TIFF", "*.tiff", '*.mat']


class DeepSR(object):

    def __init__(self, args = None):
        """ takes arguments via a dictionary object and generate class members

        """
        self.__version__ = '0.0.42'

        if args is None:
            return
        else:
            self.set_settings(args)



    def apply_activation(self, model, activation="relu", name=None):

        # there is no activation function defined
        if activation is None or activation =="":
            print("No activation function defined for this layer of model. Returned the same model without applying any activation function.")
            return model

        if activation.lower() not in ["prelu", "leakyrelu", "lrelu"]:
            return Activation(activation=activation, name=name)(model)

        elif activation.lower() == "prelu":
            return PReLU(shared_axes=[1, 2], name=name)(model)

        elif activation.lower() in ["lrelu", "leakyrelu"]:
            return LeakyReLU(name=name)(model)

        else:
            print("Unknown activation function!", activation)
            return None


    def calculate_mean_of_images(self, imagedir, separate_channels=True, direction='both' ):
        """
        Calculates the mean of images in a folder.
        :param imagedir: Folder containing images.
        :param separate_channels: Indicates that the mean values are separately calculated fro each color channnels.
        Default is True.
        :param direction: Calculation style of mean values. Use 'column' to calculate mean values of each column separately.
        Use 'row' to calculate the mean values of each rows separately. Use 'both' for calculation without separating rows or columns.
        :return: Returns the mean value of image(s).
        """

        files = utils.get_files(imagedir, image_extentions, remove_extension=False, full_path=True)

        total_images =0.0
        total_mean =0.0

        for f in files:
            if not isfile(f):
                pass
            else:
                img = self.read_image(f, colormode=self.colormode)
                mean = utils.mean_image(img, separate_channels= separate_channels, direction=direction)
                total_mean += mean
                total_images +=1
        print(total_mean, total_images)
        print('Mean : ', total_mean / total_images)
        return total_mean / total_images


    def get_number_of_samples(self, imagedir):
        """
        Gives the number of sub sample images to be produced from the images.
        :param imagedir: Path to the directory containing image(s).
        :return: The number of sub images in total
        """
        total_num = 0 # number of sub images in total

        files = utils.get_files(imagedir, image_extentions, remove_extension=False, full_path=True)

        for f in files:
            if not isfile(f):
                print('Warning! the file ', f, '  is not a valid file!')
                continue

            img = self.read_image_by_PIL(f, colormode=self.colormode)

            for im_aug in utils.augment_image(img, self.augment):

                row, col = im_aug.shape[0:2]
                # we presume remains cropped from the image
                row = int(row / self.scale)
                col = int(col / self.scale)

                if self.upscaleimage:
                    row *= self.scale
                    col *= self.scale

                ver_num = int((row - self.inputsize) // self.stride) + 1
                hor_num = int((col - self.inputsize) // self.stride) + 1

                total_num += ver_num * hor_num

        return total_num


    def generate_batch_on_fly(self, imagedir, shuffle = True):
        """
        Generates asinput and reference (ground-truth) image patches from images in a folder
        for every batches in training procedure.
        :param imagedir: path to directory where images are in
        :param shuffle: Images in batch will be shuffled if it is True.
        :return: Yields two lists composed of as many input and reference image patches as the number of batch.
        """

        lr_stride = self.stride


        # "size" variable is used for calculation of the left, right, top and bottom
        # coordinates of the output patch of ground truth image.
        # if the model's layers use border effect set to "valid", output image would be
        # smaller than the input image. Hence, the sub patch of ground truth image, relevant to
        # the sub patch of the input image must be the located at the same region and size.
        count = 0
        # iteration = 0
        #print("\nIteration : ", iteration, "\n")

        if self.upscaleimage:
            pad = int((self.inputsize - self.outputsize) / 2)
            size = self.inputsize
            hr_stride = lr_stride
        else:
            pad = int((self.inputsize * self.scale - self.outputsize) / 2)
            size = self.outputsize
            hr_stride = self.scale * lr_stride


        if image_data_format() == "channels_last":
            input_list = np.empty((self.batchsize, self.inputsize, self.inputsize, self.channels))
            output_list = np.empty((self.batchsize, self.outputsize, self.outputsize, self.channels))
        else:
            input_list = np.empty((self.batchsize, self.channels, self.inputsize, self.inputsize))
            output_list = np.empty((self.batchsize, self.channels, self.outputsize, self.outputsize))

        files = utils.get_files(imagedir, image_extentions, remove_extension=False, full_path=True)

        while True:

            if shuffle: # shuffle the file orders
                np.random.shuffle(files)

            for i in tqdm(range(len(files))):
                f = files[i]

                if not isfile(f):
                    continue

                img = self.read_image_by_PIL(f, colormode=self.colormode)

                # if channel is set 3 and the input image is a single channel image,
                # image to be converted to 3 channel image copying itself along the third
                # axis
                if len(img.shape) < 3 and self.channels ==3:
                    img = self.single_to_3channel(img, f)

                augmented_images =  utils.augment_image(img, self.augment)

                if shuffle:
                    np.random.shuffle(augmented_images)

                for img_aug in augmented_images:

                    # prepare ground truth image
                    img_ground = utils.preprocess_image(img_aug, scale=self.scale, pad=0,
                                                        channels=self.channels, upscale=False,
                                                        crop_remains=True, img_type='ground')

                    if self.normalizeground: # normalization the ground truth image if it is set to True
                        img_ground = self.normalize_image(img_ground, self.normalization)


                    # if floatmode is True, image to be processed before converting it to float32
                    # in misc.resize method. this is necessary for some situations, e.g.,
                    # for data range is beyond the range 0-255.
                    if self.floatmode:
                        mode = 'F'
                    else:
                        mode = None


                    img_input = utils.preprocess_image(img_aug, scale=self.scale, pad=0, decimation=self.decimation,
                                                       channels=self.channels, upscale=self.upscaleimage,
                                                       crop_remains=True, img_type='input', noise=self.noise,
                                                       interp_up=self.interp_up, mode=mode)

                    #
                    # Normalize image
                    #
                    if self.normalization:
                        img_input = self.normalize_image(img_input, self.normalization)

                    # how many sub pictures we can generate
                    ver_num = int((img_input.shape[0] - self.inputsize) // self.stride) + 1
                    hor_num = int((img_input.shape[1] - self.inputsize) // self.stride) + 1

                    # to get sub images in random order, we make a list
                    # and then, shuffle it
                    ver_num_list = list(range(0,ver_num))
                    hor_num_list = list(range(0,hor_num))
                    np.random.shuffle(ver_num_list)
                    np.random.shuffle(hor_num_list)

                    for i in ver_num_list:
                        for j in hor_num_list:

                            lrow_start = i * lr_stride
                            lcol_start = j * lr_stride
                            ihr_stride = i * hr_stride
                            jhr_stride = j * hr_stride

                            hrow_start= ihr_stride + pad
                            hrow_stop = ihr_stride + size - pad
                            hcol_start = jhr_stride + pad
                            hcol_stop = jhr_stride + size - pad

                            # sub patch of input image
                            sub_img = img_input[lrow_start: lrow_start + self.inputsize,
                                      lcol_start: lcol_start + self.inputsize]
                            # reference (ground truth) image patch
                            sub_img_label = img_ground[hrow_start: hrow_stop, hcol_start: hcol_stop]

                            if image_data_format() =="channels_last": # tensorflow backend
                                sub_img = sub_img.reshape([1, self.inputsize, self.inputsize, self.channels])
                                sub_img_label = sub_img_label.reshape([1, self.outputsize, self.outputsize, self.channels])
                            else:# theano backend
                                sub_img = sub_img.reshape([1, self.channels, self.inputsize, self.inputsize])
                                sub_img_label = sub_img_label.reshape([1, self.channels, self.outputsize, self.outputsize])

                            input_list[count]  = sub_img
                            output_list[count] = sub_img_label

                            if count == self.batchsize -1:
                                yield input_list, output_list
                                count = 0
                            else:
                                count += 1


    def generate_from_hdf5(self, filepath, batchsize, shuffle=False):
        """
        Generates training data in batches from training image patches stored in a .h5 file.
        :param filepath: The path to folder containing input images.
        :param batchsize: The number of image samples for a batch.
        :param shuffle: The images will be shuffled if it is True.
        :return: Two lists composed of training data (input images and corresponding reference images).
        """
        f = h5py.File(filepath, "r")
        rows = f['input'].shape[0]
        indexes = np.arange((rows // batchsize) * batchsize)
        f.close()

        while 1:
            f = h5py.File(filepath, "r")
            if shuffle:
                np.random.shuffle(indexes)

            # count how many entries we have read
            n_entries = 0
            # as long as we haven't read all entries from the file: keep reading
            while n_entries < (rows - batchsize):
                if shuffle:
                    i = indexes[n_entries: n_entries + batchsize]
                    i = np.sort(i).tolist()

                    xs = f['input'][i]
                    ys = f['label'][i]
                else:
                    xs = f['input'][n_entries: n_entries + batchsize]
                    ys = f['label'][n_entries:n_entries + batchsize]

                n_entries += batchsize
                yield (xs, ys)
            f.close()


    def generate_from_hdf5_without_shuffle(self, filepath, batchsize):
        """
        Generates training data in batches from training image patches stored in a .h5 file.
        It is the same as the method 'generate_from_hdf5' except that this method does not shuffle the images.
        :param filepath: The path to folder containing input images.
        :param batchsize: The number of image samples for a batch.
        :return: Two lists composed of training data (input images and corresponding reference images).
        """

        f = h5py.File(filepath, "r")
        rows = f['input'].shape[0]

        f.close()

        while 1:

            f = h5py.File(filepath, "r")

            # count how many entries we have read
            n_entries = 0
            # as long as we haven't read all entries from the file: keep reading
            while n_entries < (rows - batchsize):
                xs = f['input'][n_entries: n_entries + batchsize]
                ys = f['label'][n_entries:n_entries + batchsize]

                n_entries += batchsize
                yield (xs, ys)

            f.close()

    def get_backend_shape(self, input_size=None):
        """
        Forms a tuple object having the same dimensioning style  as the keras backends: Theano or Tensorflow.
        For example, Tensorflow uses following image shape (input_size, input_size, channels), while
        Theano uses (channels, input_size, input_size).
        :param input_size: The size of image. The same value is used for both width and height.
        :return: A tuple object having the shape format as Keras backend.
        """

        if image_data_format() == "channels_last":  # tensorflow style input shape
            shape = (input_size, input_size, self.channels)
        else:  # theano style input shape
            shape = (self.channels, input_size, input_size)

        return shape


    def get_mean_and_deviation(self, separate=False):
        """
        Gives the mean and deviation values from the training images.
        :param separate: Boolean. Each color channel is processed individually if it is True.
        This functionality to be added in the future releases.
        :return: The mean and standard deviation values.
        """

        files = utils.get_files(self.traindir, image_extentions, remove_extension=False, full_path=True)
        result = np.ones((len(files), 2,self.channels), np.float64)

        for i in range(len(files)):

            f = files[i]
            if not isfile(f):
                print('Warning! the file ', f, '  is not a valid file!')
                continue

            img = self.read_image(f, colormode=self.colormode)
            if len(img.shape) == 2: # single channel image
                img = img.reshape(-1,1)
            else: # multi channel image
                img = img.reshape(-1, img.shape[-1])

            result[i] = np.array([img.mean(axis=0), img.std(axis=0)])
        r  = result.mean(axis=0)

        self.mean = r[0]
        self.std  = r[1]

        return self.mean, self.std


    def model_exist(self):
        """
        Check if model exist or not.
        :return: False, if model does not exist. True, otherwise
        """
        if self.model is None:  # build_model method not successfull
            print('Model couldn\'t have been created. Please, check whether \n" +'
                  '\'build_model\' method exist and working properly')
            return False

        return True


    def normalize_and_add_noise(self, im, normalize_first=False):
        """

        :param im: Image to be normalized and/or noise added.
        :param normalize_first: Boolean. Normalization is applied to the image if it is True, and
        then, noise to be added. Or, vise versa.
        :return: Image as numpy ndarray
        """

        # pad is set zero since input image should not be cut from borders. But, Ground truth and bicubic
        # images are cut from borders as muuch as pad size in case the model reduces the size of the output
        # image as much as pad. Some models work like this to avoid border effect in calculation of metrics.
        # If the parameter of padding is set to 'valid' in layers of models in KERAS is set as , output size
        # of images are decreases as floor integer value of kernel's half width at each layers. Therefore,
        # to have the same size of ground truth, bicubic and output images, bicubic and ground truth images
        # are cropped from borders as much as padding value.
        #
        # The padding value is set to 'same' in keras layers as to have the same size of output and input
        # input images. If this is the case, padding is not applied in here since they are of the same size
        #
        if self.normalization and normalize_first:

            img_input = self.normalize_image(im, self.normalization)

            img_input = utils.preprocess_image(img_input, scale=self.scale, pad=0, decimation=self.decimation,
                                               channels=self.channels, upscale=self.upscaleimage,
                                               crop_remains=True, img_type='input', noise=self.noise,
                                               interp_up=self.interp_up)

        elif self.normalization and not normalize_first:


            img_input = utils.preprocess_image(im, scale=self.scale, pad=0, decimation=self.decimation,
                                               channels=self.channels, upscale=self.upscaleimage,
                                               crop_remains=True, img_type='input', noise=self.noise,
                                               interp_up=self.interp_up)

            img_input = self.normalize_image(img_input, self.normalization)

        else:

            img_input = utils.preprocess_image(im, scale=self.scale, pad=0, decimation=self.decimation,
                                               channels=self.channels, upscale=self.upscaleimage,
                                               crop_remains=True, img_type='input', noise=self.noise,
                                               interp_up=self.interp_up)

        return img_input


    def normalize_image(self, img, normalization=None):
        """
        Normalizes given image. The following normalization procedures can be implemented:

        Min-Max, divide by a value, mean normalization, and standardization.

        :param img: The image to be normalization procedure applied.
        :param normalization: List type. normalization method. Can be any of the followings:
                            divide, minmax, mean, standard. Please refer to documentation for
                            further information.
        :return: Normalized image.
        """
        if normalization is None or normalization == '':
            print('Normalization method is set None! Image is not normalized.')
            return img

        if normalization[0] == 'divide':

                if self.divident is None:
                    if len(normalization) == 2:
                        self.divident = float(normalization[1])

                    else: # divident is not given. take it as 255.0
                        self.divident = 255.0
                    print("WARNING : \n\tThe divident value of division normalization is not given. The divident is set as 255.0")

                img = img.astype(np.float64) / self.divident
        #
        # MINMAX NORMALIZATION
        # Inew = (I - min) * (newMax - newMin) / (max - min) + newMin
        #
        elif normalization[0] == 'minmax':

            if self.minimum is None or self.maximum is None:

                if len(normalization) == 3:
                    self.minimum = float(normalization[1])
                    self.maximum = float(normalization[2])

                else: # not any minimum or maximum value was given set them as 0 and 1 respectivelly.
                    print("WARNING : \n\tNeither of minimum or maximum value was given. They are set to 0 and 1, respectivelly!")
                    self.minimum = 0.0
                    self.maximum = 1.0

            minimum = img.min(axis=(0, 1))
            maximum = img.max(axis=(0, 1))

            img = (img - minimum) * (self.maximum - self.minimum) / (maximum - minimum) + self.minimum


        elif normalization[0] == 'standard':

            if 'single' in normalization or len(normalization) == 1:
                mean = img.mean(axis=(0, 1))
                std = img.std(axis=(0, 1))
                img = (img - mean) / std

                # for back normalization
                self.old_mean = mean
                self.old_std = std

            elif self.mean is not None and self.std is not None:
                img = (img - self.mean) / self.std

                # for back normalization
                self.old_mean = self.mean
                self.old_std = self.std

            elif 'whole' in normalization: # means that mean and standard deviation to be calculated from training set

                print("Calculating the mean and the standard deviation from the training set...")
                self.get_mean_and_deviation()
                print("Done!")

                # for back normalization
                self.old_mean = self.mean
                self.old_std = self.std

            elif len(normalization) == 3:
                self.mean = float(normalization[1])
                self.std = float(normalization[2])
                img = (img - self.mean) / self.std

                # for back normalization
                self.old_mean = self.mean
                self.old_std = self.std
                img = (img - self.mean) / self.std

            else:
                print("WARNING: \n\tSome parameters of standardization is missing. \"single\" parameter is set in order to \
                 \n\tcalculate mean and standard deviation from each images individually to continue.")
                mean = img.mean(axis=(0, 1))
                std = img.std(axis=(0, 1))
                img = (img - mean) / std

                # for back normalization
                self.old_mean = mean
                self.old_std = std

        elif normalization[0] == 'mean':

            if 'single' in normalization or len(normalization) == 1:
                self.mean = img.mean(axis=0)
                img = img - self.mean

            # mean is calculated from entire training images, or
            # mean value with argument is given
            # if length is 4, mean values for each color channels are given

            elif self.mean is not None:
                img = img - self.mean

            elif 'whole' in normalization:
                self.get_mean_and_deviation()

                # for back normalization
                self.old_mean = self.mean
                self.std = None  # standard deviation is not necessary
                print('Done!')
                img = img - self.mean

            else: # mean value is not given. calculate from the image
                print('Something is wrong with mean normalization!. Normalization argument is as follows:\n',
                      normalization)
                print('Image was processed without mean normalization!')

        else:
            # print('Unknown normalization procedure! Image is not being normalized')
            pass

        return img


    def normalize_image_back(self, img):
        """
        This method applies the reverse of normalization procedure to given image.
        :param img: Image to be reverse normalization
        :return: Image that reverse normalized.
        """
        if self.normalization is None:
            return img

        if self.normalization[0] == 'divide':
                img = np.uint8(img * self.divident)

        elif self.normalization[0] == 'minmax':
            pass

        elif self.normalization[0] == 'standard':
            if 'single' in self.normalization:
                print('WARNING! \n\tSince standardization type is set to \"single\", image can not be normalized back! \
                      \n\tImage returned back intact  without any processing!')

            elif self.mean is not None and self.std is not None:
                img = np.uint8(img * self.std  + self.mean)

            else: # ambiguous situation
                print('WARNING!\n\tAmbiguous situation has occurred since standardization method is not clear!')

        elif self.normalization[0] == 'mean':
            if self.mean is not None:
                img += self.mean

            else: # mean value is not given. calculate from the image
                print('WARNING! \n\tSince mean value is not known, image can not be normalized back with mean value! \
                                      \n\tImage returned back without any processing!')

        return img


    def plotimage(self, im_input, weight_file, plot=True):
        """
        Plots the output image of model along with reference image and interpolation image.
        :param im_input: Input image.
        :param weight_file: weight file of model.
        :param plot: Images to be plotted if it is True.
        :return: Result image along with PSNR and SSIM measures.
        """

        self.model = self.build_model(testmode=True)
        if not self.model_exist():  # check if model exists
            print('!ERROR: model not exist! Please, build SR with a model!')
            return None, None, None

        img = self.read_image(img_input, colormode=self.colormode)


        # prepare ground image
        img_ground = utils.preprocess_image(img, scale=self.scale, pad=self.crop, channels=3,
                                            upscale=False, crop_remains=True, img_type='ground')
        # prepare bicubic image
        img_bicubic = utils.preprocess_image(img, scale=self.scale, pad=self.crop, channels=3,
                                             upscale=True, crop_remains=True, img_type='bicubic',
                                             noise=self.noise, interp_up='bicubic')

        # prepare low resolution image
        img_input = utils.preprocess_image(img, scale=self.scale, pad=0, channels=self.channels,
                                           upscale=self.upscaleimage, crop_remains=True,
                                           img_type='input', noise=self.noise,
                                           decimation=self.decimation, interp_up=self.interp_up)

        img_input = utils.prepare_image_for_model(img_input, self.channels, image_data_format())

        # Prediction
        img_result = self.predict(img_input, weight_file)

        # PSNR and SSIM
        psnr_model, ssim_model = \
            utils.calc_metrics_of_image(img_ground, img_result, self.crop_test, self.target_channels)

        # if one channel is used only , other channels gathered from bicubic image
        if self.channels == 1:
            im_tmp = img_bicubic
            im_tmp[:, :, 0] = img_result[:, :]
            img_result = im_tmp

        if self.colormode != self.target_cmode:
            if plot:
                img_ground = Image.fromarray(img_ground, self.colormode).convert(self.target_cmode)
                img_bicubic = Image.fromarray(img_bicubic, self.colormode).convert(self.target_cmode)

                img_result = Image.fromarray(img_result, self.colormode).convert(self.target_cmode)

        if plot:
            plt.subplot(221)
            plt.imshow(img_ground)
            plt.title('Ground Image')

            plt.subplot(222)
            plt.imshow(img_bicubic)
            plt.title('Bicubic Image')

            ax = plt.subplot(224)
            plt.imshow(img_result)
            plt.title('Output Image')
            plt.text(0, 0, "PSNR: {0:.4f},  SSIM: {1:.4f}".format(psnr_model, ssim_model), color="red",
                     transform=ax.transAxes)

            plt.subplot(223)
            plt.imshow(img_result)
            plt.title('Exact Output Image')

            plt.tight_layout(pad=2, w_pad=4., h_pad=2.0)
            plt.show()

        return img_result, psnr_model, ssim_model


    def plot_all_layer_outputs(self, img_input, name, plot=False, saveimages=False, change_colorspace= False):
        """"
        Plots the outputs of each layer.
        :param img_input: Input image.
        :param name: The name of the layer.
        :param plot: The layer outputs to be plotted if it is True.
        :param saveimages: Image(s) to be saved if it is True.
        :return:
        """

        import keras.backend as K
        print('****  shape  *****')
        print(img_input.shape)

        for i in range(1, len(self.model.layers)):
            lay = self.model.layers[i]
            get_activations = K.function([self.model.layers[0].input, K.learning_phase()], [lay.output, ])
            activations = get_activations([img_input, 0])[0]

            for j in range(activations.shape[3]):
                r = activations[0, :, :, j].copy()
                if plot:
                    plt.imshow(r, cmap='gray')
                    plt.title('Layer {} , feature map {}'.format( lay.name, j+1))
                    plt.tight_layout()
                    plt.show()


                if saveimages:
                    if change_colorspace and self.colormode == 'YCbCr':

                        # In the case the color space is 'YCbCr', we should convert images from numpy array to PIL
                        # format with the argument (mode = 'YCbCr')
                        Image.fromarray(r, mode='YCbCr').save(name + '_' + lay.name + '_filter_' + str(j) + '.tif')

                    else:
                        Image.fromarray(r).save(name + '_' + lay.name + '_filter_' + str(j) + '.tif')


    def plot_history(self, save=False):


        if self.model.loss.__name__ in ["mse", "mean_squared_error"]:
            MSE = True

        else:
            MSE = False

        dpi=300

        font_size = 12.25 - (np.log(self.epoch) - np.log10(self.epoch)) / 5
        w = 3 + (np.log(self.epoch) / 4 * np.log10(self.epoch))
        h = 2 + np.sqrt(np.log10(self.epoch) * np.log(self.epoch) / 4)

        plt.rcParams['figure.figsize'] = [w, h ]

        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams['figure.dpi'] = dpi
        plt.rcParams['savefig.dpi'] = dpi
        plt.rcParams['font.size'] = font_size
        plt.rcParams['legend.fontsize'] = 'small'
        plt.rcParams['figure.titlesize'] = 'small'
        plt.rcParams['xtick.labelsize'] = "small"
        plt.rcParams['ytick.labelsize'] = "small"

        plt.figure(1)

        legend_names =[]

        for key in sorted(self.history.history.keys()):

            if key =="lr":
                continue

            if not all(v is None for v in self.history.history[key] ):
                try:
                   name = "Training Loss" if key =="loss" else "Validation Loss"
                   legend_names.append(name)

                   plt.figure(1)
                   plt.plot(self.history.history[key])

                   if MSE:
                       plt.figure(2)
                       plt.plot(self.to_psnr(self.history.history[key]))

                except IOError:
                    print('Something went wrong!')


        ticks = len(self.history.history[list(self.history.history.keys())[0]])

        if ticks >= 400:
            interval = ticks//50
            div=50
        elif ticks >= 200:
            interval = ticks//25
            div=25
        elif ticks >= 100:
            interval = ticks//20
            div=20
        elif ticks >= 50:
            interval = ticks//5
            div=5
        elif ticks > 20:
            interval = ticks//2
            div=2
        else:
            interval=ticks
            div=1

        locs=[]

        for i in range(interval):
            locs.append(i*div-1)


        if (ticks - 1) not in locs:
            locs.append(ticks - 1)

        locs[0] = 0

        plt.figure(1)
        # locs = np.uint((np.linspace(0, ticks - 1, ticks // div)))
        plt.xticks(locs, [x+1 for x in locs])

        plt.title('Training Graph')
        plt.legend(legend_names)
        plt.xlabel("Epochs")
        plt.ylabel(self.model.loss.__name__)
        plt.tight_layout()

        plt.savefig(join(self.outputdir, "Training graph of " + self.modelname +".png"))

        new_legend_names = []

        for name in legend_names:

            if "Training" in name:
                name = "Training"
            else:
                name = "Validation"

            new_legend_names.append(name)

        if MSE:
            plt.figure(2)
            plt.xticks(locs, [x + 1 for x in locs])
            plt.title('Training Graph')
            plt.legend(new_legend_names)
            plt.xlabel("Epochs")
            plt.ylabel("PSNR (dB)")
            plt.tight_layout()
            plt.savefig(join(self.outputdir, "Training graph of " + self.modelname + " in PSNR.png"))


    def plot_model(self, to_file=True, dpi=600):
        """
        Plots the diagram of the model using Keras functionality.
        :param to_file: Boolean. Saves the model's diagram as a .png file in the output folder.
        :param dpi: Resolution in dpi.
        :return:
        """
        from keras.utils import plot_model
        if to_file:

            path= Path(abspath(self.outputdir)).parent

            plot_model(self.model, to_file=join(path, self.modelname + '.png'))
        else:
            plot_model(self.model)
            plt.show()


    def plot_loss_history(self, save=False):

        dpi=300
        w = 5 + np.sqrt(np.log(self.nb_batches)) * np.log10(self.nb_batches)
        plt.rcParams['figure.figsize'] = [w, 5.0]

        font_size= 13- (np.log(self.nb_batches) - np.log10(self.nb_batches)) / 2

        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams['figure.dpi'] = dpi
        plt.rcParams['savefig.dpi'] = dpi
        plt.rcParams['font.size'] = font_size
        plt.rcParams['legend.fontsize'] = 'medium'
        plt.rcParams['figure.titlesize'] = 'medium'
        plt.rcParams['xtick.labelsize'] = "medium"
        plt.rcParams['ytick.labelsize'] = "medium"

        legend_names =[]

        plt.figure()

        for key in self.LossHistory.history.keys():

            if key =="lr":

                continue

            if not all(v is None for v in self.LossHistory.history[key] ):

                try:
                   plt.plot(self.LossHistory.history[key])

                   name = "Training Loss" if key =="loss" else "Validation Loss"
                   legend_names.append(name)

                except IOError:
                    print('Something went wrong!')

        plt.title('Losses per Batches')
        plt.legend(legend_names)
        plt.xlabel("Batches")
        plt.ylabel(self.model.loss.__name__)
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig( join(self.outputdir, "Batch losses.png"))
        # plt.show()


    def plot_layer_output(self, im, layer_idx, saveimages=False):
        """
        Plots the output of a particular layer.
        :param im: Input image.
        :param layer_idx: The index number of the layer whose outputs to be plotted.
        :param saveimages:  Image to be saved if it is True.
        :return:
        """

        import keras.backend as K

        im = self.read_image(im, colormode=self.colormode)

        img_input = utils.preprocess_image(im, scale=self.scale, pad=0, decimation=self.decimation,
                                           channels=self.channels, upscale=self.upscaleimage,
                                           crop_remains=True, img_type='input', noise=self.noise,
                                           interp_up=self.interp_up)

        if self.normalization:
            img_input = self.normalize_image(img_input, self.normalization)

        img_input = utils.prepare_image_for_model(img_input, self.channels, image_data_format())

        lay = self.model.layers[layer_idx]

        get_activations = K.function([self.model.layers[0].input, K.learning_phase()], [lay.output, ])
        activations = get_activations([img_input, 0])[0]

        for i in range(activations.shape[3]):
            r = activations[0, :, :, i].copy()
            plt.imshow(r, cmap='gray')
            plt.title('Layer {} , feature map {}'.format( lay.name, i+1))
            plt.tight_layout()
            plt.show()


    def plot_layer_weights(self, saveimages=True, plot=False, name='', dpi=300):
        """
        Plots layer weights on screen and/or as images.
        :param saveimages:Boolean. Layer weights to be saved as images if it is True.
        :param plot: Boolean. Layer weights to be drawn as figures if it is True.
        :param name: A name prefixed before layer names.
        :param dpi:  Resolution in dpi.
        :return:
        """

        border = 1
        for lay in self.model.layers:
            weights = lay.get_weights().copy()

            if len(weights) != 0:

                count=1
                weights = weights[0] # first part of layers, second part is bias of layer

                if  len(weights.shape) == 4:


                    h, w = weights.shape[0], weights.shape[1]
                    nrows, ncols= weights.shape[2], weights.shape[3]
                    nsamp = nrows * ncols
                    weights= weights.reshape(h, w, nsamp).copy()
                    weights= weights.swapaxes(1,2).swapaxes(0,1)

                    mosaic = np.ma.masked_all((nrows * h + (nrows - 1) * border,
                                            ncols * w + (ncols - 1) * border),
                                           dtype=np.float64)
                    paddedh = h + border
                    paddedw = w + border

                    for i in range(nsamp):
                        row = int(np.floor(i / ncols))
                        col = i % ncols

                        mosaic[row * paddedh:row * paddedh + h,
                        col * paddedw:col * paddedw + w] = weights[i,:,:]
                    fig = plt.figure()

                    ax = plt.subplot()
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    im = ax.imshow(mosaic, interpolation=None, cmap='gray')
                    ax.set_title(
                        "Layer '{}' of type {}".format(lay.name, lay.__class__.__name__))

                    fig.tight_layout()

                    if plot :
                        plt.show()

                    if saveimages :
                        ax.figure.savefig(name + '_layer_' + lay.name + '_' + str(count)+ '.png', dpi=dpi)

                    count +=1


    def predict(self, img_input, weight_file, normalizeback = False):
        """
        Returns the output image of the model.
        :param img_input: Input image
        :param weight_file: Weight file to load the model with.
        :param normalizeback: Boolean. Reverse normalization to be applied to the output image if it is True.
        :return:
        """

        if not self.model_exist():  # check if model exists
            print("ERROR! Model not exists!")
            return None

        self.model.load_weights(weight_file)

        img_result = self.model.predict(img_input, batch_size=1)
        if image_data_format() == "channels_last":  # tensorflow backend
            if self.channels == 1:
                img_result = img_result[0, :, :, 0]
            else:
                img_result = img_result[0, :, :, 0:self.channels]
        else:  # theano backend
            if self.channels == 1:
                img_result = img_result[0, 0, :, :]
            else:
                img_result = img_result[0, self.channels, :, :]

            # since THEANO has image channels first in order,
            # channels need to put in last in order (w,h, channels)
            img_result = img_result.transpose((1, 2, 0))


        # normalization image back if normalization is applied.
        # if normalization is applied and it is necessary to reverse the normalization back.
        if normalizeback and self.normalization is not None :  # normalization the input image if it is set to True
            img_result = self.normalize_image_back(img_result)
            return img_result

        else:
            return img_result
            # return img_result.astype(np.uint8)


    def prepare_dataset(self, imagedir=None, datafile=None):
        """
        Constructs a dataset as a .h5 file containing input and corresponding reference images for training.
        :param imagedir: Path to the folder containing training images.
        :param datafile: Path to the output file.
        :return:
        """

        if not exists(self.datadir):
            makedirs(self.datadir)

        if datafile is None:
            datafile = self.datafile

        if imagedir is None:
            imagedir = self.traindir

        chunks = 3192
        input_nums = 1024
        lr_stride = self.stride

        if self.upscaleimage:
            hr_stride = self.scale * lr_stride
        else:
            hr_stride = lr_stride


        with h5py.File(datafile, 'w') as hf:
            if image_data_format() == "channels_last":  # tensorflow backend
                hf.create_dataset("input", (input_nums, self.inputsize, self.inputsize, self.channels),
                                 maxshape=(None, self.inputsize, self.inputsize, self.channels),
                                 chunks=(self.batchsize, self.inputsize, self.inputsize, self.channels),
                                 dtype='float32')
                hf.create_dataset("label", (input_nums, self.outputsize, self.outputsize, self.channels),
                                 maxshape=(None, self.outputsize, self.outputsize, self.channels),
                                 chunks=(self.batchsize, self.outputsize, self.outputsize, self.channels),
                                 dtype='float32')
            else:
                hf.create_dataset("input", (input_nums, self.channels, self.inputsize, self.inputsize),
                                 maxshape=(None, self.channels, self.inputsize, self.inputsize),
                                 chunks=(128, self.channels, self.inputsize, self.inputsize),
                                 dtype='float32')
                hf.create_dataset("label", (input_nums, self.channels, self.outputsize, self.outputsize),
                                 maxshape=(None, self.channels, self.outputsize, self.outputsize),
                                 chunks=(128, self.channels, self.outputsize, self.outputsize),
                                 dtype='float32')

        count = 0

        files = utils.get_files(imagedir, image_extentions, remove_extension=False, full_path=True)

        for f in files:
            if not isfile(f):
                continue
            print(f)

            img = self.read_image(f, colormode=self.colormode)

            for ref_image in utils.augment_image(img, self.augment):

                w, h, c = ref_image.shape
                w -= int(w % self.scale)  # exact fold of the scale
                h -= int(h % self.scale)  # exact fold of the scale

                # prepare ground truth, input and bicubic images
                img_ground, img_bicubic, img_input = \
                    utils.prepare_input_ground_bicubic(ref_image, scale=self.scale, pad=self.crop,
                                                       upscaleimage=self.upscaleimage, channels=self.channels)


                # how many sub pictures we can generate
                ver_num = int((h - self.outputsize) / hr_stride)
                hor_num = int((w - self.outputsize) / hr_stride)

                h5f = h5py.File(datafile, 'a')

                if count + chunks > h5f['input'].shape[0]:
                    input_nums = count + chunks

                    if image_data_format() == "channels_last":  # tensorflow style image ordering
                        h5f['input'].resize((input_nums, self.inputsize, self.inputsize, self.channels))
                        h5f['label'].resize((input_nums, self.outputsize, self.outputsize, self.channels))
                    else:  # theano style image ordering
                        h5f['input'].resize((input_nums, self.channels, self.inputsize, self.inputsize))
                        h5f['label'].resize((input_nums, self.channels, self.outputsize, self.outputsize))

                for i in range(0, hor_num):
                    for j in range(0, ver_num):
                        lrow_start = i * lr_stride
                        lcol_start = j * lr_stride

                        sub_img = img_input[lrow_start: lrow_start + self.inputsize,
                                  lcol_start: lcol_start + self.inputsize]
                        if image_data_format() == "channels_last":  # tensorflow backend
                            sub_img = sub_img.reshape([1, self.inputsize, self.inputsize, self.channels])
                        else:  # theano backend
                            sub_img = sub_img.reshape([1, self.channels, self.inputsize, self.inputsize])

                        ihr_stride = i * hr_stride
                        jhr_stride = j * hr_stride
                        sub_img_label = img_ground[ihr_stride + self.crop: ihr_stride + self.outputsize,
                                        jhr_stride + self.crop: jhr_stride + self.outputsize]

                        if image_data_format() == "channels_last":  # tensorflow backend
                            sub_img_label = sub_img_label.reshape([1, self.outputsize, self.outputsize, self.channels])
                        else:
                            sub_img_label = sub_img_label.reshape([1, self.channels, self.outputsize, self.outputsize])

                        h5f['input'][count] = sub_img
                        h5f['label'][count] = sub_img_label
                        count += 1

        if image_data_format() == "channels_last":  # tensorflow backend
            h5f['input'].resize((count, self.inputsize, self.inputsize, self.channels))
            h5f['label'].resize((count, self.outputsize, self.outputsize, self.channels))
        else:  # THENAO style image ordering
            h5f['input'].resize((count, self.channels, self.inputsize, self.inputsize))
            h5f['label'].resize((count, self.channels, self.outputsize, self.outputsize))

        h5f.close()


    def prepare_callbacks(self):
        """
        Builds callback delegates for Keras model.
        :return:
        """
        loss= 'loss'

        losses = [loss]

        callbacks = []

        epoc_num = len(str(self.epoch))

        if epoc_num <= 1:
            epoc_num = 2

        epoc_num = str(epoc_num)
        txt = 'weights.{epoch:0' + epoc_num + 'd}.h5'
        path = join(self.outputdir, txt)

        self.LossHistory = LossHistory.LossHistory()

        # If there is not any validation image(s), so loss method should
        # be 'loss', 'val_loss', otherwise
        if self.valdir is not None or self.valdir != '':
           losses.append('val_loss')

        model_checkpoint = ModelCheckpoint(path, monitor=losses, save_best_only=False,
                                           mode='min', save_weights_only=False)

        callbacks.append(self.LossHistory)
        callbacks.append(model_checkpoint)
        callbacks.append(CSVLogger(self.outputdir + '/training.log'))

        # check if following parameters are given in old fashion
        if hasattr(self, "earlystoppingpatience"):
            print("Warning! provide 'earlystoppingpatience' with the new name 'espatience'. In Future it will be removed.",
                  "It is rewritten in the class as 'espatience'")
            self.espatience = self.earlystoppingpatience

        if hasattr(self, "lrateplateaupatience"):
            print(
                "Warning! provide 'lrateplateaupatience' with the new name 'lrpatience'. In Future it will be removed.",
                "It is rewritten in the class as 'lrpatience'")
            self.lrpatience = self.lrateplateaupatience

        if hasattr(self, "lrateplateaufactor"):
            print(
                "Warning! provide 'lrateplateaufactor' with the new name 'lrfactor'. In Future it will be removed.",
                "It is rewritten in the class as 'lrpatience'")
            self.lrfactor = self.lrateplateaufactor

        if hasattr(self, "espatience") and self.espatience is not None:
            callbacks.append(EarlyStopping(monitor=loss, patience=self.earlystoppingpatience, verbose=1))

        if hasattr(self, "lrpatience") and hasattr("lrfactor") and \
                self.lrpatience is not None and self.lrfactor is not None:

            callbacks.append(ReduceLROnPlateau(monitor=loss, factor=self.lrateplateaufactor,
                                               patience=self.earlystoppingpatience,
                                               verbose=1, mode='min', min_lr=self.minimumlrate))

        # append user-defined callbacks
        if hasattr(self, "callbacks"):
            for cbk in self.callbacks():

                callbacks.append(cbk)

        return callbacks


    def print_weights(self):
        """
        Prints layer weights in command prompt.
        :return:
        """

        for layer in self.model.layers:
            g = layer.get_config()
            h = layer.get_weights()
            print(g)
            print(h)


    def read_image_by_PIL(self, im_path, colormode='RGB'):
        """
        Reads an image file and converts it to a particular color space.
        :param im_path: Path to image file
        :param asGray: Boolean. Image is converted to gray-scale if it is True.
        :return: Image in a particular color space ( 'RGB' or 'YCbCr')
        """

        if '.mat' in im_path:
            mat = utils.load_mat_file(im_path)
            mat = mat.astype(np.float32)
            return mat

        im = Image.open(im_path)

        if colormode == 'RGB':
            im = im.convert('RGB')

        elif colormode == 'Gray':
            im = im.convert('L')

        elif 'YCbCr' in colormode and im.mode != 'YCbCr':
            im = im.convert('YCbCr')

        else:
            print("Unknown color space! Image was read in default color space.")

        im = np.array(im)

        return im



    def read_image(self, im_path, colormode='RGB'):
        """
        Reads an image file and converts it to a particular color space.
        :param im_path: Path to image file
        :param asGray: Boolean. Image is converted to gray-scale if it is True.
        :return: Image in a particular color space ( 'RGB' or 'YCbCr')
        """

        if '.mat' in im_path:
            return utils.load_mat_file(im_path)

        if colormode == 'Gray':

            im = imread(im_path, as_gray=True)
            return

        else :
            im = imread(im_path)

        if colormode == 'YCbCr':

            im = rgb2ycbcr(im)

        return im


    def repeat_test(self, count):
        """
        Repeats the test procedure only for one epoch at a time.
        :param count: The number repeats.
        :return:
        """

        scale = str(self.scale)
        csv_file= self.outputdir + "/results.csv"

        for i in range(0, count):
            print("\n*** ITERATION %03d ***" % (i+1))
            result_folder = abspath(join(self.outputdir, 'repeat_test', str(i + 1)))
            if not exists(result_folder):
                utils.makedirs(result_folder)

            self.set_model(self.build_model)

            self.train_on_fly()
            _im, d = self.test()
            df = None
            if i == 0:
                df = pd.DataFrame(d[scale].loc['Mean']).transpose()
                df.rename(index={'Mean': str(i + 1)}, inplace=True)
                results = df
            else:
                df = pd.DataFrame(d[scale].loc['Mean']).transpose()
                df.rename(index={'Mean': str(i + 1)}, inplace=True)
                results = results.append(df)

            files = utils.get_files(self.outputdir, ["*.h5", '*.xlsx', '*.txt', '*.log'], remove_extension=False,
                                      full_path=False)
            for f in files:
                rename(abspath(join(self.outputdir, f)), result_folder + "/" + f)

        min = results.min()
        mean = results.mean()
        max = results.max()

        results.loc['Min'] = min
        results.loc['Mean'] = mean
        results.loc['Max'] = max

        results.to_csv(csv_file, sep=';')

        # since Excel uses comma (,) for number dot
        t = ''
        with open(csv_file, 'r') as f:
            t = f.read()

        with open(csv_file, 'w') as f:
            t = t.replace('.', ',')
            f.write(t)


    def save_batch_losses(self ):
        """
        Saves the losess calculated after each batch update.
        """
        print('\nWriting Batch losses...', end='')

        """
        for key in self.history.history.keys():


            if key == "lr" :
                continue


            if not all(v is None for v in self.history.history[key] ):

                file = join(self.outputdir, key + '.txt')

                try:
                   np.savetxt(file, self.history.history[key], fmt="%s")

                except IOError:
                    print('I/O error')
        """

        for key in self.LossHistory.history.keys():

            if key == "lr" :
                continue

            if not all(v is None for v in self.LossHistory.history[key] ):

                name = "Batch losses in Training.txt" if key =="loss" else "Batch losses in Validation.txt"

                file = join(self.outputdir, name)

                try:
                   np.savetxt(file, self.LossHistory.history[key], fmt="%s")

                except IOError:
                    print('I/O error')

        print('Done!')

    def save_image(self, image, filename):
        """
        Saves given image in numpy ndarray format to an image file.
        :param image: Image as numpy array.
        :param filename: The full name of Image file
        :return: None
        """

        # convert it to uint image to save
        if image.dtype != 'uint8':
            image = image.astype(np.uint8)

        # transform image from YCbCr color space to RGB
        if self.colormode == 'YCbCr':

            im_save = self.YCbCr_array_to_RGB(image)
            im_save.save(filename)

        else:  # image is already in RGB color space

            Image.fromarray(image).save(filename)


    def save_history(self, history):
        """
        Stores the output values in a file after each epoch.
        :param history:
        :return:
        """
        length = len(history.history)
        try:
            f = open(self.outputdir + "/history.txt", 'w')
            for k in history.history.keys():

                f.write(str(k) + "; ")

            for i in range(0, length):
                text= "\r\n"

                for v in history.history.values():
                    text += str(v[i]) + "; "
                f.write(text)

            f.close()
        except:
            return False


    def set_model(self, build_fn):
        """
        Builds the model for training or test procedures.
        :param build_fn: A method returns a Keras model. Build function must take a parameter,
                            that is Boolean. This parameter indicates whether the model
                            is constructed for training, or test. If True, test mode is assumed.
                            For example:
                                def a_build_function(self,  testmode=True)
                                    model = keras.models.Sequential()
                                    ...
                                    return model
        :return:
        """
        self.build_model = build_fn
        if hasattr(self, 'mode') and self.mode == "test":
            self.model = self.build_model(testmode=True)
            print("Model has been created for TEST")
        else:
            self.model = self.build_model(testmode=False)
            print('Model has been created for TRAINING')


    def set_settings(self, args=None):
        """
        Initials the DeepSR object.
        :param args: Command arguments.
        :return:
        """

        args_has_build_function=False

        if args is None:
            print("There is not any parameters given. Class was constructed without any parameters")
            return

        command_parameters = ["train", "test", 'predict', "repeat_test", 'shutdown', 'plotimage']

        self.custom_train = None # No custom training function yet, defined by user
        self.custom_test = None # No custom test function yet, defined by user
        self.user_callbacks = None

        # parameters passed via command line has priority to the parameters in model file
        # so override the parameters that prohibited in command line
        for key in args.keys():
            if args[key] != None and key not in command_parameters:
                setattr(self, key, args[key])

        if  'modelfile' in args and args['modelfile'] != "" :
            from sys import path # necessary for importing the module
            path.append(dirname(args['modelfile']))
            model_name = basename(args['modelfile']).split('.')[0]
            module = import_module(model_name, args["modelfile"])

            # take setting from settings dictionary that exist in the module
            for key in module.settings.keys():
                setattr(self, key, module.settings[key])

            self.modelfile = args['modelfile']
            args_has_build_function = True

            #                                                 #
            # -- Import all member methods from the module -- #
            #                                                 #
            funcs = [o[0] for o in getmembers(module, isfunction)]

            for fn in funcs: # member methods defined in the module file
                method = MethodType(getattr(module, fn), self)
                setattr(self, fn, method)
            # -- done -- #

        else:
            print("Warning! DeepSR Class created without any model! Please set a model before training or testing.\n" +
                  "Or, use the class by scripting in command line." +
                  "\nRefer to the program documentation for information and instructions")

        for key in args.keys():
            if ('--'+key) in argv:
                if args[key] != None and args[key] != "" and key not in command_parameters:

                    setattr(self, key, args[key])

        """
        if hasattr(self, 'modelfile'):
            self.workingdir = dirname(self.modelfile)
        """
        if not hasattr(self, 'workingdir') or self.workingdir == "":  # working directory is not set
            from os import getcwd  # current folder is the working folder
            self.workingdir = getcwd()

        if not hasattr(self, 'outputdir') or self.outputdir == "":
            self.outputdir = abspath(self.workingdir + '/' + self.modelname + "/output/" + str(self.scale))

        if not exists(self.outputdir):
            makedirs(self.outputdir)

        if not hasattr(self, "datadir") or self.datadir =="":
            self.datadir = abspath(self.workingdir + '/' + self.modelname + '/data')
            # if not exists(self.datadir):
            #     makedirs(self.datadir)

        if not hasattr(self, "target_cmode") or self.target_cmode == "":
            self.target_cmode = self.colormode

        #
        # set the OUTPUTSIZE #
        if self.upscaleimage:
            self.outputsize = self.inputsize - 2 * self.crop # if pad value exist, subtract from the outputsize
        else:
            self.outputsize = self.inputsize * self.scale

        # prepare file names #
        data_file = "training_" + image_data_format() + "_" + str(self.scale) + "_" + \
                    str(self.inputsize) + "_" + str(self.outputsize) + \
                    "_" + str(self.stride) + ".h5"

        self.datafile = join(self.datadir, data_file)

        validation_file  = "validation_" + image_data_format() + "_" + str(self.scale) + "_" + \
                    str(self.inputsize) + "_" + str(self.outputsize) + \
                    "_" + str(self.stride) + ".h5"

        self.validationfile = join(self.datadir, validation_file)

        #
        # Find the normalization method and set relevant values for processing
        #
        if hasattr(self, 'normalization') and \
                (self.normalization is not None or self.normalization != False or self.normalization != '' ):

            if self.normalization[0] == 'standard':
                if 'whole' in self.normalization: # means that mean and standard deviation to be calculated from training set
                    print("Calculating the mean and the standard deviation from the training set...")
                    self.get_mean_and_deviation()
                    print("Done!")

                    # for back normalization
                    self.old_mean = self.mean
                    self.old_std = self.std

                elif 'single' in self.normalization:
                    self.mean = None
                    self.std = None

                elif len(self.normalization) == 3:
                    self.mean = float(self.normalization[1])
                    self.std = float(self.normalization[2])

                else:
                    print("WARNING: \n\tSome parameters of standardization is missing. \"single\" parameter is set in order to \
                     \n\tcalculate mean and standard deviation from each images individually to continue.")
                    self.normalization = ['standard', 'single']
                    self.mean = None
                    self.std = None

            elif self.normalization[0] == 'divide': # divide each image with a value

                if len(self.normalization) == 2:
                    self.divident = float(self.normalization[1])
                else: # divident is not given. take it as 255.0
                    self.divident = 255.0
                    print("WARNING : \n\tThe divident value of division normalization is not given. The divident is set as 255.0")

            elif self.normalization[0] == 'minmax': # normalization image with minmax normalization method

                if len(self.normalization) == 1: # not any minimum or maximum value was given set them as 0 and 1 respectivelly.
                    print("WARNING : \n\tNeither of minimum or maximum value was given. They are set to 0 and 1, respectivelly!")
                    self.minimum = 0.0
                    self.maximum = 1.0

                elif len(self.normalization) == 3:
                    self.minimum = float(self.normalization[1])
                    self.maximum = float(self.normalization[2])

                else:
                    print("Unresolvable min-max normalization! [0-1] min-max normalization to be applied!")
                    self.normalization = 'minmax'
                    self.minimum = 0.0
                    self.maximum = 1.0

            elif self.normalization[0] == 'mean': # subtract the mean from images

                if 'whole' in self.normalization: # calculate mean value from training set
                    print("Calculating the mean from the training set...")
                    self.get_mean_and_deviation()
                    print('Done!')

                    # for back normalization
                    self.old_mean = float(self.mean)
                    self.std =None # standard deviation is not necessary

                elif 'single' in self.normalization: # means that each image is processed by subtracting its mean value
                    self.mean = None

                elif len(self.normalization) == 2: # mean value is not given, calculate from training set
                    self.mean = float(self.normalization[1])

                else:
                    print("WARNING! \n\tThe mean value is not provided! It is going to be calculated from the training set.")
                    print("Calculating the mean from the training set...")
                    self.get_mean_and_deviation()

                    # for back normalization
                    self.old_mean = float(self.mean)
                    self.std =None # standard deviation is not necessary
                    print('Done!')

        else:
            print("WARNING! \n\tThere is not any Normalization method. Images to be processed without normalization")
            self.normalization = None


        if hasattr(self, 'metrics'):

            if str == type(self.metrics) and self.metrics.upper() == 'ALL':
                self.metrics = utils.METRICS.copy()

            elif list == type(self.metrics) and 'ALL' in [x.upper() for x in self.metrics]:
                self.metrics = utils.METRICS.copy()

        else:
            self.metrics = ['PSNR', 'SSIM']

        if hasattr(self, "interp_compare"):
            if self.interp_compare is not None:

                tmp_str = self.interp_compare

                if str == type (self.interp_compare) and  'ALL' == self.interp_compare or \
                    list == type(self.interp_compare) and 'ALL' in [x.upper() for x in self.interp_compare]:

                    self.interp_compare = utils.METHODS.copy()

                    if 'SAME' in [x.upper() for x in tmp_str]:
                        self.interp_compare.append('same')

            if isinstance(self.interp_compare, str):
                self.interp_compare = [self.interp_compare]

        else:
            self.interp_compare = None

        if not hasattr(self, "decimation"):
            self.decimation = 'bicubic'

        if not hasattr(self, "interp_up"):
            self.interp_up = 'bicubic'


        if not hasattr(self, "shuffle"):
            self.shuffle = True

        if not hasattr(self, "normalizeground"):
            self.normalizeground = False

        if not hasattr(self, "noise"):
            self.noise = None

        if not hasattr(self, "normalizeback"):
            self.normalizeback = None

        if not hasattr(self, "layeroutputs"):
            self.layeroutputs = False

        if not hasattr(self, "layerweights"):
            self.layerweights = False

        if not hasattr(self, 'lactivation'):
            if hasattr(self, 'activation'):
                self.lactivation = self.activation
            else:
                self.lactivation = None
                print("The activation is not defined! It is set to None")


        if args_has_build_function:
            self.model = self.build_model(testmode=True)


    def single_to_3channel(self, img, file_name):
        """
        Converts a single channel image to 3-channel image
        :param img:
        :param file_name: The name of image file.
        :return: 3-channel image.
        """

        tmp = np.zeros((img.shape[0], img.shape[1], 3))
        for i in range(3):
            tmp[:, :, i] = img[:, :]  # make single channel image to 3 channel image

        print("WARNING! ", file_name, "\n\t\t  is a single-channel image. It is converted ",
              "to 3-channel image copying along the third axis.")

        return tmp


    def test(self):
        """
        Tests the model with given test image(s) over given weight file(s). Takes single or multiple
        image(s) and weight file(s).The paths of multiple images or multiple weight files must be a folder path.

        Returns the output image, psnr and ssim measures of given image. The output image is the last
        image processed by the model with the last weight file, in case the 'testpath' parameter
        is a folder path.

        :param testpath: Path to the test image or the folder contains the image(s)
        :param weightpath:  Path to the weight file (.h5 file) or to the folder contains the weight file(s)
        :param saveimages: Boolean. If True, the images of ground truth, bicubic and the result of the model are saved.
        :param plot: Boolean. Used to determine whether layer weights and/or layer outputs to be plotted in graphically
                             while performing test. Default is False.
        :return: Output image, PSNR measure, SSIM measure
        """

        print("\n[START TEST]")

        self.model = self.build_model(testmode=True)
        if not self.model_exist():  # check if model exists
            print('Model does not exist! Terminating the Test procedure!')
            return None, None, None

        testpath = self.testpath
        weightpath = self.weightpath
        saveimages = self.saveimages

        # if 'weightpath' is not given during method call, take it from the class
        if weightpath == None or weightpath == "":
            weightpath = weightfolder = self.outputdir

        # if weightpath is a path to a weight file (.h5 file) 'weightfolder'
        # is set to folder path of weight file
        elif not isdir(weightpath):
            weightfolder = abspath(dirname(weightpath))

        else:
            weightfolder = weightpath

        text_scale = str(self.scale)

        if not isinstance(testpath, (list, tuple)):  # make it list or tuple
            testpath = [testpath]

        if saveimages:
            output_images_dir = join(self.outputdir, 'images')
            if not exists(output_images_dir):
                makedirs(output_images_dir)

        for path in testpath:

            weights_list = {}  # keeps the list of weights for each scale
            print(path)

            test_name = basename(path)

            if isdir(path):
                # get test images from the directory
                test_files = utils.get_files(path, image_extentions, remove_extension=False, full_path=True)

            else:  # path is a list of images
                test_files = list([path])

            test_file_names = [splitext(basename(x))[0] for x in test_files]

            if isdir(weightpath):
                weights_list = utils.get_files(weightfolder, ["*.h5"], remove_extension=False, full_path=True)

            else:
                weights_list = [weightpath]  # weightpath is a file. Should be in a list for iteration.

            if len(weights_list) == 0: # there is no any weight file
                print('Any weight file could not be found in the following path:\n', weightpath, '\nTerminating...')
                return None

            if  self.interp_compare is None or self.interp_compare == '':
                columns = list()

            else:
                columns = self.interp_compare.copy()
                if 'same' in columns:
                    columns.remove('same')

            for w in weights_list:
                columns.append(splitext(basename(w))[0])

            data_columns = columns.copy()

            columns = pd.MultiIndex.from_product([data_columns, self.metrics])
            dataset = pd.DataFrame(index=test_file_names, columns=columns)

            # for each test images
            for i in tqdm(range(len(test_files))):
                f = test_files[i]

                file_short_name = splitext(basename(f))[0]
                #print(f)
                satir = test_file_names[i]

                # _im = self.read_image(f, colormode=self.target_cmode)
                _im = self.read_image_by_PIL(f, colormode=self.colormode)

                # if channel is set 3 and the input image is a single channel image,
                # image to be converted to 3 channel image copying itself along the third
                # axis
                if len(_im.shape) < 3 and self.channels == 3:
                    _im = self.single_to_3channel(_im, f)

                # prepare ground image
                img_ground = utils.preprocess_image(_im, scale=self.scale,
                                                    pad=self.crop, channels=self.channels, upscale=False,
                                                    crop_remains=True, img_type='ground')


                if saveimages:
                    fileName = join(output_images_dir, file_short_name)
                    self.save_image(img_ground, fileName +  '_scale_' + text_scale + '_ground.png')

                # pad is set zero since input image should not be cut from borders. But, Ground truth and bicubic
                # images are cut from borders as muuch as pad size in case the model reduces the size of the output
                # image as much as pad. Some models work like this to avoid border effect in calculation of metrics.
                # If the parameter of padding is set to 'valid' in layers of models in KERAS is set as , output size
                # of images are decreases as floor integer value of kernel's half width at each layers. Therefore,
                # to have the same size of ground truth, bicubic and output images, bicubic and ground truth images
                # are cropped from borders as much as padding value.
                #
                # The padding value is set to 'same' in keras layers as to have the same size of output and input
                # input images. If this is the case, padding is not applied in here since they are of the same size
                #


                # if floatmode is True, image to be processed before converting it to float32
                # in misc.resize method. this is necessary for some situations, e.g.,
                # for data range is beyond the range 0-255.
                if self.floatmode:
                    mode= 'F'
                else:
                    mode= None

                img_input = utils.preprocess_image(_im, scale=self.scale, pad=0, decimation=self.decimation,
                                                   channels=self.channels, upscale=self.upscaleimage,
                                                   crop_remains=True, img_type='input', noise=self.noise,
                                                   interp_up=self.interp_up, mode=mode)

                if hasattr(self, 'rebuild') and self.rebuild:
                    self.img_height = img_input.shape[0]
                    self.img_width = img_input.shape[1]

                    self.model = self.build_model(testmode=True)
                    if not self.model_exist():  # check if model exists
                        print('Model does not exist! Terminating the Test procedure!')
                        return None, None, None

                if self.normalization:
                    img_input = self.normalize_image(img_input, self.normalization)

                img_input = utils.prepare_image_for_model(img_input, self.channels, image_data_format())

                # if any interpolation method is defined, do the test for
                # them
                if self.interp_compare is not None and (len(self.interp_compare) > 0 and '' not in self.interp_compare):

                    for method in self.interp_compare:
                        # if interpolation method for upscaling (interp_compare) is
                        # set as 'same', then low resolution image upscaled by the
                        # same interpolation method as decimation.

                        if method == 'same':
                            continue

                        if 'same' in self.interp_compare:
                            im_m = utils.preprocess_image(_im,
                                                          scale=self.scale, pad=self.crop, channels=self.channels,
                                                          decimation=method, interp_up=method, noise=self.noise,
                                                          upscale=True, crop_remains=True, img_type='bicubic')
                        else:
                            im_m = utils.preprocess_image(_im, scale=self.scale, pad=self.crop,
                                                          decimation=self.decimation, interp_up=method,
                                                          channels=self.channels, upscale=True, crop_remains=True,
                                                          img_type='bicubic',noise=self.noise)

                        res = utils.calc_multi_metrics_of_image(img_ground, im_m,
                                                                border=self.crop_test, channels=self.target_channels, metrics=self.metrics)

                        if saveimages:
                             self.save_image(im_m, fileName + '_scale_'+ text_scale + '_' + method + '.png')

                        for key, value in res.items():
                            dataset.loc[file_short_name, (method, key)] = value

                # do test for each weight file(s)
                for j in range(0, len(weights_list)):
                    sutun = splitext(basename(weights_list[j]))[0]

                    # Prediction
                    img_result = self.predict(img_input, weights_list[j],
                                              self.normalizeback)

                    res = utils.calc_multi_metrics_of_image(img_ground, img_result,
                                                            border=self.crop_test, channels=self.target_channels, metrics=self.metrics)

                    for key, value in res.items():
                        dataset.loc[file_short_name, (sutun, key)] = value

                    if saveimages:
                        self.save_image(img_result, fileName + '_scale_' + text_scale +'_Result_' + sutun + '.png')


                    if self.layeroutputs:
                        name = join(fileName, 'layer_outputs')

                        if not exists(name):
                            makedirs(name)
                        name = join(name, 'scale_') + text_scale + '_' + \
                                self.modelname + '_' + sutun

                        self.plot_all_layer_outputs(img_input, name, saveimages=self.saveimages, plot=False)

                    if self.layerweights :
                        name = join(fileName, 'layer_weights')

                        if not exists(name):
                            makedirs(name)

                        name = join(name, 'scale_') + text_scale + \
                               self.modelname + '_' + sutun

                        self.plot_layer_weights(saveimages=self.saveimages,name=name, plot=False, dpi=300)

            # write results in an excel file in weight folder
            excel_file = self.modelname + '_' + test_name + "_results_scale_" + text_scale + ".xlsx"

            dataset.sort_index(axis=1, inplace=True)  # sort columns alphabetically
            dataset.sort_index(axis=0, inplace=True)  # sort rows alphabetically

            # utils.write_to_excel(datasets, excel_file)
            dataset.loc["Mean"] = dataset.mean()
            utils.measures_to_excel(dataset, self.outputdir, excel_file)

            print("[TEST FINISHED]")

        return dataset


    def to_psnr(self, mse, max_value= 255.0):
        """
            Calculates PSNR value from given MSE value.
            :param mse:
            :param max_value:
            :return:
            """

        if mse is None or mse == float('Inf') or mse == 0:
            psnr = 0
        else:
            if isinstance(mse, list):
                psnr = [10 * np.log10( max_value**2 / x ) for x in mse ]
            else:
                psnr = 10 * np.log10( max_value**2 / mse)

        return psnr


    def train_with_h5file(self, weightpath = None, plot=False):
        """
        Training procedure with images in a .h5 file.
        :param weightpath: Path to model weight.
        :param plot:
        :return:
        """

        self.model = self.build_model(testmode=False)
        if not exists(self.datafile): # check if training data exists
            self.prepare_dataset(self.traindir, self.datafile)

        h5f = h5py.File(self.datafile, 'r')
        X = h5f['input']
        y = h5f['label']

        # load weight
        if  weightpath != None and weightpath !="":

            if isdir(weightpath):
                print("Given weight file path is not a file path. Model will run without loading weight")

            else:
                self.model.load_weights(weightpath)

        print(self.model.summary())

        print("Training starting...")
        start_time = time.time()

        self.history = self.model.fit(X, y, validation_split=0.1, batch_size=self.batchsize,
                                      epochs=self.epoch, verbose=0, shuffle=self.shuffle,
                                      callbacks=self.prepare_callbacks())
        self.elapsed_time = time.time() - start_time

        self.write_summary()

        print("Training has taken %.3f seconds." % (self.elapsed_time))


    def train_on_batch(self, weihgtpath=None, plot=False):
        """
        Training procedure with batches. To be implemented in future.
        :param weihgtpath:
        :param plot:
        :return:
        """
        pass


    def train(self):
        """
        Training procedure of the method.
        :param weightpath: Path to the weight file to load the model with.
        :param plot:
        :return:
        """

        print("\n[START TRAINING]")

        self.model = self.build_model(testmode=False)

        weightpath = self.weightpath

        # load weight
        if  weightpath != None and weightpath !="":

            if isdir(weightpath):
                print("Given weight file path is not valid. Model will run without loading weight")

            else:
                self.model.load_weights(weightpath)

        print('Calculating the number of training samples...', end="", flush=True),
        train_samples = self.get_number_of_samples(self.traindir)
        print('Done!')
        print('Totally {0} of training samples to be extracted from training set.'.format(train_samples))
        self.n_training_samples = train_samples

        self.nb_batches = int(train_samples // self.batchsize)

        # if there is no any validation image(s), train the model without validation images,
        # otherwise, if exist any, train model with the validation image(s)
        if self.valdir is None or self.valdir == "" or self.valdir==" ":
            valData= None
            val_steps = None

        else:
            print('Calculating the number of validation samples...', end="", flush=True)
            val_samples = self.get_number_of_samples(self.valdir)
            print('Done!')
            print('Totally {0} of validation samples to be extracted from training set.'.format(val_samples))

            print(f'Totally {self.nb_batches} batch updates to be calculated for training.')

            self.n_validation_samples = val_samples

            val_steps = int(val_samples // self.batchsize)
            valData = self.generate_batch_on_fly(self.valdir, shuffle=self.shuffle)


        file_write = open(join(self.outputdir, "training samples.txt"), "w")
        file_write.writelines(f"Training samples: {train_samples}\n")
        file_write.writelines(f"Training updates: {self.nb_batches}\n")

        if valData is not None:
            file_write.writelines(f"Validation samples: {val_samples}\n")
            file_write.writelines(f"Validation updates: {val_steps}\n")

        file_write.close()

        print(self.model.summary())

        print("Batch generator starting...")
        start_time = time.time()

        self.history = self.model.fit_generator(
                                 self.generate_batch_on_fly(self.traindir, shuffle=self.shuffle),
                validation_data= valData,
                epochs =self.epoch, workers=1,max_queue_size=1, callbacks = self.prepare_callbacks(),
                steps_per_epoch=self.nb_batches, validation_steps = val_steps, verbose=2 )

        self.elapsed_time = time.time() - start_time

        self.write_summary()

        self.plot_history()
        # self.plot_loss_history()


        print("Training has taken %.3f seconds." % (self.elapsed_time))
        print("[TRAIN FINISHED]")

        self.save_batch_losses()


    def train_with_fit_generator(self, weightpath = None, plot=False):
        """
        Training procedure with Keras Generator.
        :param weightpath: Path to the weight file to load the model.
        :param plot:
        :return:
        """

        self.n_training_samples = None
        self.n_validation_samples = None

        self.model = self.build_model(testmode=False)
        if not exists(self.datafile): # check if training data exists
            print("\nTRAIN DATASET")
            self.prepare_dataset(self.traindir, self.datafile)

        f = h5py.File(self.datafile, "r")
        row_number = f['input'].shape[0]
        f.close()
        self.nb_batches = int(row_number // self.batchsize)

        # if there is no any validation image(s), train the model without validation images,
        # otherwise, if exist any, train model with the validation image(s)
        if (self.valdir is None or self.valdir == '' or self.valdir==" "):
            valData= None
            val_steps= None

        elif not exist(self.validationfile) :

            print("\nVALIDATION DATASET")
            self.prepare_dataset(self.valdir, self.validationfile)

            f = h5py.File(self.validationfile, "r")
            row_number = f['input'].shape[0]
            f.close()
            val_steps = int(row_number // self.batchsize)
            valData = self.generate_from_hdf5_without_shuffle(self.validationfile, self.batchsize)

        # load weight
        if  weightpath != None and weightpath !="":

            if isdir(weightpath):
                print("Given weight file path is not a file path. Model will run without loading weight")

            else:
                self.model.load_weights(weightpath)

        #print(self.model.summary())

        print("Training on fit generator starting...")
        start_time = time.time()

        self.history = self.model.fit_generator(
                self.generate_from_hdf5_without_shuffle(self.datafile, self.batchsize), validation_data= valData,
                epochs =self.epoch, workers=1,max_queue_size=128, callbacks = self.prepare_callbacks(),
                steps_per_epoch=self.nb_batches, validation_steps = val_steps, verbose=2 )

        self.elapsed_time = time.time() - start_time

        self.write_summary()


        print("Training has taken %.4f seconds." % (self.elapsed_time))


    def write_summary(self, ):

        # Summary of the model
        with open(self.outputdir + '/training summary.txt', 'w') as fh:

            # Pass the file handle in as a lambda function to make it callable
            self.model.summary(print_fn=lambda x: fh.write(x + '\n'))
            fh.write("Training took: %.8f hour(s)" % (self.elapsed_time / 3600))


    def YCbCr_array_to_RGB(self, im):
        """
        Converts an image with YCbCr color space given in numpy array format to a PIL image
        in RGB color space
        :param im: Image in numpy ndarray format to be transofrmed.
        :return: PIL image in RGB color space
        """
        img = Image.fromarray(im)
        img.mode = 'YCbCr'
        img = img.convert('RGB')

        return img


def start():
    sr = DeepSR(ARGS)

    if hasattr(sr, 'model') and hasattr(sr, 'plotmodel') and sr.plotmodel ==True:
        sr.plot_model()

    if ARGS['train_with_generator'] is not None and ARGS['train_with_generator']:
        print("[Train With Generator]")
        if sr.weightpath == None or sr.weightpath=="":
            sr.train_with_fit_generator()
        else:
            sr.train_with_fit_generator(sr.weightpath)

    if ARGS['train'] is not None and ARGS['train']:
        sr.mode = 'train'
        sr.train()

    if ARGS['test'] is not None and ARGS['test']:
        sr.test()

    if ARGS['predict'] is not None and ARGS['predict']:
        print("[Predict Mode]")
        sr.predict(sr.testpath, sr.weightpath)

    if ARGS['repeat_test'] is not None and ARGS['repeat_test']:
        sr.repeat_test(ARGS['repeat_test'])

    if ARGS['plotimage'] is not None and ARGS['plotimage']:
        sr.plotimage(sr.testpath, sr.weightpath, True)

    if ARGS['shutdown'] is not None and ARGS['shutdown']:
        from os import system
        time.sleep(60) # wait for a minute so that computer finalizes processes.
        system('shutdown /' + ARGS['shutdown'])

    return sr

if __name__ == "__main__":
    start()

