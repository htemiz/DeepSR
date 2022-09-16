"""
args.py #

This is the argument file in which argument lists are defined
for the of DeepSR object (DeepSR.py)


Developed  by
        Hakan Temiz             htemiz@artvin.edu.tr

Version : 0.0.51
History :

"""

import argparse


parser = argparse.ArgumentParser(description="A Python framework for Image Super Resolution.")
parser.add_argument("--activation", type=str, choices=["relu", "lrelu", "LeakyReLU", "prelu", "tanh", "sigmoid", "elu", "selu", "elu", "exponential", "",], default='relu',
                    help="String. Activation function for layers.Default is 'relu'.")

parser.add_argument('--augment',  nargs='*',
                    help="String or List of strings.A list of translation operation of image for image data augmenting \
                    (90, 180,270, 'flipud', 'fliplr', 'flipudlr')")

parser.add_argument("--backend", type=str, choices=["theano", "tensorflow"], default='tensorflow',
                    help="String. Determines Keras backend: 'theano' or 'tensorflow'. Defaults is 'tensorflow'.")

parser.add_argument("--batchsize", type=int, default=256,
                    help="Integer. Batch size for model training. Default is 256.")

parser.add_argument("--channels", type=int, choices=[1,3], default=3,
                    help="Integer. Number of color channels (either, 1 or 3) used by model. Default is 3.")

parser.add_argument("--colormode", type=str, choices=["RGB", "YCbCr"], default= 'RGB',
                    help="String. Color space (RGB or YCbCr) in which the model processes the image. Default is RGB.")

parser.add_argument("--crop", type=int, default=0,
                    help="Integer. Cropping size from Image. It is valid for training phase. Ground truth and/or \
                    interpolated images to be cropped by crop size from each borders. Some models \
                    produce images with less size than the one of input images, since they use padding \
                    value as 'valid' to avoid border effects. Therefore, interpolated and/or Ground Truth \
                    images should be cropped to make their size be the same as the size of output image of model. \
                    Default is 0.")

parser.add_argument("--crop_test", type=int, default=0,
                    help="Integer. Number pixels to be cropped from the border of images in test procedure. \
                    Same as the paramater\'crop\', except that it is used for test phase. \
                    Images (output image as well) to be cropped by crop size from each borders. \
                    Some models produce images with less size than the size of input, since they use \
                    padding value as 'valid' to avoid border effects. Therefore, interpolated and/or \
                    Ground Truth images should be cropped to make their size be the same as the size of \
                    output image of model. Default is 0.")

parser.add_argument("--decay", type=float, default=1e-6,
                    help="Float. Decay value for weights. Default is 1e-6.")

parser.add_argument("--decimation", type=str, choices=["bilinear", "bicubic", 'nearest', 'lanczos'], default='bicubic',
                    help="String. The Interpolation method used in decimating the image to get \
                    low resolution (downscaled) image). Images to be decimated  and upscaled with this method. \
                    Default is 'bicubic'.")

parser.add_argument("--drate", type=int, nargs="+", default=(1,1),
                    help="A list of dilation rate values for layers. Default is (1,1).")

parser.add_argument("--espatience", type=int, default=5,
                    help="Integer. Early stopping patience. The number of training epochs to wait before early stop if the model does not progress \
                     on validation data. this parameter helps avoiding overfitting when training the model. \
                     Default is 5.")

parser.add_argument("--epoch", type=int, default=1,
                    help="Integer. Number of Epoch(s) for training model. Default is 1.")


parser.add_argument('--floatmode', action='store_true', default=False,
                    help='Boolean. Input image is handled as float type for downscaling and/or upscaling \
                         by pre-processing method. Default is False.')


parser.add_argument("--gpu", type=str,
                    help="String. The number of GPUs to be used. '0' ... '3' to use 1. ... 4. GPU, or 'all' to use all GPUs.\
                     Multiple GPUs also can be designated with separating by comma. \
                     E.g. --gpu '0,1' for using GPUs 0 and 1 (0 and 1 indicates the first and second GPUS since it is zero-based indexed)")

parser.add_argument("--inputsize", type=int, default=30,
                    help="Integer. The size of input image patch in training mode. Default is 30.")

parser.add_argument("--interp_compare", type=str, default='', nargs='*',
                    help="String or List of strings. The interpolation method(s) to which the performance of the model is compared. \
                    the interpolation method(s) (bilinear, bicubic, nearest, lanczos) determined with this argument. \
                    If it is None, the model will not be compared with any interpolation methods. Use \'ALL\' for \
                    comparison the performance of the model with all interpolation methods. Use the keyword \'same\' to \
                    use the same interpolation method(s) in decimating down the image (overrides the method given in \
                    the argument \‘decimation\’) and upscaling back. Default is None.")

parser.add_argument("--interp_up", type=str, choices=["bilinear", "bicubic", 'nearest', 'lanczos'], default='bicubic',
                    help="String. The Interpolation method used in upscaling low resolution images.Default is 'bicubic'.")

parser.add_argument("--kernel_initializer", type=str, default="glorot_uniform",
                    help="Initializer function for layer kernels. Default is 'glorot_uniform'.")

parser.add_argument("--lactivation", type=str, choices=["relu", "lrelu", "LeakyReLU", "prelu", "tanh", "sigmoid", "elu", "selu", "elu", "exponential", ""],
                    help="String. Activation function for the Last layer of model.Default is the same as --activation argument.")

parser.add_argument("--layeroutputs",  action="store_true", default=False,
                    help="Boolean. Layer's outputs to be visualized and/or saved as images while performing test, \
                    if it is given (or set to True in DeepSR object). Default is False.")

parser.add_argument("--layerweights",  action="store_true", default=False,
                    help="Boolean. Layer weights to be visualized and/or saved as images, if it is given \
                    (or set to True in DeepSR object). Default is False.")

parser.add_argument("--lrate", type=float, default=0.001,
                    help="Float. Learning rate for weight updates. Default is 0.001")

parser.add_argument("--lrpatience", type=int,
                    help="Integer. Learning rate changing patience. The number of training epochs to wait to change learning rate \
                         if the model does not progress on validation data. this parameter helps avoiding overfitting when training the model. \
                     Default is 5.")

parser.add_argument("--lrfactor", type=float, default=0.5,
                    help="Float. Value for changing the learning rate value. Default is 0.05")

parser.add_argument('--metrics', default= ['PSNR', 'SSIM'],  nargs='*',
                    help="String or List of strings. A list of image quality assessment (IQA) metrics  , i.e., \
                    'PSNR, SSIM, FSIM, MSE, MAD, NRMSE'. OR type 'ALL' \
                    to measure entire metrics defined in the list 'METRICS' in 'utils.py'.\
                    Please refer to variable METRICS in utils.py for entire metrics. Default is ['PSNR', 'SSIM']")

parser.add_argument("--modelfile", type=str, default='',
                   help="String. Path to the model file (a .py file). Model file must have a method named construct_model \
                   returning the model object, as well as a dictionary, named settings, which has settings in key/value pairs")

parser.add_argument("--modelname", type=str, default='',
                    help="String. The name of model.")

parser.add_argument("--noise", nargs='*', default='',
                    help="List of strings. The Gaussian noise to be added to images. The \'mean\' and \'variance\' values must be \
                    provided with this parameter. \
                    \nExample: \n\t--noise 0.0 0.1 \n\tThe first value (0.0) is the mean and the second value (0.1) \
                    is the variance for Gaussian noise.")

parser.add_argument('--normalization',  nargs='*',
                    help="String or List of strings. Defines what normalization method is applied to input images of models \
                     (divide, minmax, standard, mean). \
                    \
                        \ndivide, means that each image is divided by the given value. \
                            \n\tFor example: \"--normalization divide 127\" means that, each image is divided by 127. \
                            for the purpose of normalization. Default is 255.0 \
                    \
                        \n\nminmax means that image is arranged between minimum and maximum values provided by user. \
                            The defauls values of minimum and maximum are 0.0 and 1.0 respectively if they are not given.\n\n \
                            \nFor example, \'--normalization minmax -1 1\' means that each images are processed such that \
                            the minimum and maximum values of image would be -1 and 1, respectively. \
                    \
                        \n\nstandard means that image standardized with zero mean and standard deviation of 1. \
                            \nIt should be written like this: --normalization standard \"1 2 3\" \"4 5 6\" , for mean and std values, \
                            respectivelly, if mean and std values are given. To calculate mean and std values from training set, \
                            do not provide values. Instead, along with the key \"std\", provide the key \"whole\" for calculation of \
                            both values from training set, or provide the key \"single\" to process each image with its mean and std values.\
                            \n\tFor example: \"--normalization standard whole\" calculates the mean and std values from whole training set.\
                    \
                        \n\nmean, subtracts the mean of image from image. Similar to the standard method, if mean value(s for each channel) \
                            is given, each image is processed by subtracting this (those) mean value(s of each channels) from images. \
                            If the mean value(s) is not given, the mean value(s) is calculated from training set, than. \
                             \nThe prefix \'single\' indicates that the mean value is calculated from images individually as \n \
                             they are being processed at training stage. Similarly, \'whole\' key is used for calculating the mean\n \
                             value from entire training images." )

parser.add_argument('--normalizeback', action='store_true', default=False,
                    help='Boolean. The produced image by model to be normalized back with relevant values, \
                    if it is given (or set to True in DeepSR object). Default is False.')

parser.add_argument('--normalizeground', action='store_true', default=False,
                    help='Boolean. Normalization is applied to ground truth images, if it is given \
                     (or set to True in DeepSR object). Default is False.')

parser.add_argument("--outputdir", type=str, default='',
                    help="String. Path to output folder")

parser.add_argument("--plot", action="store_true", default=False,
                    help="Boolean. Determines whether layer weight and/or layer outputs to be visualized graphically \
                     while performing test, if it is given (or set to True in DeepSR object). Default is False.")

parser.add_argument('--plotimage', action="store_true", default=False,
                    help ='Command to get predicted output image of model, visualize and/or save as image file. \
                    This procedure can also be performed by calling DeepSR object\'s member method with the same name as this argument.')

parser.add_argument("--plotmodel",  action="store_true", default=False,
                    help="Boolean. The layout of models to be visualized and saved as image files if it is given \
                    (or set to True in DeepSR object). Default is False.")

parser.add_argument("--predict", action = "store_true", default=False,
                    help='Command to get the prediction scores of models for a given image with given model weights.\
                    This procedure can also be performed by calling DeepSR object\'s member method with the same name as this argument.')

parser.add_argument('--rebuild', action='store_true', default=False,
                    help='Boolean. If its True, model is being rebuilt for each test file, \
                         since some models need to be fed with images’s width and height \
                          Otherwise, model is built only once at the beginning of Test. Default is False.')

parser.add_argument("--saveimages",  action="store_true", default=False,
                    help="Boolean. Determines whether images being saved or not while testing. \
                    Used with \"test\" argument. if it is given (or set to True in DeepSR object), images to be saved. \
                    Default is False.")

parser.add_argument("--scale", type=int, default=2,
                    help="Integer. The magnification factor. Default is 2.")

parser.add_argument('--seed', type=int, default=19, help='Integer. Seed number of number generators to be used for \
                    re-producable deterministic models.')

parser.add_argument("--shuffle",  action="store_true", default=True,
                    help="Boolean. Input images or images patches to be shuffled  if it is given \
                    (or set to True in DeepSR object), in order to ensure randomness. Default is True.")

parser.add_argument('--shutdown', action="store_true", default=False,
                    help='Boolean. Computer will be shut down after all processes have been done  if it is given \
                     (or set to True in DeepSR object). Default is False.')

parser.add_argument("--stride", type=int, default=10,
                    help="Integer. The stride for generating input images (patches) for training of model.  \
                    It is the same for both directions. Default is 10.")

parser.add_argument('--repeat_test', type=int,
                    help ='Integer. To be used to do tests for one epoch at a time for the given number of epochs in total. \
                    To be implemented in future versions.')

parser.add_argument("--target_channels", type=int, choices=[1,3], default=3,
                    help="Integer. Determines what number of color channels used for test. It should be 1 or \
                    3. If it is 1, only first channels of both images compared in tests. Default is 3.")

parser.add_argument("--target_cmode", type=str, default="", choices=["YCbCr", "RGB"],
                    help="String. Target color mode (RGB or YCbCr) for testing, saving and/or plotting of images. \
                    It will be the same as the value of the argument \'colormode\', if it is not given.")

parser.add_argument("--test",  action="store_true", default=False,
                    help="Command to perform test. Test procedure to be performed if it\
                    is given in command prompt. Test procedure can also be performed by calling the member method \
                    with the same name of DeepSR object.")

parser.add_argument("--testpath", nargs='*', default='',
                    help="String. Path to the test image file or the directory where test image(s) in")

parser.add_argument("--train", action="store_true", default=False,
                    help="Command to train model. Training procedure can also be \
                    performed by calling DeepSR object's member method with the same name as this argument.")

parser.add_argument("--traindir", type=str, default='',
                    help="String. The directory path where the training images in.")

parser.add_argument("--train_with_generator",  action="store_true", default=False,
                    help="Command to run training with generator for huge data sizes, after class is built. \
                    to be implemented in future versions.")

parser.add_argument("--upscaleimage", default=False,
                    help="Boolean. Indicates whether the input image is upscaled by the given interpolation method \
                    before giving it to the model.\
                    Some models have input image already upscaled with an interpolation method, while others\
                    upscales images from downscaled low resolution images by themselves. If model will take already \
                    upscaled image, this parameter should be given. Default is False. So, images will not be upscaled.\
                    Namely models will upscale input images by themselves.")

parser.add_argument("--valdir", type=str, default='',
                    help="String. The directory path in which the test images is.")

parser.add_argument("--weightpath", type=str, default='',
                    help="String. Path to the weight file or to the directory where weight file(s) in")

parser.add_argument("--workingdir", type=str, default='',
                    help="String. Path to working folder")


def getArgs():

    return vars(parser.parse_args())


"""   
if __name__ == '__main__':
    
    args = parser.parse_args()
"""