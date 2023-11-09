"""
utils.py #

A python program with several utilities for the task of Super Resolution,
and for image processing.

Developed  by
        Hakan Temiz             htemiz@artvin.edu.tr

    version : 0.0.81
    history :

"""

import h5py
import numpy as np
# import openpyxl
from scipy import misc
from skimage.transform import rotate
from scipy.ndimage.filters import gaussian_filter
import pandas as pd
from os import listdir, makedirs, remove
from os.path import isfile, join, exists, splitext, abspath, basename
import glob
import math
from matplotlib import  pyplot as plt
from PIL import Image
from skimage.util import random_noise
from skimage import img_as_float32, img_as_uint
from skimage.io import imread, imsave, imshow
# from PIL.Image import resize
from tqdm import tqdm

# measures
from sporco.metric import snr, bsnr, pamse, gmsd, isnr
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity as SSIM
from skimage.metrics import mean_squared_error as compare_mse, normalized_root_mse as compare_nrmse
import sewar
from skvideo.measure import  mad, niqe #, brisque , msssim, strred

"""
#from metrics.niqe import niqe
from metrics.reco import reco
#from metrics.vifp import vifp_mscale
from metrics.pymetrikz.metrikz import nqm, wsnr # pbvif, mssim, uqi

#import brisque as brsq
#import matlab.engine
from metrics.xdesign_metrics import compute_fsim
from metrics.haarPsi import  haar_psi
"""

from keras.callbacks import  Callback
from keras import backend as K

from scipy.io import loadmat
from scipy.signal import hilbert, hilbert2

image_extentions = ["*.bmp", "*.BMP", "*.jpg", "*.JPG", "*.png", "*.PNG", "*.jpeg", "*.JPEG", "*.TIFF", "*.tiff", '*.mat']


METHODS = ['bilinear', 'nearest', 'bicubic', 'lanczos']

METRICS = [  'BSNR', 'BRISQUE',  'ERGAS', 'GMSD', 'MAD', 'MSE', 'MSSSIM', 'NIQE', 'NRMSE',
             'PAMSE',  'PSNR', 'RASE', 'SAM', 'SCC', 'SNR', 'SSIM',  'UQI', 'VIF' ]
          # 'NQM',  'HAARPSI', 'FSIM',    'WSNR', ]


log10 = K.log(10.)

eps = np.finfo(float).eps

def load_mat_file(file):

    pic = loadmat(file)
    keys = pic.keys()

    if 'rf' in keys:
        rf = pic['rf']

    elif 'signals_matrix' in keys:
        rf = pic['signals_matrix']

    else:
        print('Unknown rf variable name please check the file ', file)
        return None
    return rf



def RF_to_Bmode(rf):
    hlb = hilbert(rf)

    return 20 * np.log10(np.abs(hlb))



def get_files(path, extentions, remove_extension=True, full_path = False):
    """
    Returns a list of files with particular extensions in a path.
    :param path:
    :param extentions:
    :param remove_extension:
    :param full_path:
    :return:
    """
    
    files = list()

    for ext in extentions:
        files= files + glob.glob(join(path, ext))

    files = list(dict.fromkeys(files))  # remove duplicates

    if not full_path:
        files = [basename(x) for x in files]

    if remove_extension:
        files = [splitext(x)[0] for x in files]

    return files


def write_summary(path, model, elapsed_time, n_training=None, n_validation=None, save_model=False):
    """

    :param path:
    :param model:
    :param elapsed_time:
    :return:
    """

    if save_model:
        model.save(path + '/final_model.h5', overwrite=True, include_optimizer =True)

    # Summary of the model
    with open(path + '/training summary.txt','w') as fh:

        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
        fh.write("Training took: %.4f hour(s)" % (elapsed_time / 3600))

        if n_training is not None:
            fh.write("Totally {0] image patches extracted from dataset for training.".format(n_training))
        if n_validation is not None:
            fh.write("Totally {0] image patches extracted from dataset for validation.".format(n_validation))


class History(Callback):
    """
    History of Keras model.
    """
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

    def on_train_end(selfself, logs={}):

        with h5py.File("results.h5", 'w') as f:
            f.create_dataset("loss", data= np.asarray(logs.get('loss'), dtype='float64'))


def get_psnr(mse, max_value=255.0):
    """
    Calculates PSNR value from given MSE value.
    :param mse:
    :param max_value:
    :return:
    """
    if mse is None or mse == float('Inf') or mse == 0:
        psnr = 0
    else:
        psnr = 20 * math.log(max_value / math.sqrt(mse), 10)
    return psnr


def compute_mse(image1, image2, border_size=0):
    """
    Calculates the Mean Squared Error (MSE) value from two images.
    :param image1:
    :param image2:
    :param border_size:
    :return:
    """
    if len(image1.shape) == 2:
        image1 = image1.reshape(image1.shape[0], image1.shape[1], 1)
    if len(image2.shape) == 2:
        image2 = image2.reshape(image2.shape[0], image2.shape[1], 1)

    if image1.shape[0] != image2.shape[0] or image1.shape[1] != image2.shape[1] or image1.shape[2] != image2.shape[2]:
        return None

    if image1.dtype == np.uint8:
        image1 = image1.astype(np.double)
    if image2.dtype == np.uint8:
        image2 = image2.astype(np.double)

    mse = 0.0
    for i in range(border_size, image1.shape[0] - border_size):
        for j in range(border_size, image1.shape[1] - border_size):
            for k in range(image1.shape[2]):
                error = image1[i, j, k] - image2[i, j, k]
                mse += error * error

    return mse / ((image1.shape[0] - 2 * border_size) * (image1.shape[1] - 2 * border_size) * image1.shape[2])


def MSELoss(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    loss = K.mean(K.square(y_pred - y_true))
    return loss


def PSNRLoss(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    MSE = K.mean(K.square(y_pred - y_true))

    #return -10. * (K.log(MSE) / log10)
    return 10 * (K.log(255.0 ** 2/MSE) / log10)


def prepare_folders(cur_folder, file_name, scale):
    result_folder = abspath(cur_folder + '/results')

    result_folder = abspath(result_folder + '/' + file_name + '/' + str(scale))

    # create the folder if not exists
    if not exists(result_folder):
        makedirs(result_folder)

    return result_folder


def measures_to_excel(dataset, output_dir, file_name ):
    """
    Writes the pandas dataset of measure results to a Excel file, as well as csv file with
    mean of results.

    :param dataset: pandas dataset of the results
    :param output_dir: folder path to output directory
    :param file_name: output file name
    :return:

    :type dataset: DataFrame
    :type output_dir: str
    :type file_name: str

    """
    dataset.sort_index(axis=1, inplace=True)

    writer = pd.ExcelWriter(join(output_dir, file_name), engine='openpyxl')
    dataset.transpose().to_excel(writer)
    dataset.stack().loc['Mean',:].to_excel(writer, 'Mean')
    # dataset.stack().loc['Mean',:].to_csv(join(output_dir, file_name +'.txt'))
    writer.save()
    writer.close()


def measure_images(image_path, scale=2, metrics= "ALL", channels= 1, decimation="bicubic",
                   methods= METHODS, border=0, pad=0, save_images=False, mode=None):
    """
        Used for Image Quality Assessment (IQA) of images in given folder path. Ground truth images
        maintained from the images in the given path. The images to be evaluated generated using
        given interpolation methods, and then, IQA is maintained between ground truth images and
        result images.

    :param image_path: Path to the folder where images in.
    :param scale: factor of scaling images down and up
    :param metrics: metrics to measure image quality
    :param channels: number of channels to be processed
    :param decimation: Decimation method for scaling down the image.
        onf 'bilinear', 'nearest','bicubic', 'lanczos' or 'same' to use
        the same method as in the paramter "methods" accordingly.
    :param methods: interpolation methods to upscale the image
    :param border: pixel size to be cropped from image borders for measuring image quality
    :param pad: pixel size to be cropped from entire borders of image.
    :param save_images: Boolean. If True, processed images to be saved appropriately
    :return: Returns Pandas DataFrame of the result of image quality metrics

    """

    func = {'nearest': 0, 'lanczos': 1, 'bilinear': 2, 'bicubic': 3, 'cubic': 3}

    file_list = get_files(image_path, image_extentions, remove_extension=False, full_path=True)
    test_file_names = [splitext(basename(x))[0] for x in file_list]

    data_columns = METHODS
    columns = pd.MultiIndex.from_product([data_columns, METRICS])
    dataset = pd.DataFrame(index=test_file_names, columns=columns)

    if save_images:
        folder_name = join(image_path, 'result_images')
        if not exists(folder_name):
            makedirs(folder_name)
    for i in tqdm(range(len(file_list))):
    #for file in file_list:
        file = file_list[i]
        filename = splitext(basename(file))[0]

        img = imread(file, as_gray=False)

        im_g = preprocess_image(img, scale=scale, pad=pad, channels=channels,
                                upscale=False, crop_remains=True, img_type='ground', mode=mode)

        if save_images:
            out_file = join(folder_name, filename)
            Image.fromarray(im_g).save(out_file + '_ground.png')

        if decimation != "same":
            # image to be measured interpolated image that is downscaled and upscaled sequentially
            im_b = preprocess_image(im_g, scale=scale, pad=pad, channels=channels, mode=mode,
                                    interpolation=decimation, upscale=False, crop_remains=False, img_type='bicubic')

        for method in methods:

            if decimation == "same":
                # image is decimated with the same method as method, since the option "same" is chosen
                # for decimation algorithm.
                im_b = preprocess_image(im_g, scale=scale, pad=pad, channels=channels, mode=mode,
                                        interpolation=method, upscale=False, crop_remains=False, img_type='bicubic')

                img = Image.fromarray(im_b, mode=mode)

                org_shape = (img.size[0], img.size[1])
                new_shape = np.divide(org_shape, scale).astype(int)

                im_m = img.resize(new_shape, resample=func[method])
                im_m = np.array(im_m)

            res = calc_multi_metrics_of_image(im_g, im_m, border=border, channels=channels, metrics="ALL")

            if save_images:
                Image.fromarray(im_m).save(out_file + '_' + method + '.png')

            for key, value  in res.items():
                dataset.loc[filename, (method, key)] = value

    dataset.loc["Mean"] = dataset.mean()

    return dataset


def calc_multi_metrics_of_image(img_ground, image, border=0, channels=1,
        metrics=["PSNR", "SSIM"]):
    """
    Calculates any of following image quality indexes of an image from itself or from a reference ground truth image:

        BRISQUE, BSNR,CNR, ERGAS FSIM, GMSD,ISNR, MAD,
        MSE, MSSSIM, NIQE, NQM,NRMSE, PAMSE, PSNR, RASE,
        SAM, SCC SNR, SSIM, STRRED, UQI, VIF, WSNR

    Default is ["PSNR", "SSIM"]

    Images to be converted to uint8 space. Intensity values should not be normalized
    So, values must be between 0 and 255 inclusive

    :return: a dictionary with image quality measures

    :type img_ground: ndarray
    :type border: int
    :type channels: int
    :rtype: dict
    :param border: cropping size from borders of image
    :param channels: number of color channels of image to be processed
    :param img_ground: reference image
    :param metrics: a list of combinations of following image quality indexes;


        BRISQUE, BSNR,CNR, ERGAS FSIM, GMSD,ISNR, MAD,
        MSE, MSSSIM, NIQE, NQM,NRMSE, PAMSE ,PSNR, RASE,
        SAM, SCC SNR, SSIM, STRRED, UQI, VIF, WSNR

    :type metrics: list
    .. warnings also::
    .. note::  METRICS

        FROM skimage.measure
            PSNR, Peak Signal to Noise Ratio: implemented

            SSIM

            MSE

            NRMSE

        FROM skvideo.measure
            MAD

            BRISQUE : Blind/Referenceless Image Spatial QUality Evaluator (BRISQUE). Not Implemented since returns arrays instead of a single value

            NIQE    : Natural Image Quality Evaluator

            STRRED

        FROM SEWAR

            ERGAS, Erreur Relative Globale Adimensionnelle de Synthèse, https://hal.archives-ouvertes.fr/hal-00395027/

            SAM, Spectral Angle Mapper, https://ntrs.nasa.gov/search.jsp?R=19940012238

            RASE, Relative Average Spectral Error, https://ieeexplore.ieee.org/document/1304896/

            SCC, Spatial Correlation Coefficient, https://www.tandfonline.com/doi/abs/10.1080/014311698215973

            VIF (sometimes called VIF-P or VIFP), Visual Information Fidelity, https://ieeexplore.ieee.org/abstract/document/1576816/

            UQI     : universal image quality index, https://www.ingentaconnect.com/content/asprs/pers/2008/00000074/00000002/art00003

            MSSSIM  : Multiscale Structural Similarity (MS-SSIM) Index, https://ieeexplore.ieee.org/abstract/document/1292216/

        FROM metrics folder
            RECO, Relative Polar Edge Coherence: implemented

        FROM PYMETRIKZ (https://bitbucket.org/kuraiev/pymetrikz/src/default/)
            nqm     : noise quality measure
            wsnr    : weighted signal-to-noise ratio


        FROM SPORCO (https://sporco.readthedocs.io/en/v0.1.9/sporco.metric.html)
            snr     : signal to noise ratio

            isnr    : Improvement Signal to Noise Ratio

            bsnr    : Blurred Signel to Noise Ratio (BSNR)

            pamse   : Perceptual-fidelity Aware Mean Squared Error (PAMSE)IQA metric
                        (https://sporco.readthedocs.io/en/v0.1.9/zreferences.html#xue-2013-perceptual)

            gmsd    : Gradient Magnitude Similarity Deviation (GMSD) IQA metric
                        (https://sporco.readthedocs.io/en/v0.1.9/zreferences.html#xue-2014-gradient)

    .. seealso:: REFERENCES
    Structural Similarity Index (SSIM)
        Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli,
        "Image quality assessment: From error visibility to structural similarity"
        IEEE Transactions on Image Processing, vol. 13, no. 4, pp.600-612, Apr. 2004

    Multi-scale SSIM Index (MSSIM)
        Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli,
        "Image quality assessment: From error visibility to structural similarity"
        IEEE Transactions on Image Processing, vol. 13, no. 4, pp.600-612, Apr. 2004

    Noise Quality Measure (NQM)
        N. Damera-Venkata, T. Kite, W. Geisler, B. Evans and A. Bovik,
        "Image Quality Assessment Based on a Degradation Model",
        IEEE Trans. on Image Processing, Vol. 9, No. 4, Apr. 2000

    Universal Image Quality Index (UQI)
        Zhou Wang and Alan C. Bovik, "A Universal Image Quality Index",
        IEEE Signal Processing Letters, 2001

    Visual Information Fidelity (VIF)
        H. R. Sheikh and A. C. Bovik, "Image Information and Visual Quality".,
        IEEE Transactions on Image Processing, (to appear).

    Weighted Signal-to-Noise Ratio (WSNR)
        T. Mitsa and K. Varkur, "Evaluation of contrast sensitivity functions for
        the formulation of quality measures incorporated in halftoning algorithms",
        ICASSP '93-V, pp. 301-304.

    Signal-to-Noise Ratio (SNR, PSNR)
        J. Mannos and D. Sakrison, "The effects of a visual fidelity criterion on the
        encoding of images", IEEE Trans. Inf. Theory, IT-20(4), pp. 525-535, July 1974

    """


    results = {}

    if channels == 1:
        if len(img_ground.shape) < 3:
            img_ground = img_ground[:, :]
        else:
            img_ground = img_ground[:, :, 0]

        if len(image.shape) < 3:
            image = image[:, :]
        else:
            image = image[:, :, 0]

    igcol, igrow = img_ground.shape[0:2]
    imcol, imrow = image.shape[0:2]

    if igcol == imcol and igrow == imrow:
        if border != 0:
            img_ground = img_ground[border:igcol - border, border:igrow - border]
            image = image[border:imcol - border, border:imrow - border]

        if img_ground.dtype != 'uint8':
            img_ground = img_ground.astype(np.uint8)

        if image.dtype != 'uint8':
            image = image.astype(np.uint8)
        #
        # if "BRISQUE" in metrics or "ALL" in metrics:
        #
        #     BRSQ = brsq.BRISQUE()
        #     feat = brisque.brisque_features(image)[0]
        #     feat_scaled = BRSQ._scale_feature(feat)
        #     fp = BRSQ._calculate_score(feat_scaled)
        #     results["BRISQUE"] = fp

        if "BSNR" in metrics or "ALL" in metrics:
            results["BSNR"] = bsnr(img_ground, image)
        #
        # if "CNR" in metrics or "ALL" in metrics:
        #     results["CNR"] = None

        if "ERGAS" in metrics or "ALL" in metrics:
            results["ERGAS"] = sewar.ergas(img_ground, image)
        """
        if "FSIM" in metrics or "ALL" in metrics:
            results["FSIM"] = compute_fsim(img_ground, image)[1][0]
        """

        if "GMSD" in metrics or "ALL" in metrics:
            results["GMSD"] =  gmsd(img_ground, image, rescale=True, returnMap=False)
        #
        # if "HAARPSI" in metrics or "ALL" in metrics:
        #     results["HAARPSI"] = haar_psi(img_ground, image, False)[0]

        """
        if "ISNR" in metrics or "ALL" in metrics:
            #print("ISNR metric hasn't been implemented")
            results["ISNR"] = None
            # since isnr takes 3 parameter, but we have 2
            # it is going to be modified in future
            #results["ISNR"] =  isnr()
        """
        if "MAD" in metrics or "ALL" in metrics:
            results["MAD"] = mad.mad(img_ground, image)[0]

        if "MSE" in metrics or "ALL" in metrics:
            results["MSE"] = compare_mse(img_ground, image)

        if "MSSSIM" in metrics or "ALL" in metrics:
            results["MSSSIM"] = sewar.msssim(img_ground, image)

        if "NIQE" in metrics or "ALL" in metrics:
            # im = niqe(image.astype(np.float64))
            # imr = niqe(img_ground.astype(np.float64))
            # since NIQE requires image size bigger or equal than 192 in any
            # direction, for images have size is less than 192, set 'None'
            if (image.shape[0] <= 192 or image.shape[1] <= 192):
                print("\nOne of dimensions of Image is less than 192! NIQE requires",
                  "192 pixel at minimum in any dimension. The value of NIQE ",
                  "is being set to 'None'")
                results["NIQE"]= None

            else:
                results["NIQE"] = niqe(image)[0]
        #
        # if "NQM" in metrics or "ALL" in metrics:
        #     results["NQM"] = nqm(img_ground, image)

        if "NRMSE" in metrics or "ALL" in metrics:
            results["NRMSE"] = compare_nrmse(img_ground, image)

        if "PAMSE" in metrics or "ALL" in metrics:
            results["PAMSE"] = pamse(img_ground, image, rescale=True)

        # if "PBVIF" in metrics or "ALL" in metrics:
        #     results["PBVIF"] = pbvif(img_ground, image)

        if "PSNR" in metrics or "ALL" in metrics:
            results["PSNR"] = compare_psnr(img_ground, image)

        if "RASE" in metrics or "ALL" in metrics:
            results["RASE"] = sewar.rase(img_ground, image)

        if "SAM" in metrics or "ALL" in metrics:
            results["SAM"] = sewar.sam(img_ground, image)

        if "SCC" in metrics or "ALL" in metrics:
            results["SCC"] = sewar.scc(img_ground, image)

        if "SNR" in metrics or "ALL" in metrics:
            results["SNR"] =  snr(img_ground, image)

        if "SSIM" in metrics or "ALL" in metrics:
            results["SSIM"] =  SSIM(img_ground, image, multichannel=channels)

        # if "STRRED" in metrics or "ALL" in metrics:
        #     result["STRRED"] = strred(img_ground, image)

        if "UQI" in metrics or "ALL" in metrics:
            results["UQI"] = sewar.uqi(img_ground, image)

        if "VIF" in metrics or "ALL" in metrics:
            # results["VIF"] = vifp_mscale(img_ground, image) # from pymetrikz
            results["VIF"] = sewar.vifp(img_ground, image)

        """
        if "WSNR" in metrics or "ALL" in metrics:
            results["WSNR"] = wsnr(img_ground, image)
        """

        return results
    else:
        print('\nGround Image is of shape ', img_ground.shape, ',  but the image is of shape ', image.shape)
        print('Calculation has not been done!')
        return None, None


def calc_metrics_of_image(img_ground, image, border=0, channels=1, metrics=["PSNR", "SSIM"]):
    """
    Calculates PSNR and SSIM values of ground truth and output image of model

    Images to be converted to uint8 space. Intensity values should not be normalized
    So, values must be between 0 and 255 inclusive
    Returns:
    psnr and ssim values

    """

    if channels==1:
        if len(img_ground.shape) <3 :
            img_ground=img_ground[:,:]
        else:
            img_ground=img_ground[:,:,0]

        if len(image.shape) <3:
            image=image[:,:]
        else:
            image=image[:,:,0]

    igcol, igrow= img_ground.shape[0:2]
    imcol, imrow= image.shape[0:2]

    if igcol==imcol and igrow==imrow:
        if border != 0:
            img_ground=img_ground[border:igcol-border, border:igrow-border]
            image=image[border:imcol-border, border:imrow-border]

        if img_ground.dtype != 'uint8':
            img_ground = img_ground.astype(np.uint8)

        if image.dtype != 'uint8':
            image = image.astype(np.uint8)

        psnr = compare_psnr(img_ground, image)
        ssim = SSIM(img_ground, image, multichannel=channels)
        return psnr, ssim
    else:
        print('\nGround Image is of shape ', img_ground.shape, ',  but the image is of shape ', image.shape )
        print('Calculation has not been done!')
        return None, None


def min_max_normalize_image( img , minimum =0, maximum= 255.0, direction='both', separate_channels=False):
    """
    Normalizes the image so that intensity values in the image to be in a range of [minium, maximum].
    The range might be, for example [-1,1], or [0,255], or [0,1] etc.
    The formulae of normalization is as follows:

                    normalized_image =(b−a)*(x−min(x)) / (max(x)−min(x)) +a

    b is the maximum and a is the minimum value in the range to which that the intensity value in the image set.

    :param img: the image to be min-max scaled
    :param minimum: the value to which the minimum value of image will be set. For 0-255 or 0-1 normalizing
                    should be 0, for normalizing in the range of [-1,1] should be -1
    :param maximum: the value to which the maximum value of image will be set. For 0-255 scaling it should be 255;
                    for 0-1 scaling it should be 1.
    :param direction:
    :param separate_channels: Boolean. Each channels of the image will be processed separately if True, otherwise,
                              all channels of the image will be processed together.
    :return: returns the image normalized with min-max method
    """
    img = img.astype('float')

    if len(img.shape) == 2:
        single_channel = True # image has only one channel
    else:
        single_channel = False

    if single_channel:
        im = np.zeros((img.shape[0], img.shape[1]), dtype='float')
    else:
        im = np.zeros(img.shape, dtype='float')

    if direction == 'both':
        if separate_channels and not single_channel:
            for channel in range(img.shape[2]):
                tmp = img[:, :, channel]
                im[:,:,channel] = (maximum - minimum) * (tmp - tmp.min()) / (tmp.max() - tmp.min()) + minimum

        else: # image
            im = (maximum - minimum) * (img - img.min()) / (img.max() - img.min()) + minimum

    elif direction == 'column':
        if separate_channels:
            if single_channel: # single channel image
                for col in range(img.shape[1]):
                    tmp = img[:, col]
                    im[:, col] = (maximum - minimum) * (tmp - tmp.min()) / (tmp.max() - tmp.min()) + minimum

            else: # image has more than one channel
                for channel in range(img.shape[2]):
                    for col in range(img.shape[1]):
                        tmp = img[:,col, channel]
                        im[:,col, channel] = (maximum - minimum) * (tmp - tmp.min()) / (tmp.max()- tmp.min()) + minimum
        else:
            if single_channel: # single channel image
                for col in range(img.shape[1]):
                    tmp = img[:, col]
                    im[:, col] = (maximum - minimum) * (tmp - tmp.min()) / (tmp.max()- tmp.min()) + minimum

            else:
                for col in range(img.shape[1]):
                    tmp = img[:, col,:]
                    im[:, col, :] = (maximum - minimum) * (tmp - tmp.min()) / (tmp.max()- tmp.min()) + minimum

    elif direction == 'row':
        if separate_channels:
            if single_channel:
                for row in range(img.shape[0]):
                    tmp = img[row, :]
                    im[row , :] = (maximum - minimum) * (tmp - tmp.min()) / (tmp.max()- tmp.min()) + minimum

            else:  # image has more than one channel
                for channel in range(img.shape[2]):
                    for row in range(img.shape[0]):
                        tmp = img[row, :, channel]
                        im[row, :, channel] = (maximum - minimum) * (tmp - tmp.min()) / (tmp.max()- tmp.min()) + minimum
        else:
            if single_channel:
                for row in range(img.shape[0]):
                    tmp = img[row, :]
                    im[row, :] = (maximum - minimum) * (tmp - tmp.min()) / (tmp.max() - tmp.min()) + minimum
            else:
                for row in range(img.shape[0]):
                    tmp = img[row, :, :]
                    im[row, :, :] = (maximum - minimum) * (tmp - tmp.min()) / (tmp.max()- tmp.min()) + minimum

    return im


def standardize_image(img, mean = None,  direction='both', method ='std', separate_channels=False):
    """
    Standardize the image. The mean value of image or value in parameter 'mean' will be subtracted from
    the image at first, and then result is divided by standard deviation of itself, for standardization.
    :param img: image to be standardize
    :param mean: Must be Tuple or List. the mean value to be subtracted from the image. it is calculated from the image if it is not given
                It should be a tuple or list and have a value for each channel in the image, if 'separate_channels'
                is set True.
    :param method: 'std' for standard deviation; 'var' for variance.
    :param direction: The direction of standardization. Might be one of [both, column, row]
    :param separate_channels: Boolean. Each channel processed separately if set True
    :return:
    """
    if mean is None:
        mean_not_given = True
    else:
        mean_not_given = True

    func_dict = {'std': np.std, 'var': np.var}

    if method == 'std':
        func = func_dict['std']
    elif method == 'var':
        func = func_dict['var']
    else:
        print('\nInvalid method!. Method should be "std" or "var". "std" is taken for processing. ')
        func = func_dict['std']  # divider is the standard deviation

    img = img.astype('float')

    if len(img.shape) == 2: # image has only one channel,
        single_channel = True # image has one channel only
    else:
        single_channel = False # image has multiple channels

    if direction == 'both':
        dir = None
    elif direction == 'column':
        dir = 0
    elif direction == 'row':
        dir = 1
    else:
        print('\nWrong direction imput! direction must be one of "both", "column" or'
              ' "row".\nThe "both" option is selected as default')
        dir = None

    if single_channel: # image has only one channel
        im = np.zeros((img.shape[0], img.shape[1]), dtype='float')
    else:
        im = np.zeros(img.shape, dtype='float')

#----------------------------------------------------------------
#---------------------  BOTH DIRECTION  -------------------------
#----------------------------------------------------------------
    if direction == 'both':

        if separate_channels and not single_channel:
            for channel in range(img.shape[2]):
                tmp = img[:,:, channel]

                divider = func(tmp) + eps

                if mean_not_given:
                    mean = tmp.mean()
                    im[:, :, channel] = (tmp - mean) / divider
                else:
                    if len(mean) == 1:
                        im[:, :, channel] = (tmp - mean) / (divider + eps)
                    if len(mean)  == img.shape[2]:
                        im[:, :, channel] = (tmp - mean[channel]) / (divider + eps)
                    else:
                        print('The length of parameter of mean is not equal to the number of rows, or to 1 or to the '
                            ' number of channels of image. Cancelling processing...')
                    return

        else: # image has single channel or all channels processed together
            divider = func(img)

            if mean_not_given:
                mean = img.mean()
                im = (img - mean) / (divider + eps)
            else:
                im = (img - mean[0]) / (divider + eps)

# ----------------------------------------------------------------
# ----------------------|  COLUMN WISE  |-------------------------
# ----------------------------------------------------------------
    elif direction == 'column':

        if single_channel:  # single channel image
            tmp = img  # whole column at the position of the value of col.

            divider = func(tmp, axis=0)
            mean = tmp.mean(axis=0)

            if mean_not_given:
                # mean value is not given, calculate from the image for each columns
                im = (tmp - mean) / (divider + eps)
            else:
                im = (tmp - mean[0]) / (divider + eps)

# ------------------| FOR MULTI CHANNEL IMAGES |-------------------

        elif separate_channels:

            for channel in range(img.shape[2]):
                tmp = img[:, :, channel]

                divider = func(tmp, axis=0)

                if mean_not_given:
                    mean = tmp.mean(axis=0)
                    im[:, :, channel] = (tmp - mean) / (divider + eps)
                else:

                    if len(mean.shape) == 1: # one dimensonal array given for mean values

                        if len(mean) == len(tmp) or len(mean) == 1: # mean has only one or equal number of values to number of values of columns

                            im[:, :, channel] = (tmp - mean) / (divider + eps) # each column is divided by one mean value or corresponding mean value
                        if len(mean) == img.shape[2] :
                            im[:, :, channel] = (tmp - mean[channel]) / (divider + eps)
                        else:
                            print(
                                'The length of parameter of mean is not equal to the number of rows, or to 1 or to the '
                                ' number of channels of image. Cancelling processing...')
                            return None
                    elif len(mean.shape) == 2: # mean is two dimensional array

                        if mean.shape[0] == 1: # mean has only one value for each channel along rows
                            im[:, :, channel] = (tmp - mean[0,channel])
                        else: #now mean should have the same number of values as the number of rows, in first dimension.
                            im[:, :, channel] = (tmp - mean[:,channel])
                    else:
                        print('the number of dimensions of mean values does not match with the number of channels and/or the number of columns!'
                              '\nProcessing cancelled!')

                        return None, None, None

        else: # do not separate channels, process all channels together
            tmp = img  # whole column at the position of the value of col.

            divider = func(img, axis=(0,2))

            if mean_not_given:
                mean = img.mean(axis=(0,2))
                for channel in range(img.shape[2]):
                    im[:,:,channel] = (img[:,:, channel] - mean) / (divider + eps)

            else:
                if len(mean.shape) == 1: # one dimensional array given for mean values

                    if len(mean) == len(tmp.shape[1]) or len(mean) == 1: # mean has only one or equal number of values to number of values of columns

                        for channel in range(tmp.shape[2]):
                            im[:, :, channel] = (tmp[:,:,channel] - mean) / (divider + eps) # each column is divided by one mean value or corresponding mean value

                    if len(mean) == img.shape[2] : # mean has one value for each channels

                        for channel in range(tmp.shape[2]):
                            im[:, :, channel] = (tmp[: ,: , channel] - mean[channel]) / (divider + eps)

                    else:
                        print(
                            'The length of parameter of mean is not equal to the number of rows, or to 1 or to the '
                            ' number of channels of image. Cancelling processing...')
                        return None

                elif len(mean.shape) == 2: # mean is two dimensional array

                    if mean.shape[0] == 1: # mean has only one value for each channel along rows
                        im[:, :, channel] = (tmp[:,:, channel] - mean[0,channel])

                    else: #now mean should have the same number of values as the number of columns, in first dimension.
                        for channel in range(tmp.shape[2]):
                            im[:, :, channel] = (tmp[:, :, channel] - mean[:,channel]) / (divider + eps)
                else:
                    print('the number of dimensions of mean values does not match with the number of channels and/or the number of columns!'
                          '\nProcessing cancelled!')

                    return None, None, None

# ----------------------------------------------------------------
# -----------------------|  ROW WISE  |---------------------------
# ----------------------------------------------------------------
    elif direction == 'row':
        if single_channel:  # single channel image

            for r in range(img.shape[0]):
                row = img[r, :]  # entire row in the order of the variable 'row'

                divider = func(row)   # st. deviation or variance of rows

                if mean_not_given:
                    mean = row.mean()
                    im[r, :] = (row - mean) / (divider + eps)
                else:
                    if len(mean) == len(row):
                        im[r, :] = (row - mean) / (divider + eps)  # mean has the same number of columns as row
                    else:
                        im[r, :] = (row - mean[0]) / (divider + eps)  # only first value of mean to be subtracted from row

# ------------------| FOR MULTI CHANNEL IMAGES |-----------------

        elif separate_channels: # process each channels separately from each other in terms of direction of row

            for channel in range(img.shape[2]):
                for r in range(img.shape[0]):
                    # entire row of the channel in the order of the value of the variable 'channel',
                    # in the order of the value of the variable 'row'
                    row = img[r, :, channel] # one channel and a row

                    divider = func(row)  # st. deviation or variance of rows

                    if mean_not_given:
                        mean = row.mean()
                        im[r, :, channel] = (row - mean) / (divider + eps)
                    else:
                        if len(mean) == len(row):
                            im[r, :] = (row - mean) / (divider + eps)
                        else:
                            im[r, :, channel] = (row - mean[channel]) / (divider + eps)

        else:   # multi channel image
            for r in range(img.shape[0]):
                # entire r in the order of the value of the variable 'r' over all channels,
                row = img[r, :, :]
                divider = func(row, axis=1)  # st. deviation or variance of rs

                if mean_not_given: # mean value to be calculated from the image
                    mean = row.mean(axis=1)
                    for channel in range(row.shape[1]): # each channel of overall row
                        im[r, :, channel] = (row[:, channel] - mean) / (divider + eps)
                else: # mean value is given.
                    for channel in range(row.shape[1]): # each channel of overall row

                        if len(mean) == len(row) or len(mean) == 1:
                            im[r, :, channel] = (row[:,channel] - mean) / (divider + eps)

                        elif len(mean) == im.shape[2]: # the number of values in parameter of mean equals to the number of channels
                            im[r, :, channel] = (row[:, channel] - mean[channel]) / (divider + eps)

                        else:
                            print('The length of parameter of mean is not equal to the number of rows, or to 1 or to the '
                                  ' number of channels of image. Cancelling processing...')
                            return None, None, None

    return im, mean , divider


def mean_of_image(img, separate_channels=True, direction='both' ):
    """
    Calculates the mean of given image. If 'separate_channels' is True, then, each channels of the
    image is processed separately. If 'direction' is set to 'column' or 'row', the mean of each columns or rows
     calculated respectively. If 'direction' is set to 'both' overall mean of image

    :param img:
    :param separate_channels: Boolean. If it is True, each channels of image processed separately, then. Used with
                        parameter 'direction' to process image in specific direction (i.e., column or row, or both)
    :param direction: one of ['both', 'column', row']. Used with parameter 'separate_channels', to process each channel
                        separately.
    :return: the mean of the image, either total mean over all channels, or total mean of
        image in each direction of each channels (RGB, i.e.)
    """

    if len(img.shape) == 2:
        single_channel = True
    else:
        single_channel = False

    if direction == 'both':
        if separate_channels:
            if single_channel:
                mean= np.mean(img)
            else:
                mean= np.mean(img, axis=(0,1)) # mean of each channel calculated
        else:
            mean = np.mean(img) # overall mean of image calculated

    elif direction == 'column':
        if separate_channels:
            if single_channel:
                mean= np.mean(img, axis=0) # the mean of each column of single channel image
            else:
                mean= np.mean(img, axis=(0)) # mean of each column in each channel calculated separately
        else:
            mean = np.mean(img, axis=(0,2)) # mean of each column calculated

    elif direction == 'row':
        if separate_channels:
            if single_channel:
                mean = np.mean(img, axis=1)# the mean of each row of single channel image
            else:
                mean = np.mean(img, axis=(1)) # mean of each row in each channel calculated separately
        else:
            mean = np.mean(img, axis=(1, 2)) # mean of each row calculated

    return mean


def subtract_mean_from_image(img, mean=None, separate_channels=False, direction='both'):
    """
    Subtracts given mean value(s) from image. If parameter 'separate_channels' is set True, mean value from
    parameter 'mean' for corresponding channel in the is subtracted. Otherwise, only single mean value (first
    value of the parameter 'mean' in this case) is subtracted from each channels. The parameter 'direction'
    indicates what direction of processing will be followed. For example, if direction is set 'column', mean values
    (one for each corresponding column) are subtracted from each columns respectively.
    :param img: Image from which mean to be subtracted
    :param mean: Tuple or List. Mean values for each channels of image. Calculated from the given image, if not set
    :param separate_channels: Indicates that each channels of image to be processed separately
    :param direction: The direction of standardization. Might be one of [both, column, row]
    :return: Returns the image from which the mean subtracted.
    """
    if len(img.shape) == 2:
        single_channel = True # image has one channel only
    else:
        single_channel = False # image has multiple channels

    if direction == 'both':
        if separate_channels and not single_channel:
            for i in range(img.shape[2]):
                if mean is None:
                    img[:,:,i] -= np.mean(img[:,:,i])
                else:
                    img[:,:,i] -= mean[i] # one mean value for each channel

        else: # single channel image or image channels processed together
            if mean is None:
                img -= img.mean()
            else:
                img -= mean[0]

    elif direction == 'column':
        if separate_channels:
            if single_channel:
                for i in range(img.shape[1]):
                    if mean is None:
                        img[:,i] -= np.mean(img[:,i])
                    else:
                        img[:,i] -= mean[0]

            else:# multi channel image
                for i in range(img.shape[1]):
                    for j in range(img.shape[2]):
                        if mean is None:
                            img[:,i,j] -= np.mean(img[:,i,j])
                        else:
                            img[:,i,j] -= mean[j]

        else:
            if single_channel:
                for i in range(img.shape[1]):
                    if mean is None:
                        img[:,i] -= np.mean(img[:,i])
                    else:
                        img[:, i] -= mean[0]

            else:# multi channel image
                for i in range(img.shape[1]):
                    if mean is None:
                        img[:,i,:] -= np.mean(img[:,i,:])
                    else:
                        img[:,i,:] -= mean[0]

    elif direction == 'row':

        if separate_channels:
            if single_channel:
                for i in range(img.shape[0]):
                    if mean is None:
                        img[i,:] -= np.mean(img[i,:])
                    else:
                        img[i, :] -= mean[0]

            else:# multi channel image
                for i in range(img.shape[0]):
                    for j in range(img.shape[2]):
                        if mean is None:
                            img[i, :, j] -= np.mean(img[i, :, j])
                        else:
                            img[i, :, j] -= mean[j]

        else:
            if single_channel:
                for i in range(img.shape[0]):
                    if mean is None:
                        img[i, :] -= np.mean(img[i, :])
                    else:
                        img[i, :] -= mean[0]

            else: # multi channel image
                for i in range(img.shape[0]):
                    if mean is None:
                        img[i, :, :] -= np.mean(img[i, :, :])
                    else:
                        img[i, :, :] -= mean[0]

    return img


def preprocess_image(img,  scale=2, pad=0, channels=1, upscale=False, crop_remains=True,
                     img_type='input', decimation='bicubic', interp_up='bicubic', noise=None,
                     mode=None) :
    """
    Prepares image as ground truth, bicubic or downscaled image for model
    :param img: image to pre-process
    :param scale: the scale at which image is downscaled and, or, upscaled
    :param pad: the size of pixels cropped from each borders of image
    :param channels:  processing channel. for example, Y channel (first) of YCbCr color space.
                    set 3 for all channels
    :param upscale: image is upscaled after downscaling, is set True
    :param crop_remains: The image is cropped to fit the exact fold size of scale, if set.
                        the remaining number pixels result in (image shape size) % scale
    :param img_type: One of ['input', 'bicubic' , 'ground']. they indicates if the image is processed is the image to be input to model
                        or bicubic scaled image, or ground truth image. The image is processed accordingly
    :param decimation: The interpolation method for downscaling image.
                        One of 'bicubic', 'nearest', 'lanczos', 'cubic'. Default is 'bicubic'
    :param interp_up: Interpolation method for upscaling image.
                        One of 'bicubic', 'nearest', 'lanczos', 'cubic'. Default is 'bicubic'
    :param noise: Gaussian noise to be added to images.
    :return: Results pre-processed image with ndarray.
    """
    func = {'nearest': 0, 'lanczos': 1, 'bilinear': 2, 'bicubic': 3, 'cubic': 3}

    shape_length = len(img.shape)

    if shape_length < 3:  # it is not a 3 channel image
        row, col = img.shape[0:2]
    else:
        row, col, c = img.shape  # 3 channel image

    if crop_remains:  # ground truth image, just crop borders
        row_start = int((row % scale) / 2)
        col_start = int((col % scale) / 2)

        row -= int(row % scale)  # the width of the image being made exact fold of the scale
        col -= int(col % scale)  # the height of the image being made exact fold of the scale

        if channels == 1 or shape_length < 3: # channels might be 3, but img channel might be 1
            if shape_length <3:
                img = img[row_start:row_start + row, col_start: col_start + col]
            else:
                # reference image is made exact fold of the scale, and Channel Y is taken
                img = img[row_start:row_start + row, col_start: col_start + col, 0]
        else:
            # reference image is made exact fold of the scale, and all channels are taken
            img = img[row_start:row_start + row, col_start: col_start + col, :]

    if img_type == 'ground':
        if pad != 0:
            img = img[pad:row-pad,pad:col-pad]

        return img
    img = Image.fromarray(img, mode=mode)

    org_shape = (img.size[0], img.size[1])

    new_shape = np.divide(org_shape, scale).astype(int)

    img = img.resize(new_shape, resample=func[decimation])

    # img = misc.imresize(img, 1.0 / scale, interp=decimation, mode=mode) #type uint8

    if noise is not None and noise != '':
        img = np.array(img)
        img = img_as_float32(img)
        img = random_noise(img, mode='gaussian', mean=float(noise[0]), var=float(noise[1]))
        #img = img_as_uint(img)

    if upscale:

        if 'PIL.Image.Image' not in str(type(img)):
            img = Image.fromarray(img, mode=mode)
        # if upscale interpolation method for upscaling is not defined
        # as is in the parameter 'decimation'
        if interp_up is None:
            interp_up = decimation

        img = img.resize(org_shape, resample=func[interp_up])

        img = np.array(img)

        # img = misc.imresize(img, scale / 1.0, interp= interp_up, mode=mode) # interpolated image

    if 'PIL.Image.Image' in str(type(img)):
        img = np.array(img)

    if pad != 0:
        img = img[pad:row-pad, pad:col-pad]

    return img


def prepare_input_ground_bicubic(img, scale=2, pad=0, upscaleimages=True, channels=1):
    """
    Parameters
    ----------
    img,pad, normalization

    normalization -> if normalization is True, intensity values of image is divided by 255.
    pad -> if the model's output image is cropped by some pad value, the

    Returns
    -------
    Ground truth, Input and bicubic images : numpy array
        Input is the input image for model
    """

    # prepare ground image
    img_ground = preprocess_image(img, scale = scale, pad=pad, channels=channels,
                                  upscale=False, crop_remains=True, img_type='ground')

    # prepare low image
    # pad value is zero (0) for input image, in any case, If the model outputs
    # an image cropped by some size of padding value
    img_low = preprocess_image(img, scale = scale, pad=0, channels=channels,
                               upscale= (not upscaleimages), crop_remains=True, img_type='input')

    # prepare bicubic image
    img_bicubic = preprocess_image(img, scale = scale, pad=pad, channels=channels,
                                   upscale=True, crop_remains=True, img_type='bicubic')

    return img_ground, img_bicubic, img_low


def prepare_image_for_model(img_input, channels=1, ordering_style='channels_last'):
    """
    Prepare image for model input
    """
    shape_length = len(img_input.shape)

    yatay, dikey = img_input.shape[:2]
    img_out = np.zeros((1,yatay,dikey,channels), dtype=float)

    if channels == 1 or shape_length <3:
        if shape_length == 3:
            img_out[0, :, :, 0] = img_input[:,:,0]
        else:
            img_out[0,:,:,0] = img_input[:,:]
    else: # image has 3 channels

        img_out[0,:,:,:] = img_input[:,:,:]

    # if backend is theano, image ordering must be changed to
    # obey to THEANO ordering style
    if ordering_style == 'channels_first': # if THEANO style image ordering
        img_out = img_out.transpose((2,0,1))

    return img_out


def test_single_image(model, input_image, border=0, modelscalesimage=True,
                      pad=0, channels=1, scale=2, normalize=False,
                      returnImage=True, target_channels=1):

    img_ground, img_bicubic, img_input = \
                    prepare_input_ground_bicubic(input_image, scale, pad, modelscalesimage, channels)


    if normalize:
        img_ground=img_ground * 255.
        img_ground=np.minimum(img_ground,255.0).astype(np.uint8)

        img_bicubic=img_bicubic * 255.
        img_bicubic=np.minimum(img_bicubic,255.0).astype(np.uint8)

        img_input = prepare_input_image_for_model(img_input, channels)
        img_result = model.predict(img_input, batch_size=1)
        img_result= img_result[0,:,:,0:channels]
        img_result= img_result * 255.0
        img_result = np.minimum(img_result,255.0).astype(np.uint8)
    else:
        img_input = img_input.astype(np.float32) / 255.
        img_input = prepare_input_image_for_model(img_input, channels)

        img_result = model.predict(img_input, batch_size=1)
        img_result= img_result[0,:,:,0:channels]

    psnr_model, ssim_model = \
        calc_metrics_of_image(img_ground, img_result, border, target_channels)

    psnr_bicubic, ssim_bicubic = \
        calc_metrics_of_image(img_ground, img_bicubic, border, target_channels)

    if returnImage:
        return (psnr_model, ssim_model), (psnr_bicubic, ssim_bicubic), img_result
    else:
        return psnr_model, ssim_model, None


def prepare_image(orijinal_image, scale, pad, multi_channel=False, Input_Image_Scaled=False):
    """
    Parameters
    ----------
    Input_Image_Scaled

    Returns
    -------
    Ground truth, Input and bicubic images : numpy array
        Input is the input image for model

    """
    height, width, c = orijinal_image.shape
    height -= int(height % scale) #
    width -= int(width % scale) #

    if(multi_channel):
        # reference image is being made exact fold of the scale, and color channel Y is taken
        Ground_Image = orijinal_image[0:height, 0:width,:]
    else:
        # reference image is made exact fold of the scale, and all color channels are taken
        Ground_Image = orijinal_image[0:height, 0:width, 0]

    #img_low = gaussian_filter(Ground_Image, 0.5)

    Ground_Image = Ground_Image[pad:height-pad,pad:width-pad]
    Ground_Image = Ground_Image.astype(np.float32) /255.

    # downscale the image by the scale factor
    img_low = misc.imresize(Ground_Image, 1.0 / scale, 'bicubic') #type uint8
    img_low = img_low.astype(np.float32) / 255.

    # restore back the size of the image.
    img_bicubic = misc.imresize(img_low, scale / 1.0, 'bicubic') # bicubic interpolated image.
    img_bicubic = img_bicubic.astype(np.float32) / 255.

    # if model takes the input image as downscaled
    if Input_Image_Scaled:
        img_low = img_bicubic

    # if the borders of the output image of the model is cropped
    # bicubic upscaled image should also be cropped by padding size
    # for calculation of PSNR and SSIM values.
    img_bicubic = img_bicubic[pad:height-pad, pad:width-pad]
    return Ground_Image, img_bicubic, img_low


def write_to_excel(dataset, file):
    """
    Writes pandas dataframe to excel file
    :param dataset:
    :param file:
    :return:
    """
    writer = pd.ExcelWriter(file, engine='openpyxl')
    for scale, values in dataset.items():
        values.transpose().to_excel(writer, "Scale-" + scale)
        values.stack().loc['Mean', :].to_excel(writer, 'Mean-' + scale)
    writer.save()
    writer.close()


def remove_if_exist(file_name):
    if exists(file_name):
        remove(file_name)


def augment_image(image, options):
    """
    Augments the input image. Augmentation process covers rotation of 90, 180, 270 degrees
    and transposition of upside down, left to right and mix of the both. Transposition
    styles are defined by options parameter. options can be any or all of
    [90,180,270, 'flipud', 'fliplr', 'flipudlr'].
    Takes scikit-image as input (numpy ndarray) and returns augmented images.

    :param image:
    :param options: a list consists of [90,180,270, 'flipud', 'fliplr', 'flipudlr']
    :return: returns transposed and rotated versions of original image,
     as well as the original image, totally 7 images.
    """
    images = []
    images.append(image)

    scales = []
    noises = []
    interps = []

    if 90 in options or "90" in options:
        images.append(np.rot90(image))

    if 180 in options or "180" in options: # rotate 180 degrees
        images.append(np.rot90(image,2)) # rotate image two times for 180 degrees

    if 270 in options or "270" in options: #rotate 270 degrees
        images.append(np.rot90(image,3))

    if 'flipud' in options: # flip upside down
        images.append(np.flipud(image))

    if 'fliplr' in options: # flip left to right
        images.append(np.fliplr(image))

    if 'flipudlr' in options: # flip upside down and then left to right
        images.append(np.fliplr(np.flipud(image)))

    """
    for item in options:
        if 'scale' in item:
            scales = item.split(' ')[1:]

        elif 'noise' in item:
            noises = item.split(' ')[1:]

        elif 'interp' in item:
            interps = item.split(' ')[1:]

    n_scales = len(scales)
    n_noises = len(noises)
    n_interps= len(interps)

    if n_scales <1: n_scales=1
    if n_noises <1: n_noises=1
    if n_interps<1: n_interps=1

    n_total_images = n_scales * n_noises * n_interps + len(images)

    all_images = [None] * n_total_images

    i = 0
    for im in images:
        all_images[i] = im

        if len(interps) > 0:
            f

    for i in range(n_total_images):
        for im in images:
            all_images[i] = im

            for interp in interps:
                for scale in scales:
                    for noise in noises:

                        pass

    """
    return images


def shuffle_data(output_file):
    """
    Shuffles  input images or input image patches.
    :param output_file:
    :return:
    """
    shuffle_file = h5py.File(output_file,'r')
    x_test = shuffle_file['input'][()] # nd array
    y_test = shuffle_file['label'][()] # nd array

    shuffle_file.close() # close it

    rename(output_file, output_file +"ordered.h5")

    num_records = x_test.shape[0] # what number of records?
    shuffled_list = list(range(0, num_records))

    np.random.shuffle(shuffled_list)# shuffle the list

    x_shuffled =np.ones_like(x_test)
    y_shuffled =np.ones_like(y_test)

    for i in range(0,num_records):
        j= shuffled_list[i]
        x_shuffled[i] = x_test[j]
        y_shuffled[i] = y_test[j]

    with h5py.File(output_file, 'w') as f:
        f.create_dataset("input", data=x_shuffled)
        f.create_dataset("label", data = y_shuffled)
        f.close()
    return output_file


def plot_history(history):
    pass


def get_h5_datasize(file):
    """
    Returns the number of records in a .h5 file
    :param file:
    :return:
    """
    h5f = h5py.File(file, 'r')
    data = h5f['input'].shape
    print(data)
    return data

