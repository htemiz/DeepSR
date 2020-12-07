
from os import listdir, makedirs, remove
from os.path import isfile, join, exists, splitext, abspath, basename
from os import getcwd
from urllib.request import urlretrieve
from skimage.io import imsave, imread
from time import sleep
import glob
from PIL import Image, ImageOps

width  = 600
height = 450

working_dir = r"D:\calismalarim\Program\DeepSR\DeepSR\samples\sampe_images"


model_names = ['EDSR_minmax 0 1_0.001_24_13_16']

model_names = ['EDSR_minmax 0 1_0.001_24_13_16',
               'EDSR_minmax 0 1_0.001_24_23_16',
               'EDSR_minmax 0 1_0.001_48_13_16',
               'EDSR_minmax 0 1_0.001_48_23_16',
               'EDSR_minmax 0 1_0.0001_24_13_16',
               'EDSR_minmax 0 1_0.0001_24_23_16',
               'EDSR_minmax 0 1_0.0001_48_13_16',
               'EDSR_minmax 0 1_0.0001_48_23_16',
               'EDSR_minmax -1 1_0.001_24_13_16',
               'EDSR_minmax -1 1_0.001_24_23_16',
               'EDSR_minmax -1 1_0.001_48_13_16',
               'EDSR_minmax -1 1_0.001_48_23_16',
               'EDSR_minmax -1 1_0.0001_24_13_16',
               'EDSR_minmax -1 1_0.0001_24_23_16',
               'EDSR_minmax -1 1_0.0001_48_13_16',
               'EDSR_minmax -1 1_0.0001_48_23_16',
               'EDSR_minmax 0 1_0.001_24_13_32',
               'EDSR_minmax 0 1_0.001_24_23_32',
               'EDSR_minmax 0 1_0.001_48_13_32',
               'EDSR_minmax 0 1_0.001_48_23_32',
               'EDSR_minmax 0 1_0.0001_24_13_32',
               'EDSR_minmax 0 1_0.0001_24_23_32',
               'EDSR_minmax 0 1_0.0001_48_13_32',
               'EDSR_minmax 0 1_0.0001_48_23_32',
               'EDSR_minmax -1 1_0.001_24_13_32',
               'EDSR_minmax -1 1_0.001_24_23_32',
               'EDSR_minmax -1 1_0.001_48_13_32',
               'EDSR_minmax -1 1_0.001_48_23_32',
               'EDSR_minmax -1 1_0.0001_24_13_32',
               'EDSR_minmax -1 1_0.0001_24_23_32',
               'EDSR_minmax -1 1_0.0001_48_13_32',
               'EDSR_minmax -1 1_0.0001_48_23_32'
               ]



row_start = 60
col_start= 60

h = 200
w = 300


for model in model_names:

    yol = join(working_dir, model + r"\output\2\images")

    dosyalar = glob.glob(join(yol, "*.png"))

    for file in dosyalar:
        img = imread(file)

        if len(img.shape) == 3:
            h, w, z = img.shape
        else:
            h, w = img.shape

        name = file.split('\\')[-1]

        if 'comic' in name:
            row_start = 145
            col_start = 30
            h = 30
            w = 45

        elif 'bird' in name:
            row_start = 25
            col_start = 135
            h = 30
            w = 45

        elif '0880' in name:
            row_start = 425
            col_start = 525
            h = 120
            w = 180



        file_write = join(yol,  name + "_" + model + '_cropped.png')
        new_img = img[row_start: row_start + h, col_start: col_start + w,:]

        im_write = Image.fromarray(new_img)
        im_write= im_write.resize((720, 480))
        im_write.save(file_write, dpi=(600, 600))



