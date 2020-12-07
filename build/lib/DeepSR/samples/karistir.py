
from os import listdir, makedirs, remove
from os.path import isfile, join, exists, splitext, abspath, basename
import glob
import numpy as np
from pathlib import Path
import csv
from DeepSR import utils
# import utils
import shutil


n_images = 100

traindir = r"D:\calismalarim\datasets\DIV2K\DIV2K_train_HR"
testdir = r"D:\calismalarim\datasets\DIV2K\DIV2K_valid_HR"

new_path = r"D:\calismalarim\datasets\DIV2K\sub_train"


l = utils.get_files(path = traindir, extentions = ["*.png"], remove_extension=False, full_path = False)

test_list = np.random.choice(range(len(l)),n_images, replace=False)

for i in test_list:
    fname = l[i]

    new_file = join(new_path , fname)
    shutil.move(join(traindir, fname), new_file)
