
# this file composes command line scripts for training and test of EDSR

script_file = "batch script.bat"
script_file = "model names.txt"

lines = [] # command lines

traindir = r"D:\calismalarim\datasets\DIV2K\sub_train"
testdir = r"D:\calismalarim\datasets\DIV2K\sub_test"

model = "EDSR"

# 2^6 = 64 combinations of parameters.
# we do not take other parameters into calculation
# otherwise, it takes month/years to accomplish trainings and tests.
input_sizes = [24, 48]  # the size of input image patches for training
strides = [13, 23]  # stride values when taking
learning_rates  = [0.001, 0.0001 ]
normalizations =    [
                      'minmax 0 1',  # minmax normalization between -1 and 1
                      'minmax -1 1'  # minmax normalization between -1 and 1
                   ]
batch_sizes = [16, 32]
# color_modes = [ 'RGB', 'YCbCr']
color_modes = [ 'RGB']


metrics = "ERGAS GMSD PAMSE PSNR SAM SCC SSIM VIF"  # image quality measures (IQM). DeepSR can measure 18 IQMs

decimations = ['bicubic', 'lanczos']

sep = '_'

for color_mode in color_modes:
    for batch_size in batch_sizes:
        for normalization in normalizations:
            for learning_rate in learning_rates:
                for input_size in input_sizes:
                    for stride in strides:
                        # line = "python.exe -m DeepSR.DeepSR --modelname \"" + model + sep + color_mode + sep + normalization + sep + str(learning_rate) + sep
                        line =  "'" + model + sep + normalization + sep + str(learning_rate) + sep
                        line += str(input_size) + sep + str(stride) + sep + str(batch_size) + "\'"

                        """
                        # line = "python.exe -m DeepSR.DeepSR --modelname \"" + model + sep + color_mode + sep + normalization + sep + str(learning_rate) + sep
                        line = "python.exe -m DeepSR.DeepSR --modelname \"" + model + sep + normalization + sep + str(learning_rate) + sep
                        line += str(input_size) + sep + str(stride) + sep + str(batch_size) + '\"'

                        if color_mode == "RGB":
                            line += " --modelfile EDSR.py"

                        else:
                            line += " --modelfile EDSR_YCbCr.py"

                        line += " --train"  # do test after training has finished. Test folder given in 'Settings'.
                        # give a different folder path for test to do test
                        # for another directory than given in 'Settings'
                        # Multiple paths can be given in case it is necessary to do test for
                        # different sets of images located in multiple folders.
                        # line += " --testpath " + test_path

                        line += ' --batchsize ' + str(batch_size)
                        line += ' --normalization ' + normalization
                        line += ' --lrate ' + str(learning_rate)
                        line += ' --inputsize ' + str(input_size)
                        line += ' --stride ' + str(stride)
                        line += '\n'
                        """
                        line += ', '
                        lines.append(line)

with open(script_file, 'a') as f:
    f.writelines(lines)
