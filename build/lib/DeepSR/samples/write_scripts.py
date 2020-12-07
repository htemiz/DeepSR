

file_train = "training script.bat"
file_test  = "test script.bat"

python_path = r"C:\Users\hakan\AppData\Local\Programs\Python\Python36\python.exe"

lines_train = []
lines_test = []


traindir = r"D:\calismalarim\datasets\DIV2K\DIV2K_train_HR"
testdir = r"D:\calismalarim\datasets\DIV2K\DIV2K_valid_HR"
valdir = r""

traindir = r"d:\training dir"

models = ["EDSR"]

input_sizes = [24, 36, 48]

strides = [11, 17, 23]

learning_rates  = [0.01, 0.001 ]
normalizations =    [
                        'divide 255.0',  # divide each intensity values of images by 255.0
                      'standard whole', # standardization with mean and std calculated from entire training set
                      'mean',  # mean normalization
                      'minmax 0 1'  # minmax normalization between 0 and 1
                   ]

batch_sizes = [16, 32, 64]

color_modes = ['RGB', 'YCbCr']
color_modes = ['RGB']

metrics = "ERGAS GMSD PAMSE PSNR SAM SCC SSIM VIF"

decimations = ['bicubic', 'lanczos']

sep = '_'

for model in models:

    for color_mode in color_modes:

        for normalization in normalizations:

            for learning_rate in learning_rates:

                for input_size in input_sizes:

                    for stride in strides:

                        for batch_size in batch_sizes:

                            line = python_path + " -m DeepSR.DeepSR --modelname \"" + model + sep + color_mode + sep + normalization + sep + str(learning_rate) + sep
                            line += str(input_size) + sep + str(stride) + sep + str(batch_size) + '\"'

                            if color_mode == "RGB":
                                line += " --modelfile EDSR_RGB.py"

                            else:
                                line += " --modelfile EDSR_YCbCr.py"

                            line += " --train --traindir " + traindir
                            line += ' --colormode ' + color_mode
                            line += ' --normalization ' + normalization
                            line += ' --lrate ' + str(learning_rate)
                            line += ' --inputsize ' + str(input_size)
                            line += ' --stride ' + str(stride)
                            line += ' --batchsize ' + str(batch_size)
                            line += '\n'

                            lines_train.append(line)


                            line = "python -m DeepSR.DeepSR --modelname \"" + model + sep + color_mode + sep + normalization + sep + str(learning_rate) + sep
                            line += str(input_size) + sep + str(stride) + sep + str(batch_size) + '\"'

                            if color_mode == "RGB":
                                line += " --modelfile EDSR_RGB.py"

                            else:
                                line += " --modelfile EDSR_YCbCr.py"

                            line += " --test --testpath " + testdir
                            line += ' --colormode ' + color_mode
                            line += ' --normalization ' + normalization
                            line += ' --lrate ' + str(learning_rate)
                            line += ' --inputsize ' + str(input_size)
                            line += ' --stride ' + str(stride)
                            line += ' --batchsize ' + str(batch_size)
                            line += '\n'

                            lines_test.append(line)

with open(file_train, 'a') as f:
    f.writelines(lines_train)

with open(file_test, 'a') as f:
    f.writelines(lines_test)






