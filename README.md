## <p align='center'> A Python Tool for Obtaining and Automating Super Resolution with Deep Learning Algorithms </p>

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4310169.svg)](https://doi.org/10.5281/zenodo.4310169)

<p align='justify'>
DeepSR is an open source progam that eases the entire processes of the Super Resolution (SR) problem in terms of Deep Learning (DL) algorithms. DeepSR makes it very simple to design and build DL models. Thus, it empowers researchers to focus on their studies by saving them from struggling with time consuming and challenging workloads while pursuing successful DL algorithms for the task of SR.
</p>

<br/>
 <p align='justify'>
Each step in the workflow of SR pipeline, such as pre-processing, augmentation, normalization, training, post-processing, and test are governed by this framework in such a simple, easy and efficient way that there would remain at most very small amount of work for researchers to accomplish their experiments. In this way, DeepSR ensures a way of fast prototyping and providing a basis for researchers and/or developers eliminating the need of commencing whole job from the scratch, or the need of adapting existing program(s) to new designs and implementations. 
</p>

</br>



<p align='center'>
<img align='center' src="https://github.com/htemiz/DeepSR/blob/master/DeepSR/docs/conceptual scheme.png" style="width:160px,height=100px;text-aiign:center;display:blcok;"/>
<span>Conceptual scheme of DeepSR for Super Sesolution with Deep Learning
 </p>




<p align='justify'>
DeepSR is designed in such a way that one can interact with it from the command prompt, or use it as class object by importing it into another program. It is mainly tailored for using it with scripts from command prompt by providing many ready-to-use functionalities at hand. Hence, multiple tasks/experiments can be performed successively by using batch scripts. However, addition to command prompt interface, it is ready also to do the same tasks by calling it as a class object from another Python program. Addition to this, one can develop his/her own programs or can write Python scripts performing subject specific tasks at his/her disposal. Even more, he/she can add new features to the program to contribute this effort for making it better.
</p>
 
  In order to construct DL models, DeepSR utilizes [Kears](https://www.keras.io). Keras is capable of running on top of the
  following prominent DL frameworks: 
  
  [TensorFlow](https://www.tensorflow.org), [Theano](http://deeplearning.net/software/theano/) and [CNTK](https://github.com/microsoft/CNTK)
   
 It transforms DL models designed by coding with its API into the models of these frameworks. Thus, this program  is capable of running models on both CPUs or GPUs seamlessly by leveraging these frameworks.


## Installation

In order to install the DeepSR, issue the following command in command prompt.

    python -m pip install DeepSR
     
The downloadable binaries and source code are available at PyPI repository in the following address:
 
 https://pypi.org/project/DeepSR

  ## How to Use It
<p align='justify'>
DeepSR fundamentally consists of three Python files: DeepSR.py, args.py, and utils.py. 
The DeepSR object is declared in the file DeepSR.py. It is the core python module called first for running the program.
 All parameters that can be used with DeepSR in command prompt are defined in the file args.py.
  The module, utils.py accompanies many tools making the life easier and maintaining the reusability.

    
<p align='center'>
<img align='center' src="https://github.com/htemiz/DeepSR/blob/master/DeepSR/docs/The&#32;DeepSR.png" style="width:160px,height=100px;text-aiign:center;display:blcok;"/>
 </p>
 
<p align='justify'>
All needs to be done is to build up a DL model using the API of Keras within a model file (a python file),
  and then, provide the program the relevant information and/or instructions. A Model file is an ordinary 
  Python file (a .py) must have at least two things within itself:
- A dictionary named `settings`. This dictionary is used to supply the DeepSR with all necessary settings
    and/or instructions to perform tasks.
- A method named `build_model`. This method preserves the definition of the model in Keras API,
  and returns it the composed version of it. By this way, DeepSR intakes deep learning models 
  (either from command prompt, or whereas it is used as an class object)
 </p>

<p align='justify'>
Please refer to the <b>Section 3</b> in the program manual, for further information about model files.
 To see a sample model file, locate the samples folder within DeepSR folder.
 </p>
 
<p align='justify'>
This dictionary is used to provide the program all necessary settings and/or instructions to do intended tasks. 
Settings are given in the dictionary in a key-value pairs. All keys are summarized in the [manual](DeepSR/docs/DeepSR&#32;Manual&#32;v1.0.0.pdf)
The keys correspond to command arguments in command prompt. One can easily construct and set his/her models
with parameters by providing key-value pairs in this dictionary to perform Super Resolution task with 
Deep Learning methods.
</p>

<p align='justify'>
However, the same settings can also be given to the program as arguments in command prompt.
Please note that although settings were designated in the dictionary, the same settings provided as arguments
in command prompt override them. Thus, the program ignores the values/instructions of those settings designated 
in the dictionary and uses the ones given as command arguments. 
 </p>
 
<p align='justify'>
The alteration of the settings is valid only for the class object, not valid for model files.
Namely, settings are not changed in the dictionary within the model files. Model files will remain intact. 
</p>


### Interacting with Command Prompt

<p align='justify'>

The following code snippet is a sample for the use of the program from command line. This code instructs the DeepSR 
to start training the model given (returned from build_model function) in the file 'sample_model_1.py' with the parameters
given in the dictionary 'settings' within the same file. If you check the model file, for example,
you will notice that training will to be performed for 10 epoch, scale is 4, learning rate is 0.001, and so on. 
The test results (scores of image quality measures used in test procedure) 
obtained from each test images are saved in an Excel file with the average scores of measures over test set.
</p>
    
    python DeepSR.py --modelfile “sample_model_1.py” --train

Please note that it is assumed that the location of the DeepSR.py file has already set to the system's PATH.

To perform the test for the evaluation of the model's performance on given test set of images 
by a number of image quality measures, add `--test` argument to above code
 
    python DeepSR.py --modelfile “sample_model_1.py” --train --test

The following code augments images with 90, 180, 270 degree of rotations, and applies 
Min-Max normalization (0-1) before they were given to the model for training procedure. After training, test is 
going to be applied also. 
 
    python DeepSR.py --modelfile “sample_model_1.py” -–train -–augment 90 180 270 --test  --normalization minmax

Below, another model file (you can locate this file under samples folder) is used to perform the test for 
a trained model by saving the outputs (images) of each layers in image files.

    python DeepSR.py --modelfile “sample_model_2.py” -–test --layeroutputs --saveimages



### Example for Running DECUSR Model With Scripting Interface (command prompt) 
Below is the DECUSR model designed by me for Ultrasound Super Resolution. It is a special kind of CNN with densely connected 
repeating blocks. As typical, hyper-parameter values are given in the dictionary 'settings'. The model is defined below this dictionary.

Let us train it and do some works:

    python –m DeepSR.DeepSR --train --test --plot --saveimages --modelfile 'DECUSR.py’ --inputsize 25 
    --metrics PSNR SSIM MAD --scale 3 --gpu 1 --shuffle --shutdown  --normalize minmax -1 1 --stride 12
    --noise 0.0 0.1 --lrpatience 3 --espatience 5 --epoch 100 --batchsize 256 --backend tensorflow 
    --layeroutput --layerweights 

| Argument | Explanation  |
| :------  | :-----  |
| --train | Train the model with given hyper-parameter values |
| --test | Evaluate the performance of the model with given measures after training finished |
| --plot | Plot the moodel's architecture as an image file |
| --saveimages | Save output images generated by DECUSR while testing.|
| --modelfile 'DECUSR.py' | Use DECUSR.py file for the DL model with its hyper-parameters |
| --inputsize 25 | Image patches taken from training images have 25x25 pixel size. This overwrites the value given in the 'settings' dictionary | 
| --metrics | Use PSNR SSIM and MAD measures for the evaluation |
| --scale 3 | Scale factor is 3. Overwrites the value 2 in the 'settings'. |
| --gpu 1 | Do all tasks with only second GPU, if there are multiple GPUs. (order is zero based).|
| --shuffle | Shuffle training images before starting each epoch. |
| --nomralize minmax -1 1 | Normalize images between -1 and 1. |
| --stride 12 | Give 12 pixel apart for extracting image patches from training images. |
| --noise 0.0 0.1 | Add Gaussian noise with zero mean and 0.1 std on images berfore passing to the model. |
| --lrpatience 3 | Reduce learning rate if there is no improvement on the performance for 3 epochs. |
| --espatience 5 | Stop training if there is no improvement on the performance for 5 epochs. |
| --epoch 100 | Train the model for 100 epochs. |
| --batchsize 256 | Calculate the training loss after passing 256 image patches to the model. |
| --backend tensorflow | Tensorflow will be used as the backend framework. |
| --layeroutput | Write in files the outputs of all layers of the model. | 
| --layerweights | Write in files the weights of the layers. |

Please note that command arguments overwrites the hyper-parameter values in the 'settings'. DeepSR automatize some important
features. For example, the activation fucntions in the layers are defined as 'activation=self.activation', which means that it
can be dynamically set by command argument or in the dictionary 'settings' (for example, in command prompt, '--activation "tanh"').
 In this way, activations can be changed with dynamically scripts in the implementation stage. 

```python
# DECUSR.py file
# DECUSR model with 4 Repeating Blocks for Scaling Factor of 2.

from keras import metrics
from keras import losses
from keras.models import Model
from keras.layers import Input, merge, ZeroPadding2D,  LocallyConnected2D, Conv2DTranspose, concatenate, BatchNormalization
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import keras.backend as K
from os.path import  dirname, abspath, basename
from keras.optimizers import Adam, SGD, RMSprop

eps = 1.1e-6

settings = \  
    {  
        'activation': 'relu', # activation function for layers is relu.
        'augment': [90, 180],  # can be any combination of [90,180,270, 'flipud', 'fliplr', 'flipudlr' ], or []  
        'backend': 'tensorflow', # keras is going to use theano framework in processing.  
        'batchsize':128,  # number of batches  
        'channels': 1,  # color channels to be used in training . Only one channel in this case  
        'colormode': 'RGB',  # the color space is RGB. 'YCbCr' or 'RGB'  
        'crop': 0,  # do not crop from borders of images.  
        'crop_test': 0,  # do not crop from borders of images in tests.  
        'decay': 1e-6,  # learning rate decay.
        'decimation': 'bicubic', # can be one of 'bicubic', 'bilinear', 'lanczos', 'nearest'  
        'espatience': 5,  # Early stopping patience. Stop after 5 epochs if the performance of the model has not improved.  
        'epoch': 50,  # train the model for total 50 passes on training data.
        'gpu': 0,1, # use the first and second GPUs in the computing system.  
        'inputsize': 16, # size of input image patches is 33x33.  
        'lactivation': 'prelu', # activation function of the last layer in the model is prelu.  
        'lrate': 0.001, # learning rate
        'lrpatience': 3,  # The learning rate plateau patience. The number of epochs to wait before reducing the learning rate.  
        'lrfactor': 0.5,  # Learning rate plateau factor. The ratio of decrease in learning rate value.  
        'minimumlrate': 1e-7,  # learning rate can be reduced down to a maximum of this value.  
        'modelname': basename(__file__).split('.')[0],  # modelname is the same as the name of this file.  
        'metrics': ['PSNR', 'SSIM'],  # evaluation metrics PSNR and SSIM.  
        'normalization': ['divide', '255.0'],  # normalize images by dividing 255.0  
        'outputdir': '',  # sub directories automatically created.  
        'scale': 2,  # magnification factor is 4.  
        'shuffle': True, # shuffle training images at the begining of each epoch.
        'stride': 19,  # give a step of 11 pixels apart between patches while cropping them from images for training.  
        'target_channels': 1,  # color channels to be used in tests . Only one channel in this case  
        'target_cmode': 'RGB',  # 'YCbCr' or 'RGB'  
        'testpath': [r'D:\test_images'],  # path to the folder in which test images are. Can be more than one.  
        'traindir': r'D:\training_images',  # path to the folder in which training images are.  
        'upscaleimage': False,  # The model is going to upscale the given low resolution image.  
        'valdir': r'',  # path to the folder in which validation images are.  
        'workingdir': r'',  # path to the working directory. All outputs to be produced within this directory  
        'weightpath': '',  # path to model weights either for training to start with, or for test.  
    }  


def build_model(self, testmode=False):
    if testmode:
        input_size = None
    else:
        input_size = self.inputsize

    input_shape = (input_size, input_size, self.channels)

    main_input = Input(shape=input_shape, name='main_input')

    pre_block = Conv2D(16, (3, 3), kernel_initializer='glorot_uniform', activation=self.activation, padding='same')(main_input)
    pre_block = Conv2D(16, (3, 3), kernel_initializer='glorot_uniform', activation=self.activation, padding='same')(pre_block)
    pre_block = Conv2D(16, (3, 3), kernel_initializer='glorot_uniform', activation=self.activation, padding='same')(pre_block)
    pre_block = Conv2D(16, (1, 1), kernel_initializer='glorot_uniform', activation=self.activation, padding='same')(pre_block)

    upsampler_LC = UpSampling2D(self.scale, name='upsampler_locally_connected')(pre_block)
    upsampler_direct = UpSampling2D(self.scale)(main_input)

    # REPEATING BLOCKS

    block3 = concatenate([upsampler_LC, upsampler_direct])
    block3 = Conv2D(16, (3, 3), kernel_initializer='glorot_uniform', activation=self.activation, padding='same')(block3)
    block3 = Conv2D(16, (3, 3), kernel_initializer='glorot_uniform', activation=self.activation, padding='same')(block3)
    block3 = Conv2D(16, (1, 1), kernel_initializer='glorot_uniform', activation=self.activation, padding='same')(block3)

    block4 = concatenate([upsampler_LC, upsampler_direct, block3])
    block4 = Conv2D(16, (3, 3), kernel_initializer='glorot_uniform', activation=self.activation, padding='same')(block4)
    block4 = Conv2D(16, (3, 3), kernel_initializer='glorot_uniform', activation=self.activation, padding='same')(block4)
    block4 = Conv2D(16, (1, 1), kernel_initializer='glorot_uniform', activation=self.activation, padding='same')(block4)

    block5 = concatenate([upsampler_LC, upsampler_direct, block3, block4])
    block5 = Conv2D(16, (3, 3), kernel_initializer='glorot_uniform', activation=self.activation, padding='same')(block5)
    block5 = Conv2D(16, (3, 3), kernel_initializer='glorot_uniform', activation=self.activation, padding='same')(block5)
    block5 = Conv2D(16, (1, 1), kernel_initializer='glorot_uniform', activation=self.activation, padding='same')(block5)

    fourth = concatenate([upsampler_LC, upsampler_direct, block3, block4, block5])
    fourth = Conv2D(16, (3, 3), kernel_initializer='glorot_uniform', activation=self.activation, padding='same')(fourth)
    fourth = Conv2D(16, (3, 3), kernel_initializer='glorot_uniform', activation=self.activation, padding='same')(fourth)
    fourth = Conv2D(16, (1, 1), kernel_initializer='glorot_uniform', activation=self.activation, padding='same')(fourth)

    final = Conv2D(self.channels, (3, 3), kernel_initializer='glorot_uniform', activation=self.lactivation, padding='same')(fourth)

    model = Model(main_input, outputs=final)
    model.compile(Adam(self.lrate, self.decay), loss=losses.mean_squared_error)

    # model.summary()

    return model

```























More detailed examples and explanation are given in the program manual.  

### Using DeepSR as Class Object

<p align='justify'>
DeepSR can be used as a Python object from another program as well. The key point here is,
 that all parameters along with settings must be assigned to the object before doing any 
 operation with it. User need to designate each parameter/instruction as class member or methods 
 by manually. On the other hand, there is another way of doing this procedure.
 </p>
 
 <p align='justify'>
DeepSR object takes only one parameter in class construction phase for setting it up with parameters:
 args (arguments or setting, in another word). The argument, args must be a dictionary with
  key-value pairs similar to the dictionary, settings in model files. The parameters/instructions 
  in args can be taken from a file programmatically or can be written in actual python code where 
  the actual program is being coded. It is up to the user. 
</p>
 
 <p align='justify'>
DeepSR can be constructed without providing any settings in args. The class will have no any members
 or methods in such case. This user is informed about this situation in command prompt. However,
  each parameter of the program (and also build method) must still be designated in the class as members
   before they are being used for any operations. 
</p>
 
 <p align='justify'>
The following code snippet is an example for creating DeepSR object from Python scripts by 
assigning settings to class in construction stage of the class.
</p>

```python
import DeepSR  
from os.path import basename  
  
settings = \  
    {  
        'activation': 'relu', # activation function for layers is relu.
        'augment': [90, 180],  # can be any combination of [90,180,270, 'flipud', 'fliplr', 'flipudlr' ], or []  
        'backend': 'theano', # keras is going to use theano framework in processing.  
        'batchsize':9,  # number of batches  
        'channels': 1,  # color channels to be used in training . Only one channel in this case  
        'colormode': 'YCbCr',  # the color space is YCbCr. 'YCbCr' or 'RGB'  
        'crop': 0,  # do not crop from borders of images.  
        'crop_test': 0,  # do not crop from borders of images in tests.  
        'decay': 1e-6,  # learning rate decay.  
        'espatience': 5,  # Early stopping patience. Stop after 5 epochs if the performance of the model has not improved.  
        'epoch': 2,  # train the model for total 10 passes on training data.
        'gpu': 0,1 , # use the first and second GPUs in the computing system.  
        'inputsize': 33,  # size of input image patches is 33x33.  
        'lactivation': prelu. # activation for the last layer of the model is prelu.  
        'lrate': 0.001, # learning rate
        'lrpatience': 3,  # The learning rate plateau patience. The number of epochs to wait before reducing the learning rate.  
        'lrfactor': 0.5,  # Learning rate plateau factor. The ratio of decrease in learning rate value.  
        'minimumlrate': 1e-7,  # learning rate can be reduced down to a maximum of this value.  
        'modelname': basename(__file__).split('.')[0],  # modelname is the same as the name of this file.  
        'metrics': ['PSNR', 'SSIM'],  # Evaluation metrics are PSNR and SSIM.  
        'normalization': ['standard', 53.28, 40.732],  # apply standardization to input images (mean, std)  
        'outputdir': '',  # sub directories automatically created.  
        'scale': 4,  # magnification factor is 4.  
        'stride': 11,  # give a step of 11 pixels apart between patches while cropping them from images for training.  
        'target_channels': 1,  # color channels to be used in tests . Only one channel in this case  
        'target_cmode': 'RGB',  # 'YCbCr' or 'RGB'  
        'testpath': [r'D:\test_images'],  # path to the folder in which test images are. Can be more than one.  
        'traindir': r'D:\training_images',  # path to the folder in which training images are.  
        'upscaleimage': False,  # The model is going to upscale the given low resolution image.  
        'valdir': r'',  # path to the folder in which validation images are.  
        'workingdir': r'',  # path to the working directory. All outputs to be produced within this directory  
        'weightpath': '',  # path to model weights either for training to start with, or for test.  
    }  
  
DSR = DeepSR.DeepSR(settings)  # instance of DeepSR object without the build_model method.  
```

<p align='justify'>
At this point, DeepSR object was created with parameters but without the method “build_model”. 
Therefore this method must be declared in the class object in order to compose a deep learning model. 
User can write this method in the same script and assign it to the class by calling the member method 
of the DeepSR object: set_model. In the following code snippet, a sample method for constructing 
a deep learning model defined and assigned to DeepSR object by the member method.

```python
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
    x = Conv2D(32, (3, 3), activation=self.activation, padding='same')(input_img)  
    x = Conv2D(16, (1, 1), activation=self.activation, padding='same')(x)  
    x = Conv2D(8, (3, 3), activation=self.activation, padding='same')(x)  
    x = Conv2D(3, (1, 1), activation=self.activation, padding='same')(x)  
  
    # decoder  
    x = UpSampling2D((self.scale, self.scale))(x)  # upscale by the scale factor  
    x = Conv2D(8, (3, 3), activation=self.activation, padding='same')(x)  
    x = Conv2D(16, (1, 1), activation=self.activation, padding='same')(x)  
    x = Conv2D(32, (3, 3), activation=self.activation, padding='same')(x)  
    decoded = Conv2D(self.channels, (3, 3), activation=self.lactivation, padding='same')(x)  
    autoencoder = Model(input_img, decoded)  
    autoencoder.compile(Adam(self.lrate, self.decay), loss=losses.mean_squared_error)  
    return autoencoder  
```

Now, the class object is ready for further processes. A training and test procedure is being implemented below.

```python
DSR.set_model(build_model)  # set build_model function to compose a DL model in the class.  
  
DSR.epoch =1 # training will be implemented for 1 time instead of 10 as defined in settings.  
  
DSR.train()  # training procedure.  
  
# the performance of the model is evaluated below.  
DSR.test(testpath=DSR.testpath, weightpath=DSR.weightpath, saveimages=False, plot=False)  

```

To wrap all together, whole code is below.

```python
import DeepSR  
from os.path import basename  
  
settings = \  
    {  
        'augment': [90, 180],  # can be any combination of [90,180,270, 'flipud', 'fliplr', 'flipudlr' ], or []  
        'backend': 'theano', # keras is going to use theano framework in processing.  
        'batchsize':9,  # number of batches  
        'channels': 1,  # color channels to be used in training . Only one channel in this case  
        'colormode': 'YCbCr',  # the color space is YCbCr. 'YCbCr' or 'RGB'  
        'crop': 0,  # do not crop from borders of images.  
        'crop_test': 0,  # do not crop from borders of images in tests.  
        'decay': 1e-6,  # learning rate decay.  
        'espatience': 5,  # stop after 5 epochs if the performance of the model has not improved.  
        'epoch': 2,  # train the model for total 10 passes on training data.  
        'inputsize': 33,  # size of input image patches is 33x33.  
        'lrate': 0.001,  
        'lrpatience': 3,  # number of epochs to wait before reducing the learning rate.  
        'lrfactor': 0.5,  # the ratio of decrease in learning rate value.  
        'minimumlrate': 1e-7,  # learning rate can be reduced down to a maximum of this value.  
        'modelname': basename(__file__).split('.')[0],  # modelname is the same as the name of this file.  
        'metrics': ['PSNR', 'SSIM'],  # the model name is the same as the name of this file.  
        'normalization': ['standard', 53.28, 40.732],  # apply standardization to input images (mean, std)  
        'outputdir': '',  # sub directories automatically created.  
        'scale': 4,  # magnification factor is 4.  
        'stride': 11,  # give a step of 11 pixels apart between patches while cropping them from images for training.  
        'target_channels': 1,  # color channels to be used in tests . Only one channel in this case  
        'target_cmode': 'RGB',  # 'YCbCr' or 'RGB'  
        'testpath': [r'D:\test_images'],  # path to the folder in which test images are. Can be more than one.  
        'traindir': r'D:\training_images',  # path to the folder in which training images are.  
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
  
DSR.train()  # training procedure.  
  
# evaluate the performance of the model.  
DSR.test(testpath=DSR.testpath, weightpath=DSR.weightpath, saveimages=False, plot=False)  
    
```


