"""

This is the definition file of DECUSR model for ultrasound super resolution.
Please refer to the article by Hakan Temiz and Hasan Şakir Bilge:
	https://ieeexplore.ieee.org/abstract/document/9078131

This file is tailored to use DECUSR with DeepSR, which eases and automates
the super-resolution-specific processes (training, test, augmenting, normalization, etc.),
for super-resolution.

For more information on how to use DeepSR, please refer to:
	https://github.com/htemiz/DeepSR

	and PyPi page:

	https://pypi.org/project/DeepSR/

For program manual please refer to:
	https://github.com/htemiz/DeepSR/blob/master/DeepSR/docs/DeepSR%20Manual.pdf


Just a basic instructions to run this model for training and test with DeepSR:
	python.exe -m DeepSR.DeepSR --modelfile <full path of this file > --train --test .... <other command arguments>


To install DeepSR:
	pip install DeepSR


"""

from keras import losses
from keras.models import Model
from keras.layers import Input,  concatenate
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.optimizers import Adam


# PARAMETER INSTRUCTION RELATED TO SCALE FACTOR #
"""
CHANGE THE FOLLOWING PARAMETERS ACCORDING TO THE SCALE
SCALE 2:
        stride=5, inputsize=16

SCALE 3:
        stride=4, inputsize=11

SCALE 4:
        stride=3, inputsize=8

SCALE 8:
        stride=1, inputsize=4

"""

# These parameters to be read by DeepSR. It will then build model based on these settings.
# These parameters can also easily be owerwritten and dynamically changed
#  by DeepSR via command line script when running

settings = \
{
"activation": "relu",
'augment':[], # must be any or all lof [90,180,270, 'flipud', 'fliplr', 'flipudlr' ]
'backend': 'tensorflow',
'batchsize':128,
'channels':1,
'colormode':'RGB', # 'YCbCr' or 'RGB'
'crop': 0,
'crop_test': 6,
'decay':1e-6,
'dilation_rate':(1,1),
'decimation': 'bicubic',
'espatience' : 50,
'epoch':50,
'inputsize':16, #
'interp_compare': 'lanczos',
'interp_up': 'bicubic',
'kernel_initializer': 'glorot_uniform',
'lrate':1e-3,
'lrpatience': 25,
'lrfactor' : 0.5,
'metrics': ["PSNR"],
'minimumlrate' : 1e-7,
'modelname':basename(__file__).split('.')[0],
'noise':'',
'normalization':['divide', '255.0'], # ['standard', "53.28741141", "40.73203139"],
'normalizeback': False,
'normalizeground':False,
'outputdir':'',
'scale':2,
'seed': 19,
'shuffle' : True,
'stride':5, # to have approx. same number of patches, use 5,4,3,1 for scales 2,3,4 and 8, respectively.
'target_channels': 1,
'target_cmode' : 'RGB',
'testpath': [r'D:\test'],
'traindir': r"D:\train",
'upscaleimage': False,
'valdir': r'D:\val',
'weightpath':'',
'workingdir': '',
}



def build_model(self, testmode=False):
	"""
	this function is read by DeepSR and assigned to it as a member method to construct
	the model for both training and test procedures, accordingly
	:param self:
	:param testmode: Bool. Determines if the model to be built for test or training
	:return: Returns the DECUSR model.
	"""
    
	# arrange the input size for running procedure: training or test
	if testmode:
		input_size = None # use the image size
	else:
		input_size = self.inputsize # use the size given in 'settings' dictionary

	input_shape = (input_size, input_size, self.channels)

	main_input = Input(shape=input_shape, name='main_input')


	# Feature extraction block (Lfeb)
	L_FEB = Conv2D(16, (3, 3), kernel_initializer='glorot_uniform', activation='relu', padding='same')(main_input)
	L_FEB = Conv2D(16, (3, 3), kernel_initializer='glorot_uniform', activation='relu', padding='same')(L_FEB)
	L_FEB = Conv2D(16, (3, 3), kernel_initializer='glorot_uniform', activation='relu', padding='same')(L_FEB)
	L_FEB = Conv2D(16, (1, 1), kernel_initializer='glorot_uniform', activation='relu', padding='same')(L_FEB)

	# Feature upscaling layer (Lfup)
	L_FUP = UpSampling2D(self.scale, name='upsampler_locally_connected')(L_FEB)

	# Direct upscaling layer (Ldup)
	L_DUP = UpSampling2D(self.scale)(main_input)

	# REPEATING BLOCKS
	RB1 = concatenate([L_FUP, L_DUP])
	RB1 = Conv2D(16, (3, 3), kernel_initializer='glorot_uniform', activation='relu', padding='same')(RB1)
	RB1 = Conv2D(16, (3, 3), kernel_initializer='glorot_uniform', activation='relu', padding='same')(RB1)
	RB1 = Conv2D(16, (1, 1), kernel_initializer='glorot_uniform', activation='relu', padding='same')(RB1)

	RB2 = concatenate([L_FUP, L_DUP, RB1])
	RB2 = Conv2D(16, (3, 3), kernel_initializer='glorot_uniform', activation='relu', padding='same')(RB2)
	RB2 = Conv2D(16, (3, 3), kernel_initializer='glorot_uniform', activation='relu', padding='same')(RB2)
	RB2 = Conv2D(16, (1, 1), kernel_initializer='glorot_uniform', activation='relu', padding='same')(RB2)

	RB3 = concatenate([L_FUP, L_DUP, RB1, RB2])
	RB3 = Conv2D(16, (3, 3), kernel_initializer='glorot_uniform', activation='relu', padding='same')(RB3)
	RB3 = Conv2D(16, (3, 3), kernel_initializer='glorot_uniform', activation='relu', padding='same')(RB3)
	RB3 = Conv2D(16, (1, 1), kernel_initializer='glorot_uniform', activation='relu', padding='same')(RB3)

	RB4 = concatenate([L_FUP, L_DUP, RB1, RB2, RB3])
	RB4 = Conv2D(16, (3, 3), kernel_initializer='glorot_uniform', activation='relu', padding='same')(RB4)
	RB4 = Conv2D(16, (3, 3), kernel_initializer='glorot_uniform', activation='relu', padding='same')(RB4)
	RB4 = Conv2D(16, (1, 1), kernel_initializer='glorot_uniform', activation='relu', padding='same')(RB4)

	# LAST LAYER
	LAST = Conv2D(self.channels, (3, 3), kernel_initializer='glorot_uniform', activation='relu', padding='same')(RB4)

	model = Model(main_input, outputs=LAST)
	model.compile(Adam(self.lrate, self.decay), loss=losses.mean_squared_error)

	return model
