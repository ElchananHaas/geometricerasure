from scipy import misc
import glob
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint
import math
from keras.layers import Input, Dense, Dropout, Flatten, Activation, BatchNormalization, Reshape
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Lambda
from keras.layers.noise import GaussianNoise
from keras.models import Model, Sequential, Input
from keras.optimizers import Adam , SGD, Adadelta
from keras.activations import elu ,selu
import keras
subtract_layer = Lambda(lambda inputs: inputs[0] - inputs[1],
                        output_shape=lambda shapes: shapes[0])
flip_sigmoid = Lambda(lambda inputs: 1-inputs)
s=104
noisescale=.01

def load(path):
	names=glob.glob(path)
	pics=[]
	for name in names:
		pic=misc.imread(name)
		pics.append(pic)
	pics=np.array(pics)
	return pics
base=load("/home/ananya/autosegment/smallbase/*")
block=load("/home/ananya/autosegment/smallblock/*")
#print(base)
#baselen=base.shape[0]
#print(baselen)
def createpairs(size,num):
	withobj=np.zeros((num,size,size,3),dtype=np.float32)
	noobj=np.zeros((num,size,size,3),dtype=np.float32)	
	for i in range(num):
		withobj[i]=block[randint(0,block.shape[0])]/255
		noobj[i]=base[randint(0,base.shape[0])]/255
	return noobj,withobj

x=Input(shape=(s,s,3))
skips=[]
y=BatchNormalization()(x)
for i in range(1,4):
	y=Conv2D(32*(2**i),(3,3),padding='same')(y)
	y=BatchNormalization()(y)
	y=Activation('elu')(y)
	skips.append(y)
	y=Conv2D(64*(2**i),(3,3),strides=(2,2),padding='same')(y)
	y=BatchNormalization()(y)
	y=Activation('elu')(y)
for i in range(4):
	y=Dense(1024)(y)
	y=BatchNormalization()(y)
	y=Activation('elu')(y)
for i in range(3,0,-1):
	#print(i)
	y=Conv2D(64*(2**i),(3,3),padding='same')(y)
	y=BatchNormalization()(y)
	y=Activation('elu')(y)
	y=UpSampling2D()(y)
	y=keras.layers.concatenate([skips[i-1],y])
	y=Conv2D(32*(2**i),(3,3),padding='same')(y)
	y=BatchNormalization()(y)
	y=Activation('elu')(y)
y=Conv2D(3,(3,3),padding='same')(y)
y=Activation('elu')(y)
diff=subtract_layer([x,y])
error=keras.layers.core.ActivityRegularization(l1=.00001)(diff) #l1=0.0001,
generator=Model(inputs=x,outputs=[y,error])
generator.compile(loss='mse', optimizer='adam')

discriminatorinput=Input(shape=[s,s,3])
d=BatchNormalization()(discriminatorinput)
d=Conv2D(64,(3,3),activation='elu')(discriminatorinput)
d=Conv2D(64,(3,3),activation='elu')(d)
d=Conv2D(64,(3,3),strides=(2,2),activation='elu')(d)
d=Conv2D(64,(3,3),activation='elu')(d)
d=Conv2D(128,(3,3),strides=(2,2),activation='elu')(d)
d=Conv2D(128,(3,3),activation='elu')(d)
d=Conv2D(256,(3,3),strides=(2,2),activation='elu')(d)
d=Conv2D(256,(3,3),activation='elu')(d)
d=Dense(256,activation='elu')(d)
d=Dense(256)(d)
d=BatchNormalization()(d)
d=Activation('elu')(d)
d=Dense(128,activation='elu')(d)
d=Flatten()(d)
d=Dense(1,activation='sigmoid')(d)
discriminator=Model(inputs=discriminatorinput,outputs=d)
dadam=Adam(lr=.001)
discriminator.compile(loss='binary_crossentropy', optimizer='adam')

for l in discriminator.layers:
	l.trainable=False; 
ganinput=Input(shape=[s,s,3])
img=generator(ganinput)[0]
ganout=discriminator(img)
gan=Model(inputs=ganinput,outputs=ganout)
gadam=Adam(lr=.001)
gan.compile(loss='binary_crossentropy', optimizer='adam')
def show(n):
	b=createpairs(s,n)
	inputs=b[1]
	out=generator.predict(b[0])[0]
	for i in range(n):
		plt.subplot(3,3,i+1)
		plt.imshow(out[i])
		plt.subplot(3,3,i+5)
		plt.imshow(inputs[i])
	plt.show()

for count in range(10000):
	real=createpairs(s,8)[1]
	fake=createpairs(s,8)[0]
	inputs=np.concatenate([real,fake])
	labels=np.concatenate([np.ones(8),np.zeros(8)])
	dloss=discriminator.train_on_batch(inputs,labels)
	ganlabels=np.ones(16)
	fake=createpairs(s,16)[0]
	ganloss=1
	ganloss=gan.train_on_batch(fake,ganlabels)
	#fake=createpairs(s,16)[0]
	#ganloss=gan.train_on_batch(fake,ganlabels)
	print("discrim loss",dloss,"generator loss",ganloss)
	if(count%300==0):
		show(4)