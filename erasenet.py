from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint
import math
import cv2
from keras.layers import Input, Dense, Dropout, Flatten, Activation, BatchNormalization, Reshape
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Lambda
from keras.layers.noise import GaussianNoise
from keras.models import Model, Sequential, Input
from keras.optimizers import Adam , SGD
from keras.activations import elu ,selu
import keras
subtract_layer = Lambda(lambda inputs: inputs[0] - inputs[1],
                        output_shape=lambda shapes: shapes[0])
'''
You need to first wait for the disccriminator to converge some before the generator starts to make modifications. It takes a while before good erasure happens
I will also try a residualnet style middle layers
'''
s=96
def createpair(size): # generate toy data
	data = np.zeros((size,size))
	circcenter=(randint(0,size),randint(0,size))
	r=randint(size//6,size//3)
	squarecenter=(randint(0,size),randint(0,size))
	squaresize=randint(size//8,size//4)
	squareoffset=(squaresize,squaresize)
	triverts=[(randint(0,size),randint(0,size)),(randint(0,size),randint(0,size)),(randint(0,size),randint(0,size))]
	cv2.circle(data,circcenter,r,(255),thickness=-1)
	cv2.rectangle(data,tuple(np.subtract(squarecenter,squareoffset)),tuple(np.add(squarecenter,squareoffset)),(255),thickness=-1)
	notri=np.copy(data)
	cv2.fillConvexPoly(data,np.array(triverts),(255))
	data=np.expand_dims(data, axis=2)
	notri=np.expand_dims(notri, axis=2)
	return data,notri
def createpairs(size,num):
	withtri=np.zeros((num,size,size,1),dtype=np.float32)
	notri=np.zeros((num,size,size,1),dtype=np.float32)	
	for i in range(num):
		c=createpair(size)
		withtri[i]=c[0]/255
		notri[i]=c[1]/255
	return withtri,notri

x=Input(shape=(s,s,1)) # build the image modification network
skips=[]
y=BatchNormalization()(x)
for i in range(1,4):
	y=Conv2D(16*(2**i),(3,3),padding='same')(y)
	y=BatchNormalization()(y)
	y=Activation('elu')(y)
	skips.append(y)
	y=Conv2D(32*(2**i),(3,3),strides=(2,2),padding='same')(y)
	y=BatchNormalization()(y)
	y=Activation('elu')(y)
for i in range(6):
	y=Dense(256)(y)
	y=BatchNormalization()(y)
	y=Activation('elu')(y)
for i in range(3,0,-1):
	#print(i)
	y=Conv2D(32*(2**i),(3,3),padding='same')(y)
	y=BatchNormalization()(y)
	y=Activation('elu')(y)
	y=UpSampling2D()(y)
	y=keras.layers.concatenate([skips[i-1],y])
	y=Conv2D(16*(2**i),(3,3),padding='same')(y)
	y=BatchNormalization()(y)
	y=Activation('elu')(y)
y=Conv2D(16,(3,3),padding='same')(y)
y=BatchNormalization()(y)
y=Activation('elu')(y)
y=Conv2D(1,(3,3),padding='same',activation='sigmoid')(y) 
diff=subtract_layer([x,y])
error=keras.layers.core.ActivityRegularization(l1=0.0001)(diff)
generator=Model(inputs=x,outputs=[y,error])
generator.compile(loss='mse', optimizer='nadam')

discriminatorinput=Input(shape=[s,s,1]) # build the discrimiator
d=Conv2D(32,(3,3),activation='elu')(discriminatorinput)
d=Conv2D(32,(3,3),activation='elu')(d)
d=BatchNormalization()(d)
d=Conv2D(64,(3,3),strides=(2,2),activation='elu')(d)
d=Conv2D(64,(3,3),activation='elu')(d)
d=BatchNormalization()(d)
d=Conv2D(64,(3,3),strides=(2,2),activation='elu')(d)
d=Conv2D(64,(3,3),activation='elu')(d)
d=BatchNormalization()(d)
d=Conv2D(128,(3,3),strides=(2,2),activation='elu')(d)
d=Conv2D(128,(3,3),activation='elu')(d)
d=BatchNormalization()(d)
d=Dense(256,activation='elu')(d)
d=BatchNormalization()(d)
d=Dense(256,activation='elu')(d)
d=BatchNormalization()(d)
d=Dense(256,activation='elu')(d)
d=BatchNormalization()(d)
d=Dense(128,activation='elu')(d)
d=Flatten()(d)
d=Dense(1,activation='sigmoid')(d)
discriminator=Model(inputs=discriminatorinput,outputs=d)
dadam=Adam(lr=.001)
discriminator.compile(loss='binary_crossentropy', optimizer='adam')

for l in discriminator.layers:
	l.trainable=False; 

ganinput=Input(shape=[s,s,1]) # build the gan
img=generator(ganinput)[0]
ganout=discriminator(img)
gan=Model(inputs=ganinput,outputs=ganout)
gan.compile(loss='binary_crossentropy', optimizer='adam')
def show(n): # show images 
	b=createpairs(s,n)
	targets=np.squeeze(b[1])
	inputs=np.squeeze(b[0])
	out=np.squeeze(generator.predict(b[0])[0])
	for i in range(n):
		plt.subplot(3,3,i+1)
		plt.imshow(inputs[i])
		plt.subplot(3,3,i+5)
		plt.imshow(out[i])
	plt.show()

for count in range(10000):
	real=np.array(createpairs(s,8)[1])
	fake=np.array(createpairs(s,8)[0])
	images=np.concatenate([real,fake])
	labels=np.concatenate([np.ones(8),np.zeros(8)])
	dloss=discriminator.train_on_batch(images,labels) # train the discriminator
	ganlabels=np.ones(16)
	fake=createpairs(s,16)[0]
	ganloss=gan.train_on_batch(fake,ganlabels) # train the generator
	print("discriminator loss",dloss,"generator loss",ganloss)
	if(count%100==0): # show every 50 iterations
		print("")
		show(4)
