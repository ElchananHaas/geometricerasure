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
You need to first wait for the discriminator to converge some before the generator starts to make modifications. It takes a while before good erasure happens
the squares aften end up somewhat rounded
I am using a patchGAN becuase it is giving far better results
Experience replay could stabalize training more
annealing the l1 loss may improve results
'''
s=96
savenum=0
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
y=Conv2D(32,(3,3),padding='same')(y)
for i in range(1,4): #for loops make it easer to change network structure, and are easier to read
	skips.append(y)	
	y=MaxPooling2D()(y)
	y=Conv2D(32*(2**i),(3,3),padding='same')(y)
	y=BatchNormalization()(y)
	y=Activation('elu')(y)
	y=Conv2D(32*(2**i),(3,3),padding='same')(y)
	y=BatchNormalization()(y)
	y=Activation('elu')(y)
for i in range(0):
	addskip=y
	y=Conv2D(256,(3,3),padding='same')(y)
	y=BatchNormalization()(y)
	y=Activation('elu')(y)
	y=Conv2D(256,(3,3),padding='same')(y)
	y=BatchNormalization()(y)
	y=Activation('elu')(y)
	y=keras.layers.Add()([addskip,y])
for i in range(3,0,-1):
	y=Conv2D(32*(2**i),(3,3),padding='same')(y)
	y=BatchNormalization()(y)
	y=Activation('elu')(y)
	y=Conv2D(32*(2**i),(3,3),padding='same')(y)
	y=BatchNormalization()(y)
	y=Activation('elu')(y)
	y=UpSampling2D()(y)
	y=keras.layers.concatenate([skips[i-1],y])
y=Conv2D(32,(3,3),padding='same')(y)
y=BatchNormalization()(y)
y=Activation('elu')(y)
y=Conv2D(1,(3,3),padding='same',activation='sigmoid')(y) 
#y=GaussianNoise(.3)(y)

diff=subtract_layer([x,y])
error=keras.layers.core.ActivityRegularization(l1=0.0000002)(diff) #much higher l1 loss and it doesn't make any changes, much lower and it fails to resemble the original image, or converge at all
gen=Model(inputs=x,outputs=[y,error])
g=Input(shape=[s,s,1])  
out=gen(g)[0]  
generator=Model(inputs=g,outputs=[out])
generator.compile(loss='mse', optimizer='adam')
discriminatorinput=Input(shape=[s,s,1]) # build the discrimiator
d=discriminatorinput
d=Conv2D(32,(3,3),padding='same')(d)
#d=BatchNormalization()(d)
d=Activation('elu')(d)
for i in range(1,4): 
	d=MaxPooling2D(padding='same')(d)
	d=Conv2D(32*(2**i),(3,3),padding='same')(d)
	#d=BatchNormalization()(d)
	d=Activation('elu')(d)
	d=Conv2D(32*(2**i),(3,3),padding='same')(d)
	#d=BatchNormalization()(d)
	d=Activation('elu')(d)
d=Conv2D(1,(1,1),padding='same')(d)
#d=BatchNormalization()(d)
d=Activation('sigmoid')(d)
discriminator=Model(inputs=discriminatorinput,outputs=d)
discriminator.compile(loss='binary_crossentropy', optimizer='adam')
discriminator.summary()
for l in discriminator.layers:
	l.trainable=False; 

ganinput=Input(shape=[s,s,1]) # build the gan
img=generator(ganinput)
ganout=discriminator(img)
gan=Model(inputs=ganinput,outputs=ganout)
gan.compile(loss='binary_crossentropy', optimizer='adam')
def show(n): # show images 
	b=createpairs(s,n)
	targets=np.squeeze(b[1])
	inputs=np.squeeze(b[0])
	out=np.squeeze(generator.predict(b[0]))
	for i in range(n):
		plt.subplot(3,3,i+1)
		plt.imshow(inputs[i])
		plt.subplot(3,3,i+5)
		plt.imshow(out[i])
	#plt.show()
	global savenum
	plt.savefig(str(savenum)+".png")
	savenum+=1
"""for count in range(10000):
	c=createpairs(s,8)
	loss=generator.train_on_batch(c[0],c[1])
	print(loss)
	if(count%100==0): # show every 50 iterations
		print("")
		show(4)"""
for count in range(10000):
	real=np.array(createpairs(s,16)[1])
	fake=generator.predict(createpairs(s,16)[0])
	images=np.concatenate([real,fake])
	labels=np.concatenate([np.ones((16,12,12,1)),np.zeros((16,12,12,1))])
	dloss=discriminator.train_on_batch(images,labels) # train the discriminator
	ganlabels=np.ones((32,12,12,1))
	fake=createpairs(s,32)[0]
	ganloss=gan.train_on_batch(fake,ganlabels) # train the generator
	print("discriminator loss",dloss,"generator loss",ganloss)
	if(count%100==0): # show every 50 iterations
		print("")
		show(4)
