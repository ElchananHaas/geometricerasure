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
This model uses an encoder decoder model with a cheat connection between the encoder and decoder, so it can erase and readd features
'''
s=96 #s must be a multiple of 8
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

encoderin=Input(shape=(s,s,1))
cheatencoderin=Input(shape=(s,s,1))
encoder=keras.layers.concatenate([encoderin,cheatencoderin])
encoder=BatchNormalization()(encoder)
encoder=Conv2D(32,(3,3),padding='same')(encoder)
for i in range(1,4): #for loops make it easer to change network structure, and are easier to read	
	encoder=MaxPooling2D()(encoder)
	encoder=Conv2D(32*(2**i),(3,3),padding='same')(encoder)
	encoder=BatchNormalization()(encoder)
	encoder=Activation('elu')(encoder)
	encoder=Conv2D(32*(2**i),(3,3),padding='same')(encoder)
	encoder=BatchNormalization()(encoder)
	encoder=Activation('elu')(encoder)
encodermodel=Model(inputs=[encoderin,cheatencoderin],outputs=encoder)

cheatin=Input(shape=(s//8,s//8,256))
cheat=Flatten()(cheatin)
cheat=Dense(100)(cheat)
cheat=BatchNormalization()(cheat)
cheat=Activation('elu')(cheat)
cheat=Dense(s*s)(cheat)
cheat=BatchNormalization()(cheat)
cheat=Activation('elu')(cheat)
cheat=Reshape((s,s,1))(cheat)
cheatskip=Model(inputs=cheatin,outputs=cheat)

zeros=Input(shape=(s,s,1))

maindecoderin=Input(shape=(s//8,s//8,256))
decoder=maindecoderin
for i in range(3,0,-1):
	decoder=Conv2D(32*(2**i),(3,3),padding='same')(decoder)
	decoder=BatchNormalization()(decoder)
	decoder=Activation('elu')(decoder)
	decoder=Conv2D(32*(2**i),(3,3),padding='same')(decoder)
	decoder=BatchNormalization()(decoder)
	decoder=Activation('elu')(decoder)
	decoder=UpSampling2D()(decoder)
decoder=Conv2D(32,(3,3),padding='same')(decoder)
decoder=BatchNormalization()(decoder)
decoder=Activation('elu')(decoder)
decoder=Conv2D(1,(3,3),padding='same',activation='sigmoid')(decoder) 
decodermodel=Model(inputs=[maindecoderin],outputs=[decoder])

eraserin=Input(shape=(s,s,1))
enc=encodermodel([eraserin,zeros])
skipped=cheatskip(enc)
notriangle=decodermodel(enc)
reenc=encodermodel([notriangle,skipped])
reconstructed=decodermodel(reenc)
eraser=Model(inputs=[eraserin,zeros],outputs=[notriangle,reconstructed])  #binary_crossentropy
eraser.compile(loss='binary_crossentropy', optimizer='adam')

discriminatorinput=Input(shape=[s,s,1]) # build the discrimiator
d=discriminatorinput
d=Conv2D(32,(3,3))(d)
#d=BatchNormalization()(d)
d=Activation('elu')(d)
for i in range(1,4): 
	d=MaxPooling2D()(d)
	d=Conv2D(32*(2**i),(3,3))(d)
	#d=BatchNormalization()(d)
	d=Activation('elu')(d)
	d=Conv2D(32*(2**i),(3,3))(d)
	#d=BatchNormalization()(d)
	d=Activation('elu')(d)
d=Flatten()(d)
for i in range(2):
	d=Dense(512)(d)
	#d=BatchNormalization()(d)
	d=Activation('elu')(d)
d=Dense(1,activation='sigmoid')(d)
discriminator=Model(inputs=discriminatorinput,outputs=d)
discriminator.compile(loss='binary_crossentropy', optimizer='adam')


ganinput=Input(shape=[s,s,1]) # build the gan
img=eraser([ganinput,zeros])
ganout=discriminator(img[0])
gan=Model(inputs=[ganinput,zeros],outputs=[ganout,img[1]])
gan.compile(loss='binary_crossentropy', optimizer='adam')
def show(n): # show images 
	b=createpairs(s,n)
	targets=np.squeeze(b[1])
	inputs=np.squeeze(b[0])
	out=eraser.predict([b[0],z])
	outa=np.squeeze(out[0])
	outb=np.squeeze(out[1])
	for i in range(n):
		plt.subplot(4,4,i+1)
		plt.imshow(inputs[i])
		plt.subplot(4,4,i+1+n)
		plt.imshow(outa[i])
		plt.subplot(4,4,i+1+n*2)
		plt.imshow(outb[i])
	#plt.show()
	global savenum
	plt.savefig(str(savenum)+".png")
	savenum+=1
z=np.zeros(shape=(16,s,s,1))
for count in range(10000):
	c=createpairs(s,16)
	loss=eraser.train_on_batch([c[0],z],[c[1],c[0]])
	print(loss)
	if(count%100==0): # show every 50 iterations
		print("")
		show(4)
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
