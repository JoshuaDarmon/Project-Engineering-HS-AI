from tqdm import tqdm 
import cv2 
from glob import glob 
import keras
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Activation
from keras.optimizers import Adam
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np


x_tout = []
y_tout = []


print('all good')
img_porte = glob(r'dataset_SI/O_2/*')
img_lampe = glob(r'dataset_SI/L_2/*')
img_aucun_signe = glob(r'dataset_SI/aucun_bg/*')
img_store_haut = glob(r'dataset_SI/M_2/*')
img_store_bas = glob(r'dataset_SI/B_2/*')

#ajouts post ecriture

for image in tqdm(img_aucun_signe) :
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image,(100,100))
    image = image.astype("float32")
    image /=255
    x_tout.append(image)
    y_tout.append(2)



for image in tqdm(img_store_haut) :
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image,(100,100))
    image = image.astype("float32")
    image /=255
    x_tout.append(image)
    y_tout.append(3)



for image in tqdm(img_store_bas) :
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image,(100,100))
    image = image.astype("float32")
    image /=255
    x_tout.append(image)
    y_tout.append(4)



for image in tqdm(img_porte) :
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image,(100,100))
    image = image.astype("float32")
    image /=255
    x_tout.append(image)
    y_tout.append(0)



for image in tqdm(img_lampe) :
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image,(100,100))
    image = image.astype("float32")
    image /=255
    x_tout.append(image)
    y_tout.append(1)

print ('tout fonctionne')


x_tout, y_tout = shuffle(x_tout, y_tout)


x_tout = np.asarray(x_tout)
y_tout = np.asarray(y_tout)

y_tout = keras.utils.to_categorical(y_tout, 5)






#verifying shape just to be sure
x_train , x_test , y_train, y_test = train_test_split(x_tout, y_tout, test_size= 0.10)
print ("X train : " + str(x_train.shape))  
print ("Y train : " + str(y_train.shape))
print ("X test : " + str(x_test.shape))
print ("Y test : " + str(y_test.shape))



x_train = np.expand_dims(x_train,3)
x_test = np.expand_dims(x_test,3)

#model with tests to optimize

model=Sequential()
model.add(Conv2D(64,(3,3), input_shape=(100,100,1)))  #je définis le format de l'image
model.add(Activation("relu"))


model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))


#model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))


model.add(Flatten())



model.add(Dense(64,activation='relu'))
model.add (Dense(5,activation='softmax'))




model.summary()

model.compile(loss="categorical_crossentropy", 
	optimizer = Adam(lr=1e-6),
	metrics=["accuracy"]) 

model.fit(x_train,y_train,
epochs = 7,
batch_size = 32,
validation_data = (x_test,y_test),
verbose = 1)

#image_gen= ImageDataGenerator(
    #zoom_range=0.15,
#    width_shift_range=0.15,
#   height_shift_range=0.1,
#    horizontal_flip=True)

#image_gen.fit(x_train)


#model.fit_generator(image_gen.flow(x_train,y_train,batch_size = 32),
    #steps_per_epoch=len(x_train),
    #epochs=30,     
    #validation_data=(x_test, y_test),   
    #verbose=1)


model.save('model_projet_SI_V2_noBG_2.h5')
