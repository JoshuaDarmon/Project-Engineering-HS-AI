from tqdm import tqdm #baaaaaaarre de chargement
import cv2 #analyse d'image, le best
from glob import glob #ça me permet de gérer mes dossiers directement, histoire d'aller chercher les images
from termcolor import colored, cprint #des petits messages en couleur

import os

#là j'importe tous les trucs pas drôles ( ou drôles selon à qui tu parle)
import keras
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Activation
from keras.optimizers import Adam
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np


#je crée des listes qui vont me permettre de stocker toutes le images
x_tout = []
y_tout = []


#####################################################
 #là je vais chercher les images et je les manipule #
 #pour que ma RAM n'ait pas de problèmes et que tout#
 #soit sous le même format                          #
#####################################################
print('wah des photos de gens faisant des signes chelous')
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

#ici les images sont mélangées

x_tout, y_tout = shuffle(x_tout, y_tout)

#je transforme les listes précédentes en array pour qu'elles soient lues

x_tout = np.asarray(x_tout)
y_tout = np.asarray(y_tout)

#ici je transforme les "labels" en matrice ce qui me
#permet d'associer une erreur à chaque signe


y_tout = keras.utils.to_categorical(y_tout, 5)




#####################################
#l'entraînement du modèle commence  #
#####################################


# ici la photo est stockée dans x_tout, et l'information du signe dans y_tout
# maintenant on va séparer x_tout et y_tout en train et test
# les images de train permettent d'entrainer alors que celles de test permettent de tester


x_train , x_test , y_train, y_test = train_test_split(x_tout, y_tout, test_size= 0.10)
print ("X train : " + str(x_train.shape))  #je regarde cb il y a d'images dans chaque trucs
print ("Y train : " + str(y_train.shape))
print ("X test : " + str(x_test.shape))
print ("Y test : " + str(y_test.shape))


#ça me permet de rajouter une dimension à mon array  pour dire que c'est noir et blanc
x_train = np.expand_dims(x_train,3)
x_test = np.expand_dims(x_test,3)


#ici je définis les paramètres du modèle qui s'organise en plusieurs couches

model=Sequential()
model.add(Conv2D(64,(3,3), input_shape=(100,100,1)))  #je définis le format de l'image
model.add(Activation("relu"))

#j'utilise un modèle avec des couches de convoltion car c'est le meilleures
#en analyse d'image

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))


#model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))

#ici j'ajoute une couche qui permet "d'aplatir" les images
#afin que le programme les lisent en suite de nombres
model.add(Flatten())



model.add(Dense(64,activation='relu'))
model.add (Dense(5,activation='softmax'))

#la dernière couche a 5 neurones car chacune donne la probabilité d'un résultat


model.summary()

model.compile(loss="categorical_crossentropy", #la fonction qui venait avec to_categorical
	optimizer = Adam(lr=1e-6),
	metrics=["accuracy"]) #analyse la précision du programme

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
    #epochs=30,      #peut être augmenter
    #validation_data=(x_test, y_test),   #avec quoi il va verifier
    #verbose=1)

#nuuuuu on a fini
#faut pas oublier de sauvegarder parce que sinon c'est la PLS

model.save('model_projet_SI_V2_noBG_2.h5')
