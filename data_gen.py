#coding:utf-8
import numpy as np
import keras
import glob
from keras.preprocessing.image import img_to_array, load_img,array_to_img
import os
import cv2

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(480,480), n_channels=3,
                 n_classes=2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self,idx):
        batch_x = []
        for i in range(self.batch_size):
            k = np.random.randint(0,len(self.list_IDs)-1)
            '''
            components = self.list_IDs[k].split()
            if 'old' in components or 'm' in components:
                    i-=1
                    continue
            '''
            if self.list_IDs[k] in batch_x:
                i-=1
                continue
            batch_x.append(self.list_IDs[k])
        print batch_x
        x_arrays,y = self.data_generation(batch_x)
        return x_arrays, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def data_generation(self,batch_x):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim[0],self.dim[1], self.n_channels))
        y = np.empty((self.batch_size,1), dtype=int)
        for i,ID in enumerate(batch_x):
                # Store sample
                #X[i,] = np.load('data/' + ID + '.npy')
                #print(ID)
                image_name = ID[0:ID.find('.bmp')+4]
                #print image_name
                train_img = load_img(image_name)     #.decode('utf-8').encode('gbk')
                train_data = img_to_array(train_img)
                if train_data.shape[0]!=480 or train_data.shape[1]!=480:
                    train_data = cv2.resize(train_data,(480,480),interpolation=cv2.INTER_CUBIC)
                X[i]=train_data
                label = ID[-2:-1]
                print train_data.max(),label
                label = int(label)
                y[i] = label
                    # Store class
        Y = keras.utils.to_categorical(y, num_classes=self.n_classes)
        print Y.shape
        return X, Y

def data_gen(IDs):
    from skimage import exposure
    while True:
        bs = 64
        batch_x = []
        for i in range(bs):
            k = np.random.randint(0,len(IDs)-1)
            '''
            components = self.list_IDs[k].split()
            if 'old' in components or 'm' in components:
                    i-=1
                    continue
            '''
            if IDs[k] in batch_x:
                i-=1
                continue
            batch_x.append(IDs[k])
        X = np.empty((bs, 320,320, 3))
        y = []
        for i,ID in enumerate(batch_x):
            # Store sample
            #X[i,] = np.load('data/' + ID + '.npy')
            words = ID.split(',')
            #print ID_
            image_name = words[0]
            label = words[1]
            #print image_name
            try:
                train_img = load_img(image_name)     #.decode('utf-8').encode('gbk')
                train_data = img_to_array(train_img)
                if train_data.shape[0]!=320 or train_data.shape[1]!=320:
                    train_data = cv2.resize(train_data,(320,320),interpolation=cv2.INTER_CUBIC)
                #p2,p98 = np.percentile(train_data,(2,98))
                #train_data = exposure.rescale_intensity(train_data,in_range=(p2,p98))
                '''
                if np.random.random() <0.4:
                    train_data = exposure.equalize_hist(train_data)
                '''
                X[i]=train_data
                if label == '':
                    i-=1
                    continue
                try:
                    label = int(label)
                except ValueError:
                    i-=1
                    continue
                y.append(label)
                if len(y)!=bs:
                    continue
                y = np.asarray(y)
                Y = keras.utils.to_categorical(y, num_classes=5)
                yield X,Y
            except IOError:
                i-=1
                continue

        
