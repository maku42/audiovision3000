from fastai.vision import *
import librosa
from itertools import chain
import pickle as pkl
import os
import numpy as np
import matplotlib

!git clone https://github.com/Jakobovski/free-spoken-digit-dataset

#!rm -rf audiodata/*
!mkdir audiodata
!mkdir audiodata/train
!mkdir audiodata/valid

!mkdir audiodata/train/george
!mkdir audiodata/train/jackson
!mkdir audiodata/train/lucas
!mkdir audiodata/train/nicolas
!mkdir audiodata/train/theo
!mkdir audiodata/train/yweweler
!mkdir audiodata/valid/george
!mkdir audiodata/valid/jackson
!mkdir audiodata/valid/lucas
!mkdir audiodata/valid/nicolas
!mkdir audiodata/valid/theo
!mkdir audiodata/valid/yweweler

class DataHandler3000:

    def __init__(self):
        self.folder='free-spoken-digit-dataset/recordings/'
        self.namedict={'george': 2, 'jackson': 3, 'lucas': 0, 'nicolas': 5, 'theo': 4, 'yweweler': 1}
        self.fnames=fnames = os.listdir(self.folder)
        self.f_cut=300
        self.length=40
        self.sr=44100
        #self.means=[0.485, 0.456, 0.406]
        #self.stds=[0.229, 0.224, 0.225]
        self.learn=None

    def save_as_image(self,X,fname):
        number,name=fname.split('_')[:2]
        fn=fname.split('.')[0]
        tmp=np.expand_dims(X, axis=2)
        colordata=np.repeat(tmp,3,axis=2)
        if np.random.rand() < 0.9:
          matplotlib.image.imsave(f'audiodata/train/{name}/{fn}.png', colordata)
        else:
          matplotlib.image.imsave(f'audiodata/valid/{name}/{fn}.png', colordata)

    def _imagify2(self, fname, f_cut=800, length=40):
        x, sr = librosa.load(self.folder+fname, sr=self.sr)
        X = librosa.stft(x)
        Xdb = librosa.amplitude_to_db(abs(X))

        while Xdb.shape[1] < length:
            Xdb=np.hstack([Xdb,np.flip(Xdb,axis=1)])

        Xdb=Xdb[:f_cut, :length]
        Xdb-=Xdb.min()
        Xdb=Xdb/Xdb.max() #better set mean and std according to self.mean, self.std
        self.save_as_image(Xdb,fname) 
             
    def process_to_image(self):
        for i,fname in enumerate(self.fnames):
          self._imagify2(fname, f_cut=self.f_cut, length=self.length)
          if i % 100 == 0:
            print('processed images: ' , i)
            
    def train_model(self):
        path = Path('./audiodata/')
        data = ImageDataBunch.from_folder(path) 
        learn = cnn_learner(data, models.resnet18, metrics=accuracy) 
        #add data augmentation
        learn.fit_one_cycle(6)
        self.learn = learn

    def confusion_matrix(self):
        if self.learn is not None:
          interp = ClassificationInterpretation.from_learner(self.learn)
          interp.plot_confusion_matrix()

    def main(self):
        self.process_to_image()
        self.train_model()
        self.confusion_matrix()
