import numpy as np
import matplotlib
import librosa

## does it make a difference whether first FFT, then split or first split and then FFT. 
#first option is preferable but needs to be consistent -> min lenght of audio file!!

class AudioSampler:
    """takes from one audio file (one speaker) multiple samples of same length
    """
    
    def __init__(self, fname):
        self.fname=fname
        self.segment=260 # = length of sample
        self.f_cut=300
        self.trim=10
        self.sr=44100
        self.folder = '.'
        self.doublesampling=true # offest by half a segment n + (n-1)
        self.x = None
        
    def load_audio(self):
        #potentially long file, is slow.
        x, sr = librosa.load(self.folder+self.fname, sr=self.sr)

        #n_seg segments fit into x
        self.n_seg=x.shape[1]//self.segment
        self.residual = x.shape[1]-self.n_seg*self.segment
        self.x = x
        
        
    def _get_index_of_segment(self): 
        sample=[]
        if self.n_seg > 0:
            for n in range(self.n_seg):
                offset  =np.random.randint(0,residual//self.n_seg)+trim
                sample.append((offset+n*seg, offset+(n+1)*self.segment))
        if self.n_seg > 1 and self.doublesampling:
            for n in range(self.n_seg-1):
                offset =np.random.randint(0,residual//self.n_seg)+trim+self.segment//2
                sample.append((offset+n*seg, offset+(n+1)*self.segment))

        return sample
    
    
    def _save_as_image(self,X,number):
        #number,name=fname.split('_')[:2]
        #fn=fname.split('.')[0]
        tmp=np.expand_dims(X, axis=2)
        colordata=np.repeat(tmp,3,axis=2)
        matplotlib.image.imsave(f'audiodata/{self.fname}_{number}.png', colordata)
        #if np.random.rand() < 0.9:
        #  matplotlib.image.imsave(f'audiodata/train/{name}/{fn}.png', colordata)
        #else:
        #  matplotlib.image.imsave(f'audiodata/valid/{name}/{fn}.png', colordata)
        pass
    
    def imagify_segments(self):
        X = librosa.stft(x)
        Xdb = librosa.amplitude_to_db(abs(X))

        segments = self._get_index_of_segment()
        
        for i, (start,end) in enumerate(segments):
            Xdb=Xdb[:self.f_cut, start:end]
            Xdb-=Xdb.min()
            Xdb=Xdb/Xdb.max() #better set mean and std according to self.mean, self.std
            self._save_as_image(Xdb, number=i) 

    def main(self):
        self.load_audio()
        self.imagify_segments()
        