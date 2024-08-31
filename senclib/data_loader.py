import numpy as np
from senclib import loadAER
from senclib import parameters

class mnist():
    def __init__(self,n_images=10):
        self.imgs = list()  #list of all images
        self.aers = list()  #list of image AERs
        self.lbls = list()  #list of image labels

        param                   = parameters()
        xdim                    = param.mnist.xdim
        ydim                    = param.mnist.ydim
        time_samples_per_pixel  = param.mnist.time_samples_per_pixel
        max_spike_rate          = param.mnist.max_spike_rate
        pad_samples             = param.mnist.pad_samples

        #generate image ids
        image_ids = np.random.randint(low=0, high=10000, size = n_images)
        for image_id in image_ids:
            img,aer,lbl = loadAER(image_id=image_id,
                                  set_type='test',
                                  xdim=xdim,
                                  ydim=ydim,
                                  time_samples_per_pixel=time_samples_per_pixel,
                                  max_spike_rate=max_spike_rate,
                                  pad_samples=pad_samples)
            self.imgs.append(img)   #save the image
            self.aers.append(aer)   #save the aer
            self.lbls.append(lbl)   #save the label

        #combine all aers
        self.all_aer = self.aers[0]
        for i in range(1,n_images):
            self.all_aer = np.concatenate((self.all_aer,self.aers[i]),axis=0)
        self.samples = self.all_aer.shape[0]

    def getAER(self):
        return self.all_aer
