import numpy as np
import cv2
import os
#from mnist_parameters import *


def loadAER(dataset='mnist',
            image_id=0,
            set_type='train',
            xdim=16,
            ydim=16,
            time_samples_per_pixel=20,
            max_spike_rate=10,
            pad_samples=1):
    #see if this image is already tested before
    #if so load it
    #otherwise, load the full dataset, return the image, and also create the image for future testing.
    load_img  = False
    dName = 'dataset/'+ dataset + '/' + set_type
    fName = dName + '/img.' + str(image_id) + '.npz'
    if (os.path.exists(fName)) and load_img:
        print('[info] Loading image ',image_id,' AER')
        with np.load(fName) as imgData:
            img     = imgData['img']
            imgSpk  = imgData['imgSpk']
            imgLbl  = imgData['imgLbl']
    else:
        print('[info] Generating image ', image_id,' AER')
        #load the whole dataset and get the required image
        img,imgSpk,imgLbl = image2AER(dataset=dataset,
                                      image_id=image_id,
                                      set_type=set_type,
                                      xdim=xdim,
                                      ydim=ydim,
                                      time_samples_per_pixel=time_samples_per_pixel,
                                      max_spike_rate=max_spike_rate,
                                      pad_samples=pad_samples)
        #first check if the directory exists, otherwise dreate one
        if not os.path.exists(dName):
            os.makedirs(dName)
        np.savez_compressed(fName,img=img,imgSpk=imgSpk,imgLbl=imgLbl)
    return img,imgSpk.astype(np.float32),imgLbl

def image2AER(dataset='mnist',
              image_id=0,
              set_type='train',
              xdim=16,
              ydim=16,
              time_samples_per_pixel=20,
              max_spike_rate=10,
              pad_samples=1):
    fname = 'dataset/'+ dataset + '.npz'
    dTypeX = 'x_' + set_type
    dTypeY = 'y_' + set_type
    with np.load(fname) as df:
        x = df[dTypeX]
        y = df[dTypeY]

    x_i = x[image_id]    #image
    y_i = y[image_id]    #label

    #reformat the image if needed
    x_i_resize      = cv2.resize(x_i, dsize=(xdim,ydim), interpolation=cv2.INTER_CUBIC)

    x_i_spk = imgenc(x_i_resize,time_samples_per_pixel,max_spike_rate,pad_samples).transpose()  #convert to spike
    xdim = x_i_spk.shape[0]
    ydim = x_i_spk.shape[1]

    return x_i,np.reshape(x_i_spk,(xdim,1,ydim)),y_i

def imgenc(img,time_samples_per_pixel,max_spike_rate,pad_samples):
    img     = img.flatten()                                 # flatten the image pixel
    spkData = np.zeros((len(img),time_samples_per_pixel),dtype=int)     # spike data for the data set
    padData = np.zeros((len(img),pad_samples),dtype=int)                # pad samples per image
    spkCnt  = [(pxl * max_spike_rate / 255) for pxl in img]             # convert each pixel value to an equivalent spike count.
    # generate random spikes for each pixel and record the spike times
    for j in range(len(img)):
        # for each pixel
        pxl         = spkCnt[j]                         # spike count for the pixel
        pxlInt      = int(np.floor(pxl))                # convert the spike count to an integer value
        rng         = np.random.default_rng()           # initialize the random generator
        ISIdist     = rng.poisson(lam=1, size=pxlInt)   # generate ISIs with poisson rate

        startTime   = 0                                 # start time of the first spike
        for isi in ISIdist:
            # for each ISI distribution
            startTime                += max(1,isi)       # generate the new start time by adding the ISI with the old spike time
            spkData[j,startTime]     = 1                 # assign a spike to the image and pixel corresponding to the start time
    
    #print(spkData.shape)
    #print(padData.shape)
    #print(np.concatenate((spkData,padData),axis=1).shape)
    #exit()
    #return spkData
    return np.concatenate((spkData,padData),axis=1)
