import snntorch as snn
import torch
import torch.nn as nn
import numpy as np
import sys
import os
import pickle
import argparse


from senclib import fsnn
from senclib import mnist
from senclib import parameters
from senclib import flush_aer,flush_weights

'''
###############################################################
# instantiate parameters
###############################################################
'''
param                   = parameters()
seed                    = param.mnist.seed
input_neurons           = param.model.input_neurons
hidden_layers           = param.model.hidden_layers
hidden_layer_neurons    = param.model.hidden_layer_neurons
output_neurons          = param.model.output_neurons
layer_enc_bits          = param.hardware.layer_enc_bits
neuron_enc_bits         = param.hardware.neuron_enc_bits
fanin_enc_bits          = param.hardware.fanin_enc_bits
'''
###############################################################
'''

np.random.seed(seed)

'''
###############################################################
# process arguments
###############################################################
'''
#add arguments
parser = argparse.ArgumentParser()
parser.add_argument('-dataset','--dataset',default='mnist')
parser.add_argument('-weight','--weight',default='random')
parser.add_argument('-n_images','--n_images',default=10)
#parse arguments
args    = vars(parser.parse_args())
dataset = args['dataset']
weight  = args['weight']
n_images= int(args['n_images'])
'''
###############################################################
'''

'''
###############################################################
# model instatiation 
###############################################################
'''
#configure snntorch
snn_arch = [input_neurons]
for hl in range(hidden_layers):
    snn_arch.append(hidden_layer_neurons)
snn_arch.append(output_neurons)
#instantiate the fully-connected snn
net = fsnn(arch=snn_arch)
print('[info] define snn ',snn_arch)
'''
###############################################################
'''

'''
###############################################################
# load model parameters
###############################################################
'''
#load or generate weights
if weight == 'random':
    wts = net.get_weights()
else:
    wt_fname = weight
    if (os.path.exists(wt_fname)):
        wts = pickle.load(open(wt_fname,'rb'))
        print('[info] Loading user specified weight')
    else:
        print('[error] Weignts file not found')
        exit()
#set weights
net.set_weights(wts)
'''
###############################################################
'''

'''
###############################################################
# read input
###############################################################
'''
m = mnist(n_images=n_images)    #instantiate the dataset
aer = m.getAER()                #extract the aer
lbl = m.lbls                    #extract all labels
n_timesteps = aer.shape[0]      #number of timesteps
torch_aer = torch.tensor(aer)   #convert to torch tensor
'''
###############################################################
'''

'''
###############################################################
# simulate model
###############################################################
'''
#simulate the fsnn
print('[info] simulate the snn')
net.simulate(torch_aer,n_timesteps) #simulate for n_timesteps
net_vmems   = net.np_vmems          #all vmems
net_spikes  = net.np_spikes         #all spikes
'''
###############################################################
'''

'''
###############################################################
# save model 
###############################################################
'''
#save the model as a dictionary
nn_dict = {
    'weights'   : wts,  #save the model weights
    'input'     : aer,  #save the input
    'vmems'     : net_vmems,    #output vmems
    'spikes'    : net_spikes,   #output spikes
    'n_timesteps'   : n_timesteps,  #number of time steps
    'timesteps_per_image' : (n_timesteps / n_images), #time steps per image
    'labels'    : lbl           #all labels
}
ofname = './output/torch_out.pkl'
pickle.dump(nn_dict,open(ofname,'wb'))

#write hardware output
print('[info] flushing aer ...')
flush_aer(aer,n=n_timesteps)
#print('[info] flushing weights ...')
#flush_weights(wts)
'''
###############################################################
'''
