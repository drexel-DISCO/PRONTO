import numpy as np

import torch
import torch.nn as nn
import snntorch as snn
import snntorch.functional as func
import snntorch.utils as utils

from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader


#from lif_parameters import *
from senclib import parameters
p = parameters()

seed=p.mnist.seed
torch.manual_seed(seed)

# Define Network
class Net(nn.Module):
    #def __init__(self,):
    def __init__(self,arch=[784,128,10]):
        super().__init__()
        
        #instantiate parameters
        param = parameters()
        reset_mechanism = param.lif.reset_mechanism
        beta            = param.lif.beta
        num_steps       = 201
        # Initialize layers
        num_inputs  = arch[0]
        num_hidden  = arch[1]
        num_outputs = arch[-1]

        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):
        #print('ANUP = ',x.shape)
        #exit()
        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        #for step in range(num_steps):
        for step in range(201):
            #print(step)
            #print(x.shape)
            #exit()
            xi = x[:,step,:]
            #print(xi.shape)
            #exit()
            cur1 = self.fc1(xi)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

class fsnn():
    def __init__(self,arch=[784,128,10]):
        self.layers  = list() #list of layers
        self.lifs    = list() #list of lif neuron types
        self.mems    = list() #list of membrane potentials

        param = parameters()
        reset_mechanism = param.lif.reset_mechanism
        beta            = param.lif.beta
        vth             = param.lif.vth

        #instantiate the layers
        #reset encoding
        if reset_mechanism == 0:
            torch_reset = 'none'
        elif reset_mechanism == 1:
            torch_reset = 'subtract'
        elif reset_mechanism == 2:
            torch_reset = 'zero'
        else:
            torch_reset = 'subtract'

        self.n_layers = len(arch)   #number of layers
        for i in range(1,self.n_layers):    #for each layer
            in_sz = arch[i-1]    #input size
            out_sz = arch[i]     #output size
            layer = nn.Linear(in_sz,out_sz,bias=False) #layer shape
            lif = snn.Leaky(beta=beta,
                            threshold=vth,
                            reset_mechanism=torch_reset)    #lif neuron
            self.layers.append(layer)    #save the layer
            self.lifs.append(lif)        #save the neuron type
        #initialize the state of lif neurons
        for lif in self.lifs:
            mem = lif.init_leaky()
            self.mems.append(mem)
        print('[info] Instantiate a fully-connected snn ',arch)

    def get_weights(self):
        wts = list()
        for i,layer in enumerate(self.layers):
            wt = layer.weight.detach().numpy().transpose() * 10
            wt = np.clip(wt,-0.9375,0.9375)
            wts.append(wt)
        return wts

    def get_scale(self):
        wts = list()
        for layer in self.layers:
            wt = layer.weight.detach().numpy().transpose().flatten()
            wts += list(wt)
        return (min(wts),max(wts))

    def set_weights(self,wts):
        for i in range(self.n_layers-1):
            #get the new layer weights
            wt = wts[i].transpose()
            #extract the current set of parameters
            sd = self.layers[i].state_dict()
            #update the layer weights
            sd['weight'] = torch.Tensor(wt)
            #load the new layer weights to the model
            self.layers[i].load_state_dict(sd)
            print('[info] Loading weight(',i,'-',(i+1),'):',sd['weight'].shape)

    def simulate(self,aer,n_steps=10):
        mem_recs = [list() for mem in self.mems]   #list to record membrane potential
        spk_recs = [list() for lif in self.lifs]   #list to record spikes
        #simulate the network
        for t in range(n_steps):
            spk = aer[t]    #input
            for i,layer in enumerate(self.layers):
                cur                 = layer(spk)    #current = spike * activation
                spk,self.mems[i]    = self.lifs[i](cur,self.mems[i])#mem[t+1] = post_synaptic current + decayed vmem
                spk_recs[i].append(spk)             #record the output spikes
                mem_recs[i].append(self.mems[i])    #record the membrane potential

        #process the output for final recording
        self.torch_vmems = [torch.stack(mr) for mr in mem_recs]
        self.torch_spikes= [torch.stack(sr) for sr in spk_recs]
        #convert to numpy
        self.np_vmems   = [m.detach().numpy()[:,0,:] for m in self.torch_vmems]
        self.np_spikes  = [s.detach().numpy() for s in self.torch_spikes]

        return torch.stack(spk_recs[-1], dim=0), torch.stack(mem_recs[-1], dim=0)
